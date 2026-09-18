// Qwen3.8 BF16 teacher-forcing logits driver for llama.cpp.
//
// This is a small consumer of llama.cpp's public llama.h API.  It deliberately
// keeps the input format and output format independent of llama.cpp's example
// helpers so the same token manifest can be replayed by the other engines.
//
// Manifest format (one case per non-comment line):
//
//   <case-id> <positions|all> <token-id>,<token-id>,...
//
// For example:
//
//   coding-short 0,15,31 151644,872,198,...
//
// Positions are zero-based positions in the token list.  The output file for a
// case is row-major float32 with one row for each requested position, in the
// order listed in the manifest.  The metadata JSON names every output file.
//
// No llama.cpp implementation code is copied here; the source and ABI are
// pinned by the caller's build directory.

#include "llama.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr const char * kSchema = "qwen38-kld-llama-logits-v1";
constexpr std::size_t kMaxManifestBytes = 256 * 1024 * 1024;
constexpr std::size_t kMaxCases = 4096;

struct CaseSpec {
    std::string id;
    std::vector<llama_token> token_ids;
    std::vector<std::size_t> positions;
};

struct Options {
    std::string model;
    std::string manifest;
    std::string output_dir;
    std::string metadata_name = "manifest.json";
    std::string backend_log_name = "backend.log";
    std::string device_list;
    std::string tensor_split;
    std::string cache_k = "f16";
    std::string cache_v = "f16";
    std::string flash_attn = "auto";
    std::string split_mode = "layer";
    uint32_t n_ctx = 0;
    uint32_t n_batch = 2048;
    uint32_t n_ubatch = 512;
    uint32_t n_seq_max = 1;
    // Use an explicit large positive value for "all layers".  The current
    // llama.cpp C API resolves negative values to all layers too, but keeping
    // that behavior implicit could silently change if a different library
    // build is substituted for the pinned reference.
    int32_t n_gpu_layers = std::numeric_limits<int32_t>::max();
    int32_t main_gpu = 0;
    int32_t n_threads = 0;
    int32_t n_threads_batch = 0;
};

struct LogState {
    std::string text;
    std::size_t max_bytes = 16 * 1024 * 1024;
    uint64_t error_count = 0;
};

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const char raw_c : value) {
        const unsigned char c = static_cast<unsigned char>(raw_c);
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned>(c) << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    out << '"';
    return out.str();
}

template <typename T>
std::string json_array(const std::vector<T> & values) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ',';
        out << values[i];
    }
    out << ']';
    return out.str();
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

uint64_t parse_unsigned(const std::string & value, const char * name) {
    if (value.empty()) throw std::runtime_error(std::string(name) + " is empty");
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    const std::size_t consumed = end == nullptr ? 0 : static_cast<std::size_t>(end - value.c_str());
    if (errno == ERANGE || consumed != value.size()) {
        throw std::runtime_error(std::string(name) + " must be an unsigned integer");
    }
    return parsed;
}

uint32_t parse_u32(const std::string & value, const char * name) {
    const uint64_t parsed = parse_unsigned(value, name);
    if (parsed > std::numeric_limits<uint32_t>::max()) throw std::runtime_error(std::string(name) + " is out of range");
    return static_cast<uint32_t>(parsed);
}

int32_t parse_i32(const std::string & value, const char * name) {
    if (value.empty()) throw std::runtime_error(std::string(name) + " is empty");
    std::size_t consumed = 0;
    errno = 0;
    const long long parsed = std::stoll(value, &consumed, 10);
    if (errno == ERANGE || consumed != value.size() ||
        parsed < std::numeric_limits<int32_t>::min() ||
        parsed > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string(name) + " is out of range");
    }
    return static_cast<int32_t>(parsed);
}

template <typename T>
std::vector<T> parse_csv(const std::string & value, const char * name, bool allow_empty = false) {
    if (value.empty()) {
        if (allow_empty) return {};
        throw std::runtime_error(std::string(name) + " is empty");
    }
    std::vector<T> result;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        const std::size_t comma = value.find(',', begin);
        const std::size_t end = comma == std::string::npos ? value.size() : comma;
        const std::string item = trim(value.substr(begin, end - begin));
        if (item.empty()) throw std::runtime_error(std::string(name) + " contains an empty item");
        const uint64_t parsed = parse_unsigned(item, name);
        if (parsed > std::numeric_limits<T>::max()) throw std::runtime_error(std::string(name) + " item is out of range");
        result.push_back(static_cast<T>(parsed));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return result;
}

std::vector<std::string> parse_words(const std::string & value) {
    std::istringstream in(value);
    std::vector<std::string> words;
    std::string word;
    while (in >> word) words.push_back(word);
    return words;
}

std::vector<llama_token> parse_token_list(const std::string & value, std::size_t line_no) {
    // Accept both comma-separated and whitespace-separated token IDs.  A
    // comma is converted to whitespace so manifests can be generated easily.
    std::string normalized = value;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream in(normalized);
    std::vector<llama_token> result;
    std::string item;
    while (in >> item) {
        const uint64_t parsed = parse_unsigned(item, "token id");
        if (parsed > static_cast<uint64_t>(std::numeric_limits<llama_token>::max())) {
            throw std::runtime_error("manifest line " + std::to_string(line_no) + ": token id is out of range");
        }
        result.push_back(static_cast<llama_token>(parsed));
    }
    if (result.empty()) throw std::runtime_error("manifest line " + std::to_string(line_no) + ": token list is empty");
    return result;
}

std::string read_text_file(const std::string & path) {
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(path, ec);
    if (ec) throw std::runtime_error("cannot stat manifest: " + path + ": " + ec.message());
    if (bytes > kMaxManifestBytes) throw std::runtime_error("manifest is too large");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open manifest: " + path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::size_t find_json_matching(const std::string & text, std::size_t begin, char open, char close) {
    int depth = 0;
    bool quoted = false;
    bool escaped = false;
    for (std::size_t i = begin; i < text.size(); ++i) {
        const char c = text[i];
        if (quoted) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') quoted = false;
            continue;
        }
        if (c == '"') quoted = true;
        else if (c == open) ++depth;
        else if (c == close && --depth == 0) return i;
    }
    throw std::runtime_error("unterminated JSON array/object in manifest");
}

std::string json_string_field(const std::string & object, const std::vector<std::string> & names, const char * field) {
    for (const auto & name : names) {
        const std::string marker = "\"" + name + "\"";
        const std::size_t key = object.find(marker);
        if (key == std::string::npos) continue;
        const std::size_t colon = object.find(':', key + marker.size());
        if (colon == std::string::npos) break;
        const std::size_t quote = object.find('"', colon + 1);
        if (quote == std::string::npos) break;
        std::string value;
        bool escaped = false;
        for (std::size_t i = quote + 1; i < object.size(); ++i) {
            const char c = object[i];
            if (escaped) {
                switch (c) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                default: value.push_back(c); break;
                }
                escaped = false;
            } else if (c == '\\') escaped = true;
            else if (c == '"') return value;
            else value.push_back(c);
        }
        break;
    }
    throw std::runtime_error(std::string("JSON case is missing ") + field);
}

std::vector<std::size_t> json_size_array_field(const std::string & object, const std::vector<std::string> & names, const char * field, bool required) {
    for (const auto & name : names) {
        const std::string marker = "\"" + name + "\"";
        const std::size_t key = object.find(marker);
        if (key == std::string::npos) continue;
        const std::size_t colon = object.find(':', key + marker.size());
        const std::size_t open = colon == std::string::npos ? std::string::npos : object.find('[', colon + 1);
        if (open == std::string::npos) break;
        const std::size_t close = find_json_matching(object, open, '[', ']');
        std::vector<std::size_t> result;
        std::size_t i = open + 1;
        while (i < close) {
            while (i < close && (std::isspace(static_cast<unsigned char>(object[i])) || object[i] == ',')) ++i;
            if (i >= close) break;
            const std::size_t start = i;
            while (i < close && std::isdigit(static_cast<unsigned char>(object[i]))) ++i;
            if (start == i) throw std::runtime_error(std::string("JSON ") + field + " contains a non-integer");
            result.push_back(static_cast<std::size_t>(parse_unsigned(object.substr(start, i - start), field)));
            while (i < close && std::isspace(static_cast<unsigned char>(object[i]))) ++i;
            if (i < close && object[i] != ',') throw std::runtime_error(std::string("JSON ") + field + " is malformed");
        }
        return result;
    }
    if (required) throw std::runtime_error(std::string("JSON case is missing ") + field);
    return {};
}

std::vector<CaseSpec> read_json_manifest(const std::string & text) {
    const std::size_t cases_key = text.find("\"cases\"");
    if (cases_key == std::string::npos) throw std::runtime_error("JSON manifest is missing cases");
    const std::size_t cases_open = text.find('[', cases_key);
    if (cases_open == std::string::npos) throw std::runtime_error("JSON manifest cases is not an array");
    const std::size_t cases_close = find_json_matching(text, cases_open, '[', ']');
    std::vector<CaseSpec> cases;
    std::set<std::string> ids;
    std::size_t cursor = cases_open + 1;
    while (cursor < cases_close) {
        while (cursor < cases_close && (std::isspace(static_cast<unsigned char>(text[cursor])) || text[cursor] == ',')) ++cursor;
        if (cursor >= cases_close) break;
        if (text[cursor] != '{') throw std::runtime_error("JSON cases must contain objects");
        const std::size_t object_close = find_json_matching(text, cursor, '{', '}');
        if (object_close > cases_close) throw std::runtime_error("JSON case escapes cases array");
        const std::string object = text.substr(cursor, object_close - cursor + 1);
        CaseSpec spec;
        spec.id = json_string_field(object, {"id", "case_id"}, "id");
        if (spec.id.empty() || spec.id == "." || spec.id == ".." || spec.id.find('/') != std::string::npos || spec.id.find('\\') != std::string::npos) throw std::runtime_error("JSON case has invalid id");
        if (!ids.insert(spec.id).second) throw std::runtime_error("JSON manifest has duplicate case id: " + spec.id);
        const auto token_ids = json_size_array_field(object, {"input_token_ids", "input_ids", "token_ids", "tokens"}, "input_token_ids", true);
        spec.token_ids.reserve(token_ids.size());
        for (const auto token : token_ids) {
            if (token > static_cast<std::size_t>(std::numeric_limits<llama_token>::max())) throw std::runtime_error("JSON token id is out of range");
            spec.token_ids.push_back(static_cast<llama_token>(token));
        }
        spec.positions = json_size_array_field(object, {"positions"}, "positions", false);
        if (spec.positions.empty()) {
            spec.positions.resize(spec.token_ids.size());
            for (std::size_t i = 0; i < spec.positions.size(); ++i) spec.positions[i] = i;
        }
        std::set<std::size_t> seen;
        for (const auto pos : spec.positions) {
            if (pos >= spec.token_ids.size()) throw std::runtime_error("JSON position is outside token list for case: " + spec.id);
            if (!seen.insert(pos).second) throw std::runtime_error("JSON manifest has duplicate position for case: " + spec.id);
        }
        cases.push_back(std::move(spec));
        cursor = object_close + 1;
        if (cases.size() > kMaxCases) throw std::runtime_error("manifest has too many cases");
    }
    if (cases.empty()) throw std::runtime_error("JSON manifest contains no cases");
    return cases;
}

std::vector<CaseSpec> read_manifest(const Options & options) {
    const std::string contents = read_text_file(options.manifest);
    const std::size_t first = contents.find_first_not_of(" \t\r\n");
    if (first != std::string::npos && contents[first] == '{') return read_json_manifest(contents);
    std::istringstream input(contents);

    std::vector<CaseSpec> cases;
    std::set<std::string> ids;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        line = trim(line);
        if (line.empty()) continue;
        const auto words = parse_words(line);
        if (words.size() < 3) {
            throw std::runtime_error("manifest line " + std::to_string(line_no) + ": expected <case-id> <positions|all> <tokens>");
        }
        CaseSpec spec;
        spec.id = words[0];
        if (spec.id.empty() || spec.id == "." || spec.id == ".." || spec.id.find('/') != std::string::npos || spec.id.find('\\') != std::string::npos) {
            throw std::runtime_error("manifest line " + std::to_string(line_no) + ": invalid case id");
        }
        if (!ids.insert(spec.id).second) throw std::runtime_error("manifest line " + std::to_string(line_no) + ": duplicate case id");
        std::string token_text = line.substr(line.find(words[2]));
        // A token list can contain whitespace, so all fields after the second
        // are intentionally included in token_text.
        spec.token_ids = parse_token_list(token_text, line_no);
        if (lower(words[1]) == "all") {
            spec.positions.resize(spec.token_ids.size());
            for (std::size_t i = 0; i < spec.positions.size(); ++i) spec.positions[i] = i;
        } else {
            spec.positions = parse_csv<std::size_t>(words[1], "positions");
            std::set<std::size_t> seen;
            for (const auto pos : spec.positions) {
                if (pos >= spec.token_ids.size()) throw std::runtime_error("manifest line " + std::to_string(line_no) + ": position is outside token list");
                if (!seen.insert(pos).second) throw std::runtime_error("manifest line " + std::to_string(line_no) + ": duplicate position");
            }
        }
        if (spec.positions.empty()) throw std::runtime_error("manifest line " + std::to_string(line_no) + ": no positions requested");
        cases.push_back(std::move(spec));
        if (cases.size() > kMaxCases) throw std::runtime_error("manifest has too many cases");
    }
    if (cases.empty()) throw std::runtime_error("manifest contains no cases");
    return cases;
}

ggml_type parse_ggml_type(const std::string & value, const char * name) {
    const std::string type = lower(value);
    if (type == "f16" || type == "fp16") return GGML_TYPE_F16;
    if (type == "bf16") return GGML_TYPE_BF16;
    if (type == "f32" || type == "fp32") return GGML_TYPE_F32;
    if (type == "q4_0" || type == "q4-0") return GGML_TYPE_Q4_0;
    if (type == "q5_0" || type == "q5-0") return GGML_TYPE_Q5_0;
    if (type == "q8_0" || type == "q8-0") return GGML_TYPE_Q8_0;
    if (type == "q2_k" || type == "q2-k") return GGML_TYPE_Q2_K;
    if (type == "q3_k" || type == "q3-k") return GGML_TYPE_Q3_K;
    if (type == "q4_k" || type == "q4-k") return GGML_TYPE_Q4_K;
    if (type == "q5_k" || type == "q5-k") return GGML_TYPE_Q5_K;
    if (type == "q6_k" || type == "q6-k") return GGML_TYPE_Q6_K;
    throw std::runtime_error(std::string("unsupported ") + name + " type: " + value);
}

llama_split_mode parse_split_mode(const std::string & value) {
    const std::string mode = lower(value);
    if (mode == "none" || mode == "single") return LLAMA_SPLIT_MODE_NONE;
    if (mode == "layer" || mode == "layers") return LLAMA_SPLIT_MODE_LAYER;
    if (mode == "row") return LLAMA_SPLIT_MODE_ROW;
    if (mode == "tensor" || mode == "tp") return LLAMA_SPLIT_MODE_TENSOR;
    throw std::runtime_error("unsupported split mode: " + value);
}

llama_flash_attn_type parse_flash_attn(const std::string & value) {
    const std::string mode = lower(value);
    if (mode == "auto") return LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (mode == "on" || mode == "enabled" || mode == "true") return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (mode == "off" || mode == "disabled" || mode == "false") return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    throw std::runtime_error("unsupported flash attention mode: " + value);
}

void usage(const char * argv0) {
    std::cerr << "usage: " << argv0 << " --model MODEL --manifest MANIFEST --output-dir DIR [options]\n"
              << "  --n-ctx N                 context size (0: model default)\n"
              << "  --n-batch N               logical decode batch bound (default: 2048)\n"
              << "  --n-ubatch N              physical batch bound (default: 512)\n"
              << "  --n-gpu-layers N          layers on GPU; use a large value for all (default: 2147483647)\n"
              << "  --split-mode MODE         none, layer, row, tensor (default: layer)\n"
              << "  --device NAME[,NAME...]   explicit ggml backend device names\n"
              << "  --tensor-split A[,B,...]  per-device split weights\n"
              << "  --main-gpu N              main GPU index (default: 0)\n"
              << "  --cache-k TYPE            f16, bf16, q4_0, q5_0, q8_0, q2_k..q6_k\n"
              << "  --cache-v TYPE            same as --cache-k\n"
              << "  --flash-attn MODE         auto, on, off (default: auto)\n"
              << "  --threads N               CPU generation threads\n"
              << "  --threads-batch N         CPU batch threads\n"
              << "  --metadata NAME           metadata filename (default: manifest.json)\n"
              << "  --backend-log NAME        backend log filename (default: backend.log)\n"
              << "  --help                    show this message\n";
}

Options parse_options(int argc, char ** argv) {
    Options options;
    auto require_value = [&](int & index, const char * name) -> std::string {
        if (index + 1 >= argc) throw std::runtime_error(std::string(name) + " requires a value");
        return argv[++index];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (arg == "--model") options.model = require_value(i, "--model");
        else if (arg == "--manifest") options.manifest = require_value(i, "--manifest");
        else if (arg == "--output-dir") options.output_dir = require_value(i, "--output-dir");
        else if (arg == "--metadata") options.metadata_name = require_value(i, "--metadata");
        else if (arg == "--backend-log") options.backend_log_name = require_value(i, "--backend-log");
        else if (arg == "--device" || arg == "--devices") options.device_list = require_value(i, "--device");
        else if (arg == "--tensor-split") options.tensor_split = require_value(i, "--tensor-split");
        else if (arg == "--cache-k" || arg == "--cache-type-k") options.cache_k = require_value(i, "--cache-k");
        else if (arg == "--cache-v" || arg == "--cache-type-v") options.cache_v = require_value(i, "--cache-v");
        else if (arg == "--flash-attn") options.flash_attn = require_value(i, "--flash-attn");
        else if (arg == "--split-mode") options.split_mode = require_value(i, "--split-mode");
        else if (arg == "--n-ctx" || arg == "--context") options.n_ctx = parse_u32(require_value(i, "--n-ctx"), "--n-ctx");
        else if (arg == "--n-batch") options.n_batch = parse_u32(require_value(i, "--n-batch"), "--n-batch");
        else if (arg == "--n-ubatch") options.n_ubatch = parse_u32(require_value(i, "--n-ubatch"), "--n-ubatch");
        else if (arg == "--n-seq-max") options.n_seq_max = parse_u32(require_value(i, "--n-seq-max"), "--n-seq-max");
        else if (arg == "--n-gpu-layers") options.n_gpu_layers = parse_i32(require_value(i, "--n-gpu-layers"), "--n-gpu-layers");
        else if (arg == "--main-gpu") options.main_gpu = parse_i32(require_value(i, "--main-gpu"), "--main-gpu");
        else if (arg == "--threads") options.n_threads = parse_i32(require_value(i, "--threads"), "--threads");
        else if (arg == "--threads-batch") options.n_threads_batch = parse_i32(require_value(i, "--threads-batch"), "--threads-batch");
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (options.model.empty()) throw std::runtime_error("--model is required");
    if (options.manifest.empty()) throw std::runtime_error("--manifest is required");
    if (options.output_dir.empty()) throw std::runtime_error("--output-dir is required");
    if (options.n_batch == 0 || options.n_ubatch == 0 || options.n_seq_max == 0) throw std::runtime_error("batch and sequence bounds must be positive");
    if (options.n_ubatch > options.n_batch) throw std::runtime_error("--n-ubatch cannot exceed --n-batch");
    if (options.metadata_name.find('/') != std::string::npos || options.metadata_name.find('\\') != std::string::npos || options.metadata_name == "." || options.metadata_name == "..") throw std::runtime_error("invalid --metadata filename");
    if (options.backend_log_name.find('/') != std::string::npos || options.backend_log_name.find('\\') != std::string::npos || options.backend_log_name == "." || options.backend_log_name == "..") throw std::runtime_error("invalid --backend-log filename");
    return options;
}

void log_callback(ggml_log_level level, const char * text, void * user_data) {
    auto * state = static_cast<LogState *>(user_data);
    if (state == nullptr) return;
    if (level == GGML_LOG_LEVEL_ERROR) ++state->error_count;
    if (text == nullptr) return;
    const std::size_t length = std::strlen(text);
    if (state->text.size() < state->max_bytes) {
        const std::size_t keep = std::min(length, state->max_bytes - state->text.size());
        state->text.append(text, keep);
    }
    std::fwrite(text, 1, length, stderr);
}

struct DeviceInfo {
    std::string name;
    std::string description;
    std::string type;
    std::string id;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
};

std::string device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
    case GGML_BACKEND_DEVICE_TYPE_CPU: return "cpu";
    case GGML_BACKEND_DEVICE_TYPE_GPU: return "gpu";
    case GGML_BACKEND_DEVICE_TYPE_IGPU: return "igpu";
    case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accel";
    case GGML_BACKEND_DEVICE_TYPE_META: return "meta";
    }
    return "unknown";
}

std::vector<DeviceInfo> enumerate_devices() {
    std::vector<DeviceInfo> result;
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const auto dev = ggml_backend_dev_get(i);
        if (dev == nullptr) continue;
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        result.push_back({ggml_backend_dev_name(dev) ? ggml_backend_dev_name(dev) : "",
                          ggml_backend_dev_description(dev) ? ggml_backend_dev_description(dev) : "",
                          device_type_name(ggml_backend_dev_type(dev)),
                          props.device_id ? props.device_id : "",
                          free_bytes, total_bytes});
    }
    return result;
}

std::vector<ggml_backend_dev_t> select_devices(const std::string & list) {
    std::vector<ggml_backend_dev_t> result;
    if (list.empty()) return result;
    std::string normalized = list;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream input(normalized);
    std::string name;
    while (input >> name) {
        const auto dev = ggml_backend_dev_by_name(name.c_str());
        if (dev == nullptr) throw std::runtime_error("unknown backend device: " + name);
        result.push_back(dev);
    }
    if (result.empty()) throw std::runtime_error("--device selected no devices");
    return result;
}

std::vector<float> parse_tensor_split(const std::string & value, std::size_t max_devices) {
    std::vector<float> result(max_devices, 0.0f);
    if (value.empty()) return result;
    std::string normalized = value;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream input(normalized);
    std::string item;
    std::size_t index = 0;
    float sum = 0.0f;
    while (input >> item) {
        if (index >= max_devices) throw std::runtime_error("--tensor-split has too many values");
        std::size_t consumed = 0;
        const float parsed = std::stof(item, &consumed);
        if (consumed != item.size() || !std::isfinite(parsed) || parsed < 0.0f) throw std::runtime_error("--tensor-split contains an invalid value");
        result[index++] = parsed;
        sum += parsed;
    }
    if (index == 0 || sum <= 0.0f) throw std::runtime_error("--tensor-split must contain a positive value");
    return result;
}

std::string case_filename(const std::string & id) {
    std::string result;
    result.reserve(id.size() + 5);
    for (const char raw_c : id) {
        const unsigned char c = static_cast<unsigned char>(raw_c);
        if (std::isalnum(c) || c == '_' || c == '-' || c == '.') result.push_back(static_cast<char>(c));
        else result.push_back('_');
    }
    return result + ".f32";
}

// SHA-256 is kept here to bind each raw file to the exact teacher-forced
// token sequence without adding a crypto library dependency to this tiny
// driver.  Token IDs are serialized as signed llama_token values in
// little-endian uint32 form before hashing.
class Sha256 {
public:
    Sha256() : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

    void update(const uint8_t * data, std::size_t size) {
        total_bytes_ += size;
        while (size != 0) {
            const std::size_t take = std::min(size, block_.size() - block_used_);
            std::memcpy(block_.data() + block_used_, data, take);
            block_used_ += take;
            data += take;
            size -= take;
            if (block_used_ == block_.size()) {
                transform(block_.data());
                block_used_ = 0;
            }
        }
    }

    std::string final_hex() {
        const uint64_t bits = static_cast<uint64_t>(total_bytes_) * 8U;
        const uint8_t one = 0x80U;
        update(&one, 1);
        const uint8_t zero = 0;
        while (block_used_ != 56) update(&zero, 1);
        uint8_t length[8];
        for (int i = 0; i < 8; ++i) length[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
        update(length, sizeof(length));
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto word : state_) out << std::setw(8) << word;
        return out.str();
    }

private:
    static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32U - n)); }

    void transform(const uint8_t * block) {
        static constexpr uint32_t k[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };
        uint32_t w[64];
        for (unsigned i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(block[4 * i]) << 24U) |
                   (static_cast<uint32_t>(block[4 * i + 1]) << 16U) |
                   (static_cast<uint32_t>(block[4 * i + 2]) << 8U) |
                   static_cast<uint32_t>(block[4 * i + 3]);
        }
        for (unsigned i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3U);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10U);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (unsigned i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + maj;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_;
    std::array<uint8_t, 64> block_{};
    std::size_t block_used_ = 0;
    std::size_t total_bytes_ = 0;
};

std::string token_ids_sha256(const CaseSpec & spec) {
    Sha256 hash;
    for (const llama_token token : spec.token_ids) {
        const uint32_t bits = static_cast<uint32_t>(token);
        const uint8_t bytes[4] = {
            static_cast<uint8_t>(bits), static_cast<uint8_t>(bits >> 8U),
            static_cast<uint8_t>(bits >> 16U), static_cast<uint8_t>(bits >> 24U)};
        hash.update(bytes, sizeof(bytes));
    }
    return hash.final_hex();
}

void write_binary_case(const CaseSpec & spec, llama_context * ctx, int32_t n_vocab, llama_batch & batch, std::size_t offset, std::ofstream & output) {
    const int32_t rc = llama_decode(ctx, batch);
    if (rc != 0) throw std::runtime_error("llama_decode failed with code " + std::to_string(rc) + " for case " + spec.id);
    llama_synchronize(ctx);
    float * contiguous = llama_get_logits(ctx);
    if (contiguous == nullptr) throw std::runtime_error("llama_get_logits returned null for case " + spec.id);
    // Preserve the manifest's requested-position order in the raw file.  The
    // common all-position case is naturally ascending, while this also makes
    // sparse/reordered probes unambiguous from metadata alone.
    for (const std::size_t position : spec.positions) {
        if (position < offset || position >= offset + static_cast<std::size_t>(batch.n_tokens)) continue;
        const int32_t local = static_cast<int32_t>(position - offset);
        float * row = llama_get_logits_ith(ctx, local);
        if (row == nullptr) throw std::runtime_error("llama_get_logits_ith returned null for case " + spec.id);
        // Keep the contiguous pointer check as an ABI sanity check.  The
        // public ith API is the source used for the actual row copy.
        if (row != contiguous + static_cast<std::size_t>(local) * static_cast<std::size_t>(n_vocab)) {
            throw std::runtime_error("llama logits row layout changed within a batch");
        }
        for (int32_t column = 0; column < n_vocab; ++column) {
            if (!std::isfinite(row[column])) {
                throw std::runtime_error("non-finite logit at case " + spec.id + " position " + std::to_string(position) + " vocab " + std::to_string(column));
            }
        }
        output.write(reinterpret_cast<const char *>(row), static_cast<std::streamsize>(sizeof(float) * static_cast<std::size_t>(n_vocab)));
        if (!output) throw std::runtime_error("failed writing logits for case " + spec.id);
    }
}

void fill_batch(llama_batch & batch, const CaseSpec & spec, std::size_t offset, std::size_t count) {
    batch.n_tokens = static_cast<int32_t>(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t slot = i;
        batch.token[slot] = spec.token_ids[offset + i];
        batch.pos[slot] = static_cast<llama_pos>(offset + i);
        batch.n_seq_id[slot] = 1;
        batch.seq_id[slot][0] = 0;
        // Every submitted token requests logits.  This is intentional: it
        // makes ith indexing and output ordering explicit even when only a
        // sparse position subset is persisted.
        batch.logits[slot] = 1;
    }
}

std::string quote_or_empty(const char * value) {
    return json_escape(value == nullptr ? "" : value);
}

void write_metadata(const Options & options, const std::vector<CaseSpec> & cases, const std::vector<std::string> & raw_paths, const std::vector<DeviceInfo> & devices, const std::vector<std::string> & selected_names, const LogState & logs, int32_t n_vocab, const llama_model * model, const std::filesystem::path & metadata_path) {
    std::ofstream out(metadata_path);
    if (!out) throw std::runtime_error("cannot open metadata output: " + metadata_path.string());
    out << "{\n"
        << "  \"schema\": " << json_escape(kSchema) << ",\n"
        << "  \"engine\": \"llama.cpp\",\n"
        << "  \"model\": " << json_escape(options.model) << ",\n"
        << "  \"llama_version\": " << quote_or_empty(llama_version()) << ",\n"
        << "  \"model_n_params\": " << llama_model_n_params(model) << ",\n"
        << "  \"model_n_layers\": " << llama_model_n_layer(model) << ",\n"
        << "  \"vocab_size\": " << n_vocab << ",\n"
        << "  \"kv\": {\"k\": " << json_escape(options.cache_k) << ", \"v\": " << json_escape(options.cache_v) << "},\n"
        << "  \"cache_type_k\": " << json_escape(options.cache_k) << ",\n"
        << "  \"cache_type_v\": " << json_escape(options.cache_v) << ",\n"
        << "  \"flash_attn\": " << json_escape(options.flash_attn) << ",\n"
        << "  \"split_mode\": " << json_escape(options.split_mode) << ",\n"
        << "  \"chunk_size\": " << options.n_batch << ",\n"
        << "  \"n_batch\": " << options.n_batch << ",\n"
        << "  \"n_ubatch\": " << options.n_ubatch << ",\n"
        << "  \"selected_devices\": [";
    for (std::size_t i = 0; i < selected_names.size(); ++i) {
        if (i != 0) out << ',';
        out << json_escape(selected_names[i]);
    }
    out << "],\n  \"available_devices\": [\n";
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const auto & dev = devices[i];
        out << "    {\"name\": " << json_escape(dev.name)
            << ", \"description\": " << json_escape(dev.description)
            << ", \"type\": " << json_escape(dev.type)
            << ", \"id\": " << json_escape(dev.id)
            << ", \"memory_free\": " << dev.free_bytes
            << ", \"memory_total\": " << dev.total_bytes << '}';
        if (i + 1 != devices.size()) out << ',';
        out << '\n';
    }
    out << "  ],\n  \"backend_log\": " << json_escape(options.backend_log_name)
        << ",\n  \"backend_error_count\": " << logs.error_count
        << ",\n  \"cases\": [\n";
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto & spec = cases[i];
        out << "    {\"id\": " << json_escape(spec.id)
            << ", \"token_count\": " << spec.token_ids.size()
            << ", \"input_token_ids_sha256\": " << json_escape(token_ids_sha256(spec))
            << ", \"positions\": " << json_array(spec.positions)
            << ", \"shape\": [" << spec.positions.size() << ',' << n_vocab << "]"
            << ", \"dtype\": \"f32\", \"layout\": \"row-major\", \"nonfinite_count\": 0"
            << ", \"logits_file\": " << json_escape(raw_paths[i])
            << ", \"raw_path\": " << json_escape(raw_paths[i]) << '}';
        if (i + 1 != cases.size()) out << ',';
        out << '\n';
    }
    out << "  ]\n}\n";
    if (!out) throw std::runtime_error("failed writing metadata output");
}

int run(const Options & options) {
    const auto cases = read_manifest(options);
    std::size_t max_tokens = 0;
    for (const auto & spec : cases) max_tokens = std::max(max_tokens, spec.token_ids.size());
    if (options.n_ctx != 0 && options.n_ctx < max_tokens) throw std::runtime_error("--n-ctx is smaller than the longest manifest case");

    std::error_code ec;
    std::filesystem::create_directories(options.output_dir, ec);
    if (ec) throw std::runtime_error("cannot create output directory: " + ec.message());

    LogState logs;
    llama_log_set(log_callback, &logs);
    llama_backend_init();
    const auto devices = enumerate_devices();
    const auto selected_devices = select_devices(options.device_list);
    std::vector<std::string> selected_names;
    for (const auto dev : selected_devices) selected_names.emplace_back(ggml_backend_dev_name(dev) ? ggml_backend_dev_name(dev) : "");
    std::vector<ggml_backend_dev_t> selected_with_null = selected_devices;
    if (!selected_with_null.empty()) selected_with_null.push_back(nullptr);
    const auto split = parse_tensor_split(options.tensor_split, llama_max_devices());

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.n_gpu_layers;
    model_params.main_gpu = options.main_gpu;
    model_params.split_mode = parse_split_mode(options.split_mode);
    model_params.devices = selected_with_null.empty() ? nullptr : selected_with_null.data();
    model_params.tensor_split = options.tensor_split.empty() ? nullptr : split.data();
    model_params.load_mtp = false;
    llama_model * model = llama_model_load_from_file(options.model.c_str(), model_params);
    if (model == nullptr) throw std::runtime_error("llama_model_load_from_file failed");

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = options.n_ctx;
    context_params.n_batch = options.n_batch;
    context_params.n_ubatch = options.n_ubatch;
    context_params.n_seq_max = options.n_seq_max;
    context_params.n_outputs_max = options.n_batch;
    context_params.n_outputs_max_per_seq = options.n_batch;
    context_params.flash_attn_type = parse_flash_attn(options.flash_attn);
    context_params.type_k = parse_ggml_type(options.cache_k, "cache-k");
    context_params.type_v = parse_ggml_type(options.cache_v, "cache-v");
    context_params.embeddings = false;
    context_params.offload_kqv = true;
    context_params.op_offload = true;
    context_params.no_perf = true;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        llama_model_free(model);
        throw std::runtime_error("llama_init_from_model failed");
    }
    if (options.n_threads > 0 || options.n_threads_batch > 0) {
        const int32_t generation = options.n_threads > 0 ? options.n_threads : llama_n_threads(context);
        const int32_t batch = options.n_threads_batch > 0 ? options.n_threads_batch : llama_n_threads_batch(context);
        llama_set_n_threads(context, generation, batch);
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        llama_free(context);
        llama_model_free(model);
        throw std::runtime_error("model returned an invalid vocabulary size");
    }

    llama_batch batch = llama_batch_init(static_cast<int32_t>(options.n_batch), 0, static_cast<int32_t>(options.n_seq_max));
    if (batch.token == nullptr || batch.pos == nullptr || batch.n_seq_id == nullptr || batch.seq_id == nullptr || batch.logits == nullptr) {
        llama_batch_free(batch);
        llama_free(context);
        llama_model_free(model);
        throw std::runtime_error("llama_batch_init failed");
    }
    std::vector<std::string> raw_paths;
    raw_paths.reserve(cases.size());
    std::set<std::string> raw_names;
    for (const auto & spec : cases) {
        std::cerr << "[qwen38_kld_llama] case " << (raw_paths.size() + 1) << '/' << cases.size()
                  << " begin id=" << spec.id << " tokens=" << spec.token_ids.size()
                  << " positions=" << spec.positions.size() << std::endl;
        llama_memory_clear(llama_get_memory(context), true);
        const std::string filename = case_filename(spec.id);
        if (!raw_names.insert(filename).second) throw std::runtime_error("case ids collide after output filename sanitization: " + spec.id);
        const std::filesystem::path path = std::filesystem::path(options.output_dir) / filename;
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open logits output: " + path.string());
        for (std::size_t offset = 0; offset < spec.token_ids.size();) {
            const std::size_t count = std::min<std::size_t>(options.n_batch, spec.token_ids.size() - offset);
            fill_batch(batch, spec, offset, count);
            write_binary_case(spec, context, n_vocab, batch, offset, output);
            offset += count;
        }
        output.close();
        if (!output) throw std::runtime_error("failed closing logits output: " + path.string());
        raw_paths.push_back(filename);
        std::cerr << "[qwen38_kld_llama] case " << raw_paths.size() << '/' << cases.size()
                  << " complete id=" << spec.id << std::endl;
    }
    llama_batch_free(batch);
    const std::filesystem::path backend_log = std::filesystem::path(options.output_dir) / options.backend_log_name;
    {
        std::ofstream out(backend_log, std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open backend log output: " + backend_log.string());
        out << logs.text;
    }
    const std::filesystem::path metadata = std::filesystem::path(options.output_dir) / options.metadata_name;
    write_metadata(options, cases, raw_paths, devices, selected_names, logs, n_vocab, model, metadata);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse_options(argc, argv);
        return run(options);
    } catch (const std::exception & error) {
        std::cerr << "qwen38_kld_llama: ERROR: " << error.what() << '\n';
        return 2;
    }
}
