// Phase 36 Session D dedicated llama.cpp comparison wrapper.
//
// This file is an original consumer of the public llama.h API.  It does not
// copy llama.cpp implementation code.  The source lane is pinned by the
// caller to llama.cpp commit 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70
// (tag b10453), and the only accepted Session D GPU is gfx942 on the current
// MI300X VM (GPU-1228c84fe776f2f4).  The 10,001-token row intentionally uses a
// 10,001-token logical batch; this is required by llama_decode's n_batch
// contract and keeps the comparison one prefill submission like sLLM auto.

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char *kSchema = "llama-phase36-session-d-v1";
constexpr const char *kLlamaCommit =
    "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70";
constexpr const char *kLlamaTag = "b10453";
constexpr const char *kTarget = "gfx942";
constexpr const char *kGpuUuid = "GPU-1228c84fe776f2f4";
constexpr llama_token kStopA = 248046;
constexpr llama_token kStopB = 248044;
constexpr int32_t kBatchSize = 1;
constexpr int32_t kSequences = 1;
constexpr int32_t kNBatch = 10001;
constexpr int32_t kNUbatch = 512;
constexpr uint32_t kWarmups = 3;
constexpr uint32_t kMeasured = 10;
constexpr size_t kLongInput = 10001;
constexpr llama_token kLongToken = 23066;

struct CaseSpec {
  const char *name;
  size_t input_tokens;
  uint32_t output_tokens;
};

constexpr CaseSpec kCases[] = {
    {"short-odd", 17, 17},
    {"32x32", 32, 32},
    {"prefill-long", 1024, 128},
    {"decode-long", 32, 256},
    {"long-10001", kLongInput, 2},
};

struct Options {
  std::string model;
  std::string model_sha256;
  std::string row_id;
  std::string case_id;
  std::vector<llama_token> input;
  uint32_t max_new_tokens = 0;
  uint32_t warmups = kWarmups;
  uint32_t measured = kMeasured;
  int32_t n_batch = kNBatch;
  int32_t n_ubatch = kNUbatch;
  int32_t batch_size = kBatchSize;
  int32_t sequences = kSequences;
  int32_t main_gpu = 0;
};

struct Sample {
  uint64_t request_start_ns = 0;
  uint64_t prefill_submit_ns = 0;
  uint64_t prefill_complete_ns = 0;
  uint64_t first_token_ns = 0;
  std::vector<uint64_t> token_publications_ns;
  uint64_t stop_ns = 0;
  uint64_t cleanup_ns = 0;
  std::vector<llama_token> generated;
  std::vector<llama_token> visible;
  std::string stop_kind;
  llama_token stop_token = LLAMA_TOKEN_NULL;
};

struct LogState {
  std::atomic<uint32_t> error_count{0};
  mutable std::mutex mutex;
  std::string captured;
  bool overflow = false;
};

struct OffloadEvidence {
  std::string name;
  std::string description;
  size_t free_before = 0;
  size_t total_before = 0;
  size_t free_ready = 0;
  size_t total_ready = 0;
  uint32_t offloaded_layers = 0;
  uint32_t offloadable_layers = 0;
  double gpu_model_buffer_mib = 0.0;
  size_t captured_log_bytes = 0;
};

constexpr size_t kMaxLogBytes = 4 * 1024 * 1024;
using Clock = std::chrono::steady_clock;

uint64_t now_ns(const Clock::time_point &origin) {
  const auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         Clock::now() - origin)
                         .count();
  if (delta <= 0) {
    throw std::runtime_error("monotonic timestamp did not advance");
  }
  return static_cast<uint64_t>(delta);
}

uint64_t after(const Clock::time_point &origin, uint64_t previous) {
  const uint64_t current = now_ns(origin);
  if (current <= previous) {
    throw std::runtime_error("timestamp order is not strictly increasing");
  }
  return current;
}

std::string json_escape(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char ch : value) {
    switch (ch) {
    case '"': out << "\\\""; break;
    case '\\': out << "\\\\"; break;
    case '\n': out << "\\n"; break;
    case '\r': out << "\\r"; break;
    case '\t': out << "\\t"; break;
    default:
      if (ch < 0x20) {
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(ch) << std::dec << std::setfill(' ');
      } else {
        out << static_cast<char>(ch);
      }
    }
  }
  out << '"';
  return out.str();
}

template <typename T> std::string json_array(const std::vector<T> &values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << values[i];
  }
  out << ']';
  return out.str();
}

std::string json_number(double value) {
  if (!std::isfinite(value)) {
    throw std::runtime_error("derived metric is not finite");
  }
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

void capture_log(enum ggml_log_level level, const char *text, void *user_data) {
  if (user_data == nullptr) return;
  auto *state = static_cast<LogState *>(user_data);
  if (level == GGML_LOG_LEVEL_ERROR) {
    state->error_count.fetch_add(1, std::memory_order_relaxed);
  }
  if (text == nullptr) return;
  std::lock_guard<std::mutex> lock(state->mutex);
  const size_t remaining =
      kMaxLogBytes - std::min(kMaxLogBytes, state->captured.size());
  if (std::strlen(text) > remaining) {
    state->overflow = true;
  } else {
    state->captured.append(text);
  }
}

uint32_t parse_u32(const std::string &value, const char *name) {
  if (value.empty()) throw std::runtime_error(std::string(name) + " is empty");
  size_t consumed = 0;
  unsigned long parsed = 0;
  try {
    parsed = std::stoul(value, &consumed, 10);
  } catch (...) {
    throw std::runtime_error(std::string(name) + " must be an unsigned integer");
  }
  if (consumed != value.size() || parsed > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error(std::string(name) + " is out of range");
  return static_cast<uint32_t>(parsed);
}

int32_t parse_i32(const std::string &value, const char *name) {
  if (value.empty()) throw std::runtime_error(std::string(name) + " is empty");
  size_t consumed = 0;
  long parsed = 0;
  try {
    parsed = std::stol(value, &consumed, 10);
  } catch (...) {
    throw std::runtime_error(std::string(name) + " must be a decimal integer");
  }
  if (consumed != value.size() || parsed < std::numeric_limits<int32_t>::min() ||
      parsed > std::numeric_limits<int32_t>::max())
    throw std::runtime_error(std::string(name) + " is out of range");
  return static_cast<int32_t>(parsed);
}

std::vector<llama_token> parse_tokens(const std::string &value) {
  if (value.empty()) throw std::runtime_error("--input-token-ids is empty");
  std::vector<llama_token> result;
  size_t begin = 0;
  while (begin < value.size()) {
    const size_t comma = value.find(',', begin);
    const size_t end = comma == std::string::npos ? value.size() : comma;
    if (end == begin) throw std::runtime_error("empty input token ID");
    size_t consumed = 0;
    long long parsed = 0;
    try {
      parsed = std::stoll(value.substr(begin, end - begin), &consumed, 10);
    } catch (...) {
      throw std::runtime_error("input token ID must be decimal");
    }
    if (consumed != end - begin || parsed < 0 ||
        static_cast<unsigned long long>(parsed) >
            static_cast<unsigned long long>(std::numeric_limits<llama_token>::max()))
      throw std::runtime_error("input token ID is outside llama_token range");
    result.push_back(static_cast<llama_token>(parsed));
    if (comma == std::string::npos) break;
    begin = comma + 1;
    if (begin == value.size()) throw std::runtime_error("empty input token ID");
  }
  return result;
}

void require_value(int argc, char **argv, int &index, const char *name,
                   std::string &value) {
  if (index + 1 >= argc) throw std::runtime_error(std::string(name) + " requires a value");
  value = argv[++index];
}

const CaseSpec *find_case(const std::string &name) {
  for (const auto &spec : kCases) if (name == spec.name) return &spec;
  return nullptr;
}

Options parse_options(int argc, char **argv) {
  Options options;
  bool model = false, sha = false, row = false, case_id = false, input = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    std::string value;
    if (arg == "--model") { require_value(argc, argv, i, "--model", options.model); model = true; }
    else if (arg == "--model-sha256") { require_value(argc, argv, i, "--model-sha256", options.model_sha256); sha = true; }
    else if (arg == "--row-id") { require_value(argc, argv, i, "--row-id", options.row_id); row = true; }
    else if (arg == "--case-id") { require_value(argc, argv, i, "--case-id", options.case_id); case_id = true; }
    else if (arg == "--input-token-ids") { require_value(argc, argv, i, "--input-token-ids", value); options.input = parse_tokens(value); input = true; }
    else if (arg == "--max-new-tokens") { require_value(argc, argv, i, arg.c_str(), value); options.max_new_tokens = parse_u32(value, arg.c_str()); }
    else if (arg == "--warmup-requests") { require_value(argc, argv, i, arg.c_str(), value); options.warmups = parse_u32(value, arg.c_str()); }
    else if (arg == "--measured-requests") { require_value(argc, argv, i, arg.c_str(), value); options.measured = parse_u32(value, arg.c_str()); }
    else if (arg == "--n-batch") { require_value(argc, argv, i, arg.c_str(), value); options.n_batch = parse_i32(value, arg.c_str()); }
    else if (arg == "--n-ubatch") { require_value(argc, argv, i, arg.c_str(), value); options.n_ubatch = parse_i32(value, arg.c_str()); }
    else if (arg == "--batch-size") { require_value(argc, argv, i, arg.c_str(), value); options.batch_size = parse_i32(value, arg.c_str()); }
    else if (arg == "--sequences") { require_value(argc, argv, i, arg.c_str(), value); options.sequences = parse_i32(value, arg.c_str()); }
    else if (arg == "--main-gpu") { require_value(argc, argv, i, arg.c_str(), value); options.main_gpu = parse_i32(value, arg.c_str()); }
    else if (arg == "--benchmark-schema-version") { require_value(argc, argv, i, arg.c_str(), value); if (value != kSchema) throw std::runtime_error("schema version is stale"); }
    else throw std::runtime_error("unknown argument: " + arg);
  }
  if (!model || !sha || !row || !case_id || !input)
    throw std::runtime_error("model, model-sha256, row-id, case-id, and input-token-ids are required");
  if (options.model_sha256.size() != 64 ||
      options.model_sha256.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::runtime_error("model-sha256 must be 64 hexadecimal characters");
  const CaseSpec *spec = find_case(options.case_id);
  if (spec == nullptr) throw std::runtime_error("case is outside the closed Session D set");
  if (options.input.size() != spec->input_tokens || options.max_new_tokens != spec->output_tokens)
    throw std::runtime_error("input or output count does not match the closed Session D recipe");
  if (options.case_id == "long-10001" &&
      std::any_of(options.input.begin(), options.input.end(), [](llama_token token) { return token != kLongToken; }))
    throw std::runtime_error("long-10001 requires token ID 23066 repeated 10001 times");
  if (options.warmups != kWarmups || options.measured != kMeasured ||
      options.batch_size != kBatchSize || options.sequences != kSequences ||
      options.main_gpu != 0 || options.n_batch != kNBatch || options.n_ubatch != kNUbatch)
    throw std::runtime_error("Session D protocol drifted (3+10, batch 1, n_batch 10001, n_ubatch 512)");
  if (options.row_id.empty()) throw std::runtime_error("row-id is empty");
  return options;
}

void validate_visibility() {
  const char *visible = std::getenv("ROCR_VISIBLE_DEVICES");
  if (visible == nullptr || std::string(visible) != kGpuUuid)
    throw std::runtime_error("ROCR_VISIBLE_DEVICES is not the exact MI300X UUID");
  for (const char *name : {"HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"})
    if (std::getenv(name) != nullptr) throw std::runtime_error(std::string(name) + " must be unset");
}

void fill_batch(llama_batch &batch, const std::vector<llama_token> &tokens,
                llama_pos first_position, bool logits_last) {
  if (tokens.empty() || tokens.size() > kNBatch)
    throw std::runtime_error("batch exceeds Session D n_batch");
  batch.n_tokens = static_cast<int32_t>(tokens.size());
  for (int32_t i = 0; i < batch.n_tokens; ++i) {
    batch.token[i] = tokens[static_cast<size_t>(i)];
    batch.pos[i] = first_position + i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = logits_last && i == batch.n_tokens - 1 ? 1 : 0;
  }
}

void decode_or_fail(llama_context *context, llama_batch &batch, const char *stage) {
  const int result = llama_decode(context, batch);
  if (result != 0) throw std::runtime_error(std::string(stage) + " llama_decode returned " + std::to_string(result));
}

bool is_stop(llama_token token) { return token == kStopA || token == kStopB; }

Sample run_request(llama_context *context, const Options &options,
                   const Clock::time_point &origin) {
  Sample sample;
  sample.request_start_ns = now_ns(origin);
  llama_memory_t memory = llama_get_memory(context);
  if (memory == nullptr) throw std::runtime_error("context has no memory object");
  llama_memory_clear(memory, true);
  llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
  if (sampler == nullptr) throw std::runtime_error("greedy sampler chain initialization failed");
  llama_sampler *greedy = llama_sampler_init_greedy();
  if (greedy == nullptr) { llama_sampler_free(sampler); throw std::runtime_error("greedy sampler initialization failed"); }
  llama_sampler_chain_add(sampler, greedy);
  llama_sampler_reset(sampler);
  sample.prefill_submit_ns = after(origin, sample.request_start_ns);

  llama_batch prefill = llama_batch_init(static_cast<int32_t>(options.input.size()), 0, kSequences);
  fill_batch(prefill, options.input, 0, true);
  decode_or_fail(context, prefill, "prefill");
  llama_synchronize(context);
  sample.prefill_complete_ns = after(origin, sample.prefill_submit_ns);
  const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(context)));
  if (vocab <= 0) { llama_batch_free(prefill); llama_sampler_free(sampler); throw std::runtime_error("empty model vocabulary"); }

  llama_token token = llama_sampler_sample(sampler, context, -1);
  if (token < 0 || token >= vocab) { llama_batch_free(prefill); llama_sampler_free(sampler); throw std::runtime_error("prefill token is out of range"); }
  sample.generated.push_back(token);
  if (is_stop(token)) { sample.stop_kind = "stop_token"; sample.stop_token = token; }
  else { sample.visible.push_back(token); llama_sampler_accept(sampler, token); }
  sample.first_token_ns = after(origin, sample.prefill_complete_ns);
  if (!sample.stop_kind.empty()) sample.stop_ns = after(origin, sample.first_token_ns);
  llama_batch_free(prefill);

  while (sample.stop_kind.empty() && sample.generated.size() < options.max_new_tokens) {
    const llama_pos position = static_cast<llama_pos>(options.input.size() + sample.visible.size() - 1);
    llama_batch decode = llama_batch_init(1, 0, kSequences);
    std::vector<llama_token> one{sample.visible.back()};
    fill_batch(decode, one, position, true);
    decode_or_fail(context, decode, "decode");
    llama_synchronize(context);
    token = llama_sampler_sample(sampler, context, -1);
    if (token < 0 || token >= vocab) { llama_batch_free(decode); llama_sampler_free(sampler); throw std::runtime_error("decode token is out of range"); }
    sample.generated.push_back(token);
    if (is_stop(token)) { sample.stop_kind = "stop_token"; sample.stop_token = token; }
    else { sample.visible.push_back(token); llama_sampler_accept(sampler, token); }
    const uint64_t previous = sample.token_publications_ns.empty() ? sample.first_token_ns : sample.token_publications_ns.back();
    sample.token_publications_ns.push_back(after(origin, previous));
    if (!sample.stop_kind.empty()) sample.stop_ns = after(origin, sample.token_publications_ns.back());
    llama_batch_free(decode);
  }
  if (sample.stop_kind.empty()) {
    sample.stop_kind = "max_new_tokens";
    const uint64_t previous = sample.token_publications_ns.empty() ? sample.first_token_ns : sample.token_publications_ns.back();
    sample.stop_ns = after(origin, previous);
  }
  llama_sampler_reset(sampler);
  llama_sampler_free(sampler);
  llama_memory_clear(memory, true);
  sample.cleanup_ns = after(origin, sample.stop_ns);
  return sample;
}

uint64_t positive_delta(uint64_t end, uint64_t begin, const char *name) {
  if (end <= begin) throw std::runtime_error(std::string(name) + " timestamps are not ordered");
  return end - begin;
}

std::string sample_json(const Sample &sample, const Options &options, uint32_t index) {
  const uint64_t ttft = positive_delta(sample.first_token_ns, sample.request_start_ns, "TTFT");
  const uint64_t prefill = positive_delta(sample.prefill_complete_ns, sample.prefill_submit_ns, "prefill");
  const uint64_t e2e = positive_delta(sample.cleanup_ns, sample.request_start_ns, "E2E");
  std::vector<uint64_t> tpot;
  uint64_t previous = sample.first_token_ns;
  for (uint64_t publication : sample.token_publications_ns) { tpot.push_back(positive_delta(publication, previous, "TPOT")); previous = publication; }
  const uint64_t decode_tokens = sample.generated.size() > 0 ? sample.generated.size() - 1 : 0;
  const bool has_decode = !sample.token_publications_ns.empty();
  const uint64_t decode_ns = has_decode ? positive_delta(sample.token_publications_ns.back(), sample.first_token_ns, "decode") : 0;
  std::ostringstream out;
  out << "{\"sample_index\":" << index << ",\"events\":{\"request_start_ns\":" << sample.request_start_ns
      << ",\"prefill_submit_ns\":" << sample.prefill_submit_ns << ",\"prefill_complete_ns\":" << sample.prefill_complete_ns
      << ",\"first_token_ns\":" << sample.first_token_ns << ",\"token_publications_ns\":" << json_array(sample.token_publications_ns)
      << ",\"stop_ns\":" << sample.stop_ns << ",\"cleanup_complete_ns\":" << sample.cleanup_ns << "}"
      << ",\"tokens\":{\"input_token_ids\":" << json_array(options.input) << ",\"generated_token_ids\":" << json_array(sample.generated)
      << ",\"visible_token_ids\":" << json_array(sample.visible) << ",\"stop_token_ids_fed_back\":[],\"bos_inserted\":false}"
      << ",\"stop\":{\"version\":1,\"kind\":" << json_escape(sample.stop_kind) << ",\"token_id\":"
      << (sample.stop_token == LLAMA_TOKEN_NULL ? "null" : std::to_string(sample.stop_token)) << "}"
      << ",\"derived\":{\"ttft_ns\":" << ttft << ",\"prefill_ns\":" << prefill
      << ",\"prefill_tokens_per_second\":" << json_number(options.input.size() * 1e9 / static_cast<double>(prefill))
      << ",\"tpot_ns\":" << json_array(tpot) << ",\"decode_tokens\":" << decode_tokens
      << ",\"decode_ns\":" << (has_decode ? std::to_string(decode_ns) : "null")
      << ",\"decode_tokens_per_second\":" << (has_decode ? json_number(decode_tokens * 1e9 / static_cast<double>(decode_ns)) : "null")
      << ",\"e2e_ns\":" << e2e << "}"
      << ",\"audit\":{\"prefill_logits_index\":" << options.input.size() - 1
      << ",\"prefill_logits_position\":" << options.input.size() - 1 << ",\"decode_first_position\":" << options.input.size()
      << ",\"stop_tokens_not_fed_back\":true,\"memory_reset_before\":true,\"memory_reset_after\":true}}";
  return out.str();
}

bool same_tokens(const Sample &left, const Sample &right) {
  return left.generated == right.generated && left.visible == right.visible && left.stop_kind == right.stop_kind && left.stop_token == right.stop_token;
}

OffloadEvidence observe_offload(const LogState &logs, ggml_backend_dev_t device,
                                size_t free_before, size_t total_before,
                                size_t free_ready, size_t total_ready) {
  OffloadEvidence evidence;
  const char *name = ggml_backend_dev_name(device);
  const char *description = ggml_backend_dev_description(device);
  if (name == nullptr || description == nullptr || std::string(name).empty()) throw std::runtime_error("selected GPU has no name");
  evidence.name = name; evidence.description = description; evidence.free_before = free_before; evidence.total_before = total_before; evidence.free_ready = free_ready; evidence.total_ready = total_ready;
  { std::lock_guard<std::mutex> lock(logs.mutex); evidence.captured_log_bytes = logs.captured.size();
    if (logs.overflow || evidence.captured_log_bytes == 0) throw std::runtime_error("llama log capture is empty or overflowed");
    const std::regex layers(R"(offloaded\s+([0-9]+)\/([0-9]+)\s+layers\s+to\s+GPU)");
    std::smatch match;
    if (!std::regex_search(logs.captured, match, layers) || match.size() != 3) throw std::runtime_error("GPU layer offload was not observed");
    evidence.offloaded_layers = parse_u32(match[1].str(), "offloaded layers"); evidence.offloadable_layers = parse_u32(match[2].str(), "offloadable layers");
    if (evidence.offloaded_layers == 0 || evidence.offloaded_layers != evidence.offloadable_layers) throw std::runtime_error("partial GPU offload observed");
    const std::regex buffer(R"(load_tensors:\s+([^\s]+)\s+model buffer size =\s+([0-9]+(?:\.[0-9]+)?) MiB)");
    for (auto it = std::sregex_iterator(logs.captured.begin(), logs.captured.end(), buffer); it != std::sregex_iterator(); ++it)
      if ((*it)[1].str() == evidence.name) evidence.gpu_model_buffer_mib += std::stod((*it)[2].str());
  }
  if (evidence.gpu_model_buffer_mib <= 0.0 || !std::isfinite(evidence.gpu_model_buffer_mib)) throw std::runtime_error("selected GPU model buffer was not observed");
  if (total_before == 0 || total_ready != total_before || free_ready >= free_before) throw std::runtime_error("GPU memory did not decrease after model load");
  return evidence;
}

std::string run(const Options &options) {
  validate_visibility();
  const auto origin = Clock::now();
  LogState logs;
  llama_log_set(capture_log, &logs);
  llama_backend_init();
  if (!llama_supports_gpu_offload()) { llama_backend_free(); throw std::runtime_error("GPU offload is unavailable"); }
  ggml_backend_dev_t selected = nullptr;
  size_t gpu_count = 0;
  for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_t device = ggml_backend_dev_get(i);
    if (device != nullptr && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) { selected = device; ++gpu_count; }
  }
  if (gpu_count != 1 || selected == nullptr) { llama_backend_free(); throw std::runtime_error("exactly one visible GPU is required"); }
  size_t free_before = 0, total_before = 0;
  ggml_backend_dev_memory(selected, &free_before, &total_before);
  ggml_backend_dev_t devices[] = {selected, nullptr};
  const uint64_t load_start = now_ns(origin);
  llama_model_params model_params = llama_model_default_params();
  model_params.devices = devices; model_params.n_gpu_layers = -1; model_params.split_mode = LLAMA_SPLIT_MODE_NONE; model_params.main_gpu = 0;
  model_params.load_mode = LLAMA_LOAD_MODE_MMAP; model_params.check_tensors = true; model_params.load_mtp = false;
  llama_model *model = llama_model_load_from_file(options.model.c_str(), model_params);
  if (model == nullptr) { llama_backend_free(); throw std::runtime_error("llama model load failed"); }
  if (llama_model_ftype(model) != LLAMA_FTYPE_MOSTLY_BF16) { llama_model_free(model); llama_backend_free(); throw std::runtime_error("loaded model is not BF16"); }
  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = static_cast<uint32_t>(options.input.size() + options.max_new_tokens);
  context_params.n_batch = static_cast<uint32_t>(options.n_batch); context_params.n_ubatch = static_cast<uint32_t>(options.n_ubatch);
  context_params.n_seq_max = 1; context_params.n_outputs_max = 1; context_params.offload_kqv = true; context_params.op_offload = true;
  context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO; context_params.type_k = GGML_TYPE_F16; context_params.type_v = GGML_TYPE_F16;
  llama_context *context = llama_init_from_model(model, context_params);
  if (context == nullptr) { llama_model_free(model); llama_backend_free(); throw std::runtime_error("llama context initialization failed"); }
  const uint64_t model_ready = now_ns(origin);
  size_t free_ready = 0, total_ready = 0; ggml_backend_dev_memory(selected, &free_ready, &total_ready);
  OffloadEvidence offload = observe_offload(logs, selected, free_before, total_before, free_ready, total_ready);
  std::vector<Sample> warmups, measured;
  try {
    for (uint32_t i = 0; i < options.warmups; ++i) warmups.push_back(run_request(context, options, origin));
    for (uint32_t i = 0; i < options.measured; ++i) measured.push_back(run_request(context, options, origin));
    if (measured.empty()) throw std::runtime_error("no measured samples");
    for (const auto &sample : warmups) if (!same_tokens(sample, measured.front())) throw std::runtime_error("warmup token sequence differs");
    for (const auto &sample : measured) if (!same_tokens(sample, measured.front())) throw std::runtime_error("measured token sequence differs");
  } catch (...) { llama_free(context); llama_model_free(model); llama_backend_free(); throw; }
  // Observe object release while the device handle is still valid, then close
  // the backend.  The HIP allocator may retain a process-local cache until
  // llama_backend_free(), so the parent runner owns the authoritative
  // post-process sysfs baseline comparison.
  llama_free(context); llama_model_free(model);
  size_t free_after_objects = 0, total_after_objects = 0;
  ggml_backend_dev_memory(selected, &free_after_objects, &total_after_objects);
  llama_backend_free();
  if (logs.error_count.load(std::memory_order_relaxed) != 0) throw std::runtime_error("llama emitted an error");
  std::ostringstream out;
  out << "{\"schema_version\":" << json_escape(kSchema) << ",\"record_kind\":\"result\",\"state\":\"PASS\""
      << ",\"llama\":{\"commit\":" << json_escape(kLlamaCommit) << ",\"tag\":" << json_escape(kLlamaTag) << "}"
      << ",\"model\":{\"path\":" << json_escape(options.model) << ",\"sha256\":" << json_escape(options.model_sha256) << ",\"format\":\"GGUF\",\"weights\":\"BF16\",\"kv\":\"F16\"}"
      << ",\"target\":{\"exact\":" << json_escape(kTarget)
      << ",\"gpu_uuid\":\"GPU-1228c84fe776f2f4\",\"logical_device_index\":0}"
      << ",\"row_id\":" << json_escape(options.row_id) << ",\"case_id\":" << json_escape(options.case_id)
      << ",\"input_token_ids\":" << json_array(options.input)
      << ",\"protocol\":{\"batch_size\":1,\"sequences\":1,\"warmup_requests\":3,\"measured_requests\":10,\"max_new_tokens\":" << options.max_new_tokens
      << ",\"n_ctx\":" << options.input.size() + options.max_new_tokens << ",\"n_batch\":10001,\"n_ubatch\":512,\"n_gpu_layers\":-1,\"split_mode\":\"none\",\"main_gpu\":0,\"offload_kqv\":true,\"op_offload\":true,\"greedy\":true,\"stop_token_ids\":[248046,248044],\"bos_inserted\":false}"
      << ",\"model_lifecycle\":{\"load_count\":1,\"context_count\":1,\"resident_reused\":true,\"load_start_ns\":" << load_start << ",\"model_ready_ns\":" << model_ready << "}"
      << ",\"warmups\":{\"count\":3,\"samples\":[";
  for (size_t i = 0; i < warmups.size(); ++i) { if (i != 0) out << ','; out << sample_json(warmups[i], options, static_cast<uint32_t>(i)); }
  out << "]},\"measured\":{\"count\":10,\"samples\":[";
  for (size_t i = 0; i < measured.size(); ++i) { if (i != 0) out << ','; out << sample_json(measured[i], options, static_cast<uint32_t>(kWarmups + i)); }
  out << "]},\"offload_evidence\":{\"gpu_offload_supported\":true,\"visible_gpu_device_count\":1,\"selected_device\":{\"name\":" << json_escape(offload.name) << ",\"description\":" << json_escape(offload.description) << ",\"type\":\"GPU\"},\"requested\":{\"n_gpu_layers\":-1,\"split_mode\":\"none\",\"main_gpu\":0,\"offload_kqv\":true,\"op_offload\":true},\"observed\":{\"offloaded_layers\":" << offload.offloaded_layers << ",\"offloadable_layers\":" << offload.offloadable_layers << ",\"gpu_model_buffer_mib\":" << json_number(offload.gpu_model_buffer_mib) << ",\"device_memory\":{\"free_before_bytes\":" << offload.free_before << ",\"total_before_bytes\":" << offload.total_before << ",\"free_model_ready_bytes\":" << offload.free_ready << ",\"total_model_ready_bytes\":" << offload.total_ready << ",\"observed_decrease_bytes\":" << offload.free_before - offload.free_ready << "},\"captured_log_bytes\":" << offload.captured_log_bytes << "}}"
      << ",\"cleanup\":{\"request_memory_reset\":true,\"backend_release_completed\":true,\"cleanup_failures\":0,\"free_after_object_release_bytes\":" << free_after_objects << ",\"total_after_object_release_bytes\":" << total_after_objects << "}"
      << ",\"audit\":{\"sample_equality\":true,\"stop_tokens_not_fed_back\":true,\"full_gpu_offload\":true,\"errors\":[]}}";
  return out.str();
}

void print_help(const char *program) {
  std::cout << "usage: " << program << " --model PATH --model-sha256 SHA256 --row-id ID --case-id CASE --input-token-ids CSV --max-new-tokens N\n"
            << "  CASE: short-odd(17/17), 32x32(32/32), prefill-long(1024/128), decode-long(32/256), long-10001(10001/2)\n"
            << "  --warmup-requests 3 --measured-requests 10 --batch-size 1 --sequences 1 --n-batch 10001 --n-ubatch 512 --main-gpu 0\n"
            << "  --benchmark-schema-version llama-phase36-session-d-v1\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) { print_help(argv[0]); return 0; }
    const Options options = parse_options(argc, argv);
    std::cout << run(options) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "llama-phase36-session-d: " << error.what() << '\n';
    return 2;
  }
}
