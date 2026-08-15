// Phase 5 P3 dedicated llama.cpp comparison wrapper.
//
// This is an original consumer of the pinned public llama.h API.  The pinned
// source was inspected for API contracts, but no implementation source is
// copied into this file.

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

constexpr const char *kSchema = "llama-phase5-v1";
constexpr const char *kLlamaCommit = "f5919bf458ef190468b5c329bb293f8a54a1e69c";
constexpr const char *kModelSha256 =
    "636158bd8a217374134cc2455aa40603f7579366fda0f0f5efcbf8bcba37c045";
constexpr llama_token kStopA = 248046;
constexpr llama_token kStopB = 248044;
constexpr int32_t kBatchSize = 1;
constexpr int32_t kSequences = 1;
constexpr int32_t kNBatch = 2048;
constexpr int32_t kNUbatch = 512;
constexpr uint32_t kWarmups = 3;
constexpr uint32_t kMeasured = 10;

struct Options {
  std::string model;
  std::string model_sha256;
  std::string target;
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
  std::vector<uint64_t> later_token_publications_ns;
  uint64_t stop_ns = 0;
  uint64_t cleanup_ns = 0;
  std::vector<llama_token> generated;
  std::vector<llama_token> visible;
  std::string stop_kind;
  llama_token stop_token = LLAMA_TOKEN_NULL;
};

struct LogState {
  std::atomic<uint32_t> error_count{0};
  std::mutex mutex;
  std::string captured;
  bool overflow = false;
};

struct OffloadEvidence {
  std::string device_name;
  std::string device_description;
  uint64_t memory_free_before_bytes = 0;
  uint64_t memory_total_before_bytes = 0;
  uint64_t memory_free_ready_bytes = 0;
  uint64_t memory_total_ready_bytes = 0;
  uint32_t offloaded_layers = 0;
  uint32_t offloadable_layers = 0;
  double gpu_model_buffer_mib = 0.0;
  uint64_t captured_log_bytes = 0;
};

constexpr size_t kMaxCapturedLogBytes = 4 * 1024 * 1024;

uint32_t parse_u32(const std::string &value, const char *name);

using Clock = std::chrono::steady_clock;

uint64_t now_ns(const Clock::time_point &origin) {
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           Clock::now() - origin)
                           .count();
  if (elapsed < 0) {
    throw std::runtime_error("monotonic clock moved backwards");
  }
  return static_cast<uint64_t>(elapsed);
}

uint64_t after(const Clock::time_point &origin, uint64_t previous) {
  const uint64_t current = now_ns(origin);
  if (current <= previous) {
    throw std::runtime_error(
        "timestamp/publication order is not strictly increasing");
  }
  return current;
}

std::string json_escape(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char ch : value) {
    switch (ch) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
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

template <typename T>
std::string json_integer_array(const std::vector<T> &values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << values[i];
  }
  out << ']';
  return out.str();
}

std::string json_u64_array(const std::vector<uint64_t> &values) {
  return json_integer_array(values);
}

std::string json_number(double value) {
  if (!std::isfinite(value)) {
    throw std::runtime_error("non-finite derived metric");
  }
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

void capture_log(enum ggml_log_level level, const char *text, void *user_data) {
  // The control plane requires an empty stderr on a PASS.  llama_log_set is
  // the pinned public API for replacing all future llama/ggml logging. Keep
  // errors observable without emitting them on the PASS path.
  if (user_data == nullptr) {
    return;
  }
  auto *state = static_cast<LogState *>(user_data);
  if (level == GGML_LOG_LEVEL_ERROR) {
    state->error_count.fetch_add(1, std::memory_order_relaxed);
  }
  if (text != nullptr) {
    std::lock_guard<std::mutex> lock(state->mutex);
    const size_t length = std::strlen(text);
    if (length > kMaxCapturedLogBytes -
                     std::min(state->captured.size(), kMaxCapturedLogBytes)) {
      state->overflow = true;
    } else {
      state->captured.append(text, length);
    }
  }
}

OffloadEvidence
observed_offload(const LogState &log_state, ggml_backend_dev_t device,
                 size_t memory_free_before, size_t memory_total_before,
                 size_t memory_free_ready, size_t memory_total_ready) {
  OffloadEvidence evidence;
  const char *name = ggml_backend_dev_name(device);
  const char *description = ggml_backend_dev_description(device);
  if (name == nullptr || description == nullptr || std::string(name).empty() ||
      std::string(description).empty()) {
    throw std::runtime_error(
        "selected GPU device has no observable name or description");
  }
  evidence.device_name = name;
  evidence.device_description = description;
  evidence.memory_free_before_bytes = memory_free_before;
  evidence.memory_total_before_bytes = memory_total_before;
  evidence.memory_free_ready_bytes = memory_free_ready;
  evidence.memory_total_ready_bytes = memory_total_ready;
  evidence.captured_log_bytes = log_state.captured.size();
  if (log_state.overflow || evidence.captured_log_bytes == 0) {
    throw std::runtime_error(
        "llama backend log capture is empty or exceeded its bound");
  }
  if (memory_total_before == 0 || memory_total_ready != memory_total_before ||
      memory_free_ready >= memory_free_before) {
    throw std::runtime_error(
        "GPU device memory did not show a model-ready allocation increase");
  }

  const std::regex layer_pattern(
      R"(load_tensors: offloaded ([0-9]+)/([0-9]+) layers to GPU)");
  std::smatch match;
  if (!std::regex_search(log_state.captured, match, layer_pattern) ||
      match.size() != 3) {
    throw std::runtime_error(
        "llama logs have no parseable GPU layer-offload observation");
  }
  evidence.offloaded_layers =
      parse_u32(match[1].str(), "observed offloaded layers");
  evidence.offloadable_layers =
      parse_u32(match[2].str(), "observed offloadable layers");
  if (evidence.offloaded_layers == 0 ||
      evidence.offloaded_layers != evidence.offloadable_layers) {
    throw std::runtime_error("llama did not report full GPU layer offload");
  }

  const std::regex buffer_pattern(
      R"(load_tensors:\s+([^\s]+) model buffer size =\s+([0-9]+(?:\.[0-9]+)?) MiB)");
  bool found_gpu_buffer = false;
  for (auto iterator =
           std::sregex_iterator(log_state.captured.begin(),
                                log_state.captured.end(), buffer_pattern);
       iterator != std::sregex_iterator(); ++iterator) {
    if ((*iterator)[1].str() == evidence.device_name) {
      evidence.gpu_model_buffer_mib += std::stod((*iterator)[2].str());
      found_gpu_buffer = true;
    }
  }
  if (!found_gpu_buffer || !std::isfinite(evidence.gpu_model_buffer_mib) ||
      evidence.gpu_model_buffer_mib <= 0.0) {
    throw std::runtime_error(
        "llama logs have no positive selected-GPU model buffer observation");
  }
  return evidence;
}

const char *expected_uuid(const std::string &target) {
  if (target == "gfx1030") {
    return "GPU-76a08c022586fed6";
  }
  if (target == "gfx1201") {
    return "GPU-a8e9ddefa2d60f55";
  }
  if (target == "gfx942") {
    return "GPU-cb0412d4d88cfa69";
  }
  throw std::runtime_error("target must be gfx1030, gfx1201, or gfx942");
}

bool is_stop(llama_token token) { return token == kStopA || token == kStopB; }

uint32_t parse_u32(const std::string &value, const char *name) {
  if (value.empty()) {
    throw std::runtime_error(std::string(name) + " must not be empty");
  }
  size_t consumed = 0;
  unsigned long parsed = 0;
  try {
    parsed = std::stoul(value, &consumed, 10);
  } catch (...) {
    throw std::runtime_error(std::string(name) +
                             " must be an unsigned decimal integer");
  }
  if (consumed != value.size() ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(std::string(name) + " is out of range");
  }
  return static_cast<uint32_t>(parsed);
}

int32_t parse_i32(const std::string &value, const char *name) {
  if (value.empty()) {
    throw std::runtime_error(std::string(name) + " must not be empty");
  }
  size_t consumed = 0;
  long parsed = 0;
  try {
    parsed = std::stol(value, &consumed, 10);
  } catch (...) {
    throw std::runtime_error(std::string(name) + " must be a decimal integer");
  }
  if (consumed != value.size() ||
      parsed < std::numeric_limits<int32_t>::min() ||
      parsed > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error(std::string(name) + " is out of range");
  }
  return static_cast<int32_t>(parsed);
}

std::vector<llama_token> parse_tokens(const std::string &value) {
  if (value.empty()) {
    throw std::runtime_error("--input-token-ids must not be empty");
  }
  std::vector<llama_token> result;
  size_t start = 0;
  while (start <= value.size()) {
    const size_t comma = value.find(',', start);
    const size_t end = comma == std::string::npos ? value.size() : comma;
    if (end == start) {
      throw std::runtime_error("--input-token-ids contains an empty element");
    }
    const int64_t token = std::stoll(value.substr(start, end - start));
    if (token < 0 || token > std::numeric_limits<llama_token>::max()) {
      throw std::runtime_error("input token ID is outside llama_token range");
    }
    result.push_back(static_cast<llama_token>(token));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return result;
}

void require_value(int argc, char **argv, int &index, const char *name,
                   std::string &output) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string(name) + " requires a value");
  }
  output = argv[++index];
}

Options parse_options(int argc, char **argv) {
  Options options;
  bool saw_model = false;
  bool saw_sha = false;
  bool saw_target = false;
  bool saw_row = false;
  bool saw_case = false;
  bool saw_input = false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    std::string value;
    if (argument == "--model") {
      if (saw_model)
        throw std::runtime_error("duplicate --model");
      require_value(argc, argv, i, "--model", options.model);
      saw_model = true;
    } else if (argument == "--model-sha256") {
      if (saw_sha)
        throw std::runtime_error("duplicate --model-sha256");
      require_value(argc, argv, i, "--model-sha256", options.model_sha256);
      saw_sha = true;
    } else if (argument == "--target") {
      if (saw_target)
        throw std::runtime_error("duplicate --target");
      require_value(argc, argv, i, "--target", options.target);
      saw_target = true;
    } else if (argument == "--row-id") {
      if (saw_row)
        throw std::runtime_error("duplicate --row-id");
      require_value(argc, argv, i, "--row-id", options.row_id);
      saw_row = true;
    } else if (argument == "--case-id") {
      if (saw_case)
        throw std::runtime_error("duplicate --case-id");
      require_value(argc, argv, i, "--case-id", options.case_id);
      saw_case = true;
    } else if (argument == "--input-token-ids") {
      if (saw_input)
        throw std::runtime_error("duplicate --input-token-ids");
      require_value(argc, argv, i, "--input-token-ids", value);
      options.input = parse_tokens(value);
      saw_input = true;
    } else if (argument == "--max-new-tokens") {
      require_value(argc, argv, i, "--max-new-tokens", value);
      options.max_new_tokens = parse_u32(value, "--max-new-tokens");
    } else if (argument == "--warmup-requests") {
      require_value(argc, argv, i, "--warmup-requests", value);
      options.warmups = parse_u32(value, "--warmup-requests");
    } else if (argument == "--measured-requests") {
      require_value(argc, argv, i, "--measured-requests", value);
      options.measured = parse_u32(value, "--measured-requests");
    } else if (argument == "--n-batch") {
      require_value(argc, argv, i, "--n-batch", value);
      options.n_batch = parse_i32(value, "--n-batch");
    } else if (argument == "--n-ubatch") {
      require_value(argc, argv, i, "--n-ubatch", value);
      options.n_ubatch = parse_i32(value, "--n-ubatch");
    } else if (argument == "--batch-size") {
      require_value(argc, argv, i, "--batch-size", value);
      options.batch_size = parse_i32(value, "--batch-size");
    } else if (argument == "--sequences") {
      require_value(argc, argv, i, "--sequences", value);
      options.sequences = parse_i32(value, "--sequences");
    } else if (argument == "--main-gpu") {
      require_value(argc, argv, i, "--main-gpu", value);
      options.main_gpu = parse_i32(value, "--main-gpu");
    } else if (argument == "--benchmark-schema-version") {
      require_value(argc, argv, i, "--benchmark-schema-version", value);
      if (value != kSchema)
        throw std::runtime_error("schema version is stale");
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (!saw_model || !saw_sha || !saw_target || !saw_row || !saw_case ||
      !saw_input) {
    throw std::runtime_error("model, model-sha256, target, row-id, case-id, "
                             "and input-token-ids are required");
  }
  if (options.model_sha256 != kModelSha256) {
    throw std::runtime_error(
        "model SHA-256 is not the locked Qwen3.5-4B BF16 GGUF identity");
  }
  if (options.batch_size != kBatchSize || options.sequences != kSequences ||
      options.main_gpu != 0 || options.n_batch != kNBatch ||
      options.n_ubatch != kNUbatch) {
    throw std::runtime_error(
        "batch, sequence, main_gpu, n_batch, or n_ubatch contract drifted");
  }
  if (options.warmups != kWarmups || options.measured != kMeasured ||
      options.max_new_tokens == 0) {
    throw std::runtime_error(
        "warmup, measured, or output budget contract drifted");
  }
  (void)expected_uuid(options.target);
  const std::vector<std::pair<const char *, size_t>> cases = {
      {"minimum", 1},        {"short-odd", 17},     {"boundary-255", 255},
      {"boundary-256", 256}, {"boundary-257", 257}, {"prefill-long", 1024},
      {"32x32", 32},        {"decode-long", 32},
  };
  bool known_case = false;
  for (const auto &item : cases) {
    if (options.case_id == item.first) {
      known_case = true;
      if (options.input.size() != item.second) {
        throw std::runtime_error(
            "input token count does not match the closed direct recipe");
      }
    }
  }
  if (!known_case) {
    throw std::runtime_error("case is outside the closed Phase 5 set");
  }
  if (options.row_id.empty()) {
    throw std::runtime_error("row ID must not be empty");
  }
  return options;
}

void validate_visibility(const Options &options) {
  const char *visible = std::getenv("ROCR_VISIBLE_DEVICES");
  if (visible == nullptr ||
      std::string(visible) != expected_uuid(options.target)) {
    throw std::runtime_error(
        "ROCR_VISIBLE_DEVICES is not the exact target UUID");
  }
  for (const char *name :
       {"HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}) {
    if (std::getenv(name) != nullptr) {
      throw std::runtime_error(std::string(name) + " must be unset");
    }
  }
}

void fill_batch(llama_batch &batch, const std::vector<llama_token> &tokens,
                llama_pos first_position) {
  if (tokens.empty() ||
      tokens.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("batch token count is outside the supported range");
  }
  // llama_batch_init allocates capacity but initializes n_tokens to zero.
  // The caller allocates this batch with exactly tokens.size() entries, then
  // this function publishes the number of populated entries to llama_decode.
  batch.n_tokens = static_cast<int32_t>(tokens.size());
  for (int32_t i = 0; i < batch.n_tokens; ++i) {
    batch.token[i] = tokens[static_cast<size_t>(i)];
    batch.pos[i] = first_position + i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = (i == batch.n_tokens - 1) ? 1 : 0;
  }
}

void decode_or_fail(llama_context *context, llama_batch &batch,
                    const char *stage) {
  const int32_t result = llama_decode(context, batch);
  if (result != 0) {
    throw std::runtime_error(std::string(stage) + " llama_decode returned " +
                             std::to_string(result));
  }
}

Sample run_request(llama_context *context, const Options &options,
                   const Clock::time_point &origin, uint32_t sample_index) {
  Sample sample;
  sample.request_start_ns = now_ns(origin);

  llama_memory_t memory = llama_get_memory(context);
  if (memory == nullptr) {
    throw std::runtime_error("context has no memory object");
  }
  // true clears metadata and data for hybrid/KV/recurrent memory.  This is
  // deliberately at both request boundaries; no request state is carried.
  llama_memory_clear(memory, true);
  llama_sampler *sampler =
      llama_sampler_chain_init(llama_sampler_chain_default_params());
  if (sampler == nullptr) {
    throw std::runtime_error("greedy sampler chain initialization failed");
  }
  llama_sampler *greedy = llama_sampler_init_greedy();
  if (greedy == nullptr) {
    llama_sampler_free(sampler);
    throw std::runtime_error("greedy sampler initialization failed");
  }
  llama_sampler_chain_add(sampler, greedy);
  llama_sampler_reset(sampler);

  sample.prefill_submit_ns = after(origin, sample.request_start_ns);

  llama_batch prefill = llama_batch_init(
      static_cast<int32_t>(options.input.size()), 0, kSequences);
  fill_batch(prefill, options.input, 0);
  decode_or_fail(context, prefill, "prefill");
  llama_synchronize(context);
  sample.prefill_complete_ns = after(origin, sample.prefill_submit_ns);

  const int32_t vocab_size =
      llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(context)));
  if (vocab_size <= 0) {
    llama_batch_free(prefill);
    llama_sampler_free(sampler);
    throw std::runtime_error("model vocabulary is empty");
  }
  llama_token token = llama_sampler_sample(sampler, context, -1);
  if (token < 0 || token >= vocab_size) {
    llama_batch_free(prefill);
    llama_sampler_free(sampler);
    throw std::runtime_error("prefill logits produced an out-of-range token");
  }
  sample.generated.push_back(token);
  if (is_stop(token)) {
    sample.stop_kind = "stop_token";
    sample.stop_token = token;
  } else {
    sample.visible.push_back(token);
    llama_sampler_accept(sampler, token);
  }
  sample.first_token_ns = after(origin, sample.prefill_complete_ns);
  if (!sample.stop_kind.empty()) {
    sample.stop_ns = after(origin, sample.first_token_ns);
  }

  llama_batch_free(prefill);
  while (sample.stop_kind.empty() &&
         sample.generated.size() < options.max_new_tokens) {
    const llama_pos position = static_cast<llama_pos>(
        options.input.size() + sample.visible.size() - 1);
    std::vector<llama_token> one_token = {sample.visible.back()};
    llama_batch decode = llama_batch_init(1, 0, kSequences);
    fill_batch(decode, one_token, position);
    decode_or_fail(context, decode, "decode");
    llama_synchronize(context);
    token = llama_sampler_sample(sampler, context, -1);
    if (token < 0 || token >= vocab_size) {
      llama_batch_free(decode);
      llama_sampler_free(sampler);
      throw std::runtime_error("decode logits produced an out-of-range token");
    }
    sample.generated.push_back(token);
    if (is_stop(token)) {
      sample.stop_kind = "stop_token";
      sample.stop_token = token;
    } else {
      sample.visible.push_back(token);
      llama_sampler_accept(sampler, token);
    }
    const uint64_t previous_publication =
        sample.later_token_publications_ns.empty()
            ? sample.first_token_ns
            : sample.later_token_publications_ns.back();
    const uint64_t publication = after(origin, previous_publication);
    sample.later_token_publications_ns.push_back(publication);
    if (!sample.stop_kind.empty()) {
      sample.stop_ns = after(origin, publication);
    }
    llama_batch_free(decode);
  }
  if (sample.stop_kind.empty()) {
    sample.stop_kind = "max_new_tokens";
    sample.stop_ns =
        after(origin, sample.generated.empty()
                          ? sample.first_token_ns
                          : (sample.later_token_publications_ns.empty()
                                 ? sample.first_token_ns
                                 : sample.later_token_publications_ns.back()));
  }
  llama_sampler_reset(sampler);
  llama_sampler_free(sampler);
  llama_memory_clear(memory, true);
  sample.cleanup_ns = after(origin, sample.stop_ns);
  (void)sample_index;
  return sample;
}

uint64_t difference(uint64_t end, uint64_t start, const char *label) {
  if (end <= start) {
    throw std::runtime_error(std::string(label) +
                             " timestamp is not strictly increasing");
  }
  return end - start;
}

std::string sample_json(const Sample &sample, const Options &options,
                        uint32_t sample_index) {
  const uint64_t ttft =
      difference(sample.first_token_ns, sample.request_start_ns, "TTFT");
  const uint64_t prefill = difference(sample.prefill_complete_ns,
                                      sample.prefill_submit_ns, "prefill");
  const uint64_t e2e =
      difference(sample.cleanup_ns, sample.request_start_ns, "E2E");
  std::vector<uint64_t> tpot;
  uint64_t previous = sample.first_token_ns;
  for (const uint64_t publication : sample.later_token_publications_ns) {
    tpot.push_back(difference(publication, previous, "TPOT"));
    previous = publication;
  }
  const uint64_t decode_tokens = sample.generated.size() - 1;
  std::string decode_rate = "null";
  if (!sample.later_token_publications_ns.empty()) {
    const uint64_t decode_window =
        difference(sample.later_token_publications_ns.back(),
                   sample.first_token_ns, "decode");
    decode_rate = json_number(static_cast<double>(decode_tokens) * 1e9 /
                              static_cast<double>(decode_window));
  }
  std::ostringstream out;
  out << "{"
      << "\"sample_index\":" << sample_index << ",\"events\":{";
  out << "\"request_start_ns\":" << sample.request_start_ns
      << ",\"prefill_submit_ns\":" << sample.prefill_submit_ns
      << ",\"prefill_complete_ns\":" << sample.prefill_complete_ns
      << ",\"first_token_ns\":" << sample.first_token_ns
      << ",\"later_token_publications_ns\":"
      << json_u64_array(sample.later_token_publications_ns)
      << ",\"stop_ns\":" << sample.stop_ns
      << ",\"cleanup_complete_ns\":" << sample.cleanup_ns << "}"
      << ",\"tokens\":{"
      << "\"input_token_ids\":" << json_integer_array(options.input)
      << ",\"generated_token_ids\":" << json_integer_array(sample.generated)
      << ",\"visible_token_ids\":" << json_integer_array(sample.visible)
      << ",\"stop_token_ids_fed_back\":[]"
      << ",\"bos_inserted\":false}"
      << ",\"stop\":{\"version\":1,\"kind\":" << json_escape(sample.stop_kind)
      << ",\"token_id\":"
      << (sample.stop_token == LLAMA_TOKEN_NULL
              ? "null"
              : std::to_string(sample.stop_token))
      << "}"
      << ",\"derived\":{\"ttft_ns\":" << ttft << ",\"prefill_ns\":" << prefill
      << ",\"prefill_tokens_per_second\":"
      << json_number(static_cast<double>(options.input.size()) * 1e9 /
                     static_cast<double>(prefill))
      << ",\"tpot_ns\":" << json_u64_array(tpot)
      << ",\"decode_tokens\":" << decode_tokens
      << ",\"decode_tokens_per_second\":" << decode_rate
      << ",\"e2e_ns\":" << e2e << "}"
      << ",\"audit\":{\"full_memory_reset_before\":true,\"full_memory_reset_"
         "after\":true"
      << ",\"sampler_reset_before\":true,\"sampler_reset_after\":true"
      << ",\"prefill_logits_index\":" << (options.input.size() - 1)
      << ",\"prefill_logits_position\":" << (options.input.size() - 1)
      << ",\"decode_logits_index\":0"
      << ",\"decode_first_position\":" << options.input.size()
      << ",\"stop_tokens_not_fed_back\":true,\"early_error_count\":0,"
         "\"errors\":[]}"
      << "}";
  return out.str();
}

std::string samples_json(const std::vector<Sample> &samples,
                         const Options &options, uint32_t first_index) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < samples.size(); ++i) {
    if (i != 0)
      out << ',';
    out << sample_json(samples[i], options,
                       first_index + static_cast<uint32_t>(i));
  }
  out << ']';
  return out.str();
}

bool same_tokens(const Sample &left, const Sample &right) {
  return left.generated == right.generated && left.visible == right.visible &&
         left.stop_kind == right.stop_kind &&
         left.stop_token == right.stop_token;
}

std::string run(const Options &options) {
  validate_visibility(options);
  const auto origin = Clock::now();
  LogState log_state;
  llama_log_set(capture_log, &log_state);
  llama_backend_init();
  if (!llama_supports_gpu_offload()) {
    llama_backend_free();
    throw std::runtime_error("llama build does not support GPU offload");
  }
  std::vector<ggml_backend_dev_t> gpu_devices;
  for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
    ggml_backend_dev_t device = ggml_backend_dev_get(index);
    if (device != nullptr &&
        ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) {
      gpu_devices.push_back(device);
    }
  }
  if (gpu_devices.size() != 1) {
    llama_backend_free();
    throw std::runtime_error("exactly one visible HIP GPU device is required");
  }
  ggml_backend_dev_t selected_gpu = gpu_devices.front();
  if (ggml_backend_dev_type(selected_gpu) != GGML_BACKEND_DEVICE_TYPE_GPU) {
    llama_backend_free();
    throw std::runtime_error(
        "selected llama backend device is not a discrete GPU");
  }
  size_t memory_free_before = 0;
  size_t memory_total_before = 0;
  ggml_backend_dev_memory(selected_gpu, &memory_free_before,
                          &memory_total_before);
  gpu_devices.push_back(nullptr);

  const uint64_t load_start = now_ns(origin);
  llama_model_params model_params = llama_model_default_params();
  model_params.devices = gpu_devices.data();
  model_params.n_gpu_layers = -1;
  model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
  model_params.main_gpu = 0;
  model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
  model_params.check_tensors = true;
  llama_model *model =
      llama_model_load_from_file(options.model.c_str(), model_params);
  if (model == nullptr) {
    llama_backend_free();
    throw std::runtime_error("llama model load failed");
  }
  const auto free_model = [&]() { llama_model_free(model); };
  if (llama_model_ftype(model) != LLAMA_FTYPE_MOSTLY_BF16) {
    free_model();
    llama_backend_free();
    throw std::runtime_error("loaded GGUF is not mostly BF16");
  }
  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx =
      static_cast<uint32_t>(options.input.size() + options.max_new_tokens);
  context_params.n_batch = static_cast<uint32_t>(options.n_batch);
  context_params.n_ubatch = static_cast<uint32_t>(options.n_ubatch);
  context_params.n_seq_max = 1;
  context_params.n_outputs_max = 1;
  context_params.offload_kqv = true;
  context_params.op_offload = true;
  context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
  llama_context *context = llama_init_from_model(model, context_params);
  if (context == nullptr) {
    free_model();
    llama_backend_free();
    throw std::runtime_error("llama context initialization failed");
  }
  const uint64_t model_ready = now_ns(origin);
  size_t memory_free_ready = 0;
  size_t memory_total_ready = 0;
  ggml_backend_dev_memory(selected_gpu, &memory_free_ready,
                          &memory_total_ready);
  const OffloadEvidence offload = observed_offload(
      log_state, selected_gpu, memory_free_before, memory_total_before,
      memory_free_ready, memory_total_ready);
  std::vector<Sample> warmups;
  std::vector<Sample> measured;
  try {
    for (uint32_t i = 0; i < options.warmups; ++i) {
      warmups.push_back(run_request(context, options, origin, i));
    }
    for (uint32_t i = 0; i < options.measured; ++i) {
      measured.push_back(
          run_request(context, options, origin, options.warmups + i));
    }
    const Sample &baseline = measured.front();
    for (const Sample &sample : warmups) {
      if (!same_tokens(baseline, sample))
        throw std::runtime_error("warmup token sequence differs after reset");
    }
    for (const Sample &sample : measured) {
      if (!same_tokens(baseline, sample))
        throw std::runtime_error("measured token sequence differs after reset");
    }
  } catch (...) {
    llama_free(context);
    free_model();
    llama_backend_free();
    throw;
  }
  llama_free(context);
  free_model();
  llama_backend_free();
  const uint32_t early_error_count =
      log_state.error_count.load(std::memory_order_relaxed);
  if (early_error_count != 0) {
    throw std::runtime_error("llama emitted an error before PASS output");
  }

  std::ostringstream out;
  out << "{\"schema_version\":" << json_escape(kSchema)
      << ",\"record_kind\":\"result\",\"state\":\"PASS\""
      << ",\"llama_commit\":" << json_escape(kLlamaCommit)
      << ",\"model\":{\"sha256\":" << json_escape(kModelSha256)
      << ",\"path\":" << json_escape(options.model)
      << ",\"format\":\"GGUF\",\"dtype\":\"BF16\"}"
      << ",\"target\":{\"exact\":" << json_escape(options.target)
      << ",\"gpu_uuid\":" << json_escape(expected_uuid(options.target))
      << ",\"main_gpu\":0,\"logical_device_index\":0}"
      << ",\"row_id\":" << json_escape(options.row_id)
      << ",\"case_id\":" << json_escape(options.case_id)
      << ",\"protocol\":{\"batch_size\":1,\"sequences\":1,\"warmup_requests\":"
         "3,\"measured_requests\":10"
      << ",\"greedy\":true,\"stop_token_ids\":[248046,248044],\"visible_stop_"
         "tokens\":false"
      << ",\"n_ctx\":" << (options.input.size() + options.max_new_tokens)
      << ",\"n_batch\":" << options.n_batch
      << ",\"n_ubatch\":" << options.n_ubatch
      << ",\"n_gpu_layers\":-1,\"split_mode\":\"none\",\"main_gpu\":0"
      << ",\"offload_kqv\":true,\"bos_inserted\":false}"
      << ",\"input_token_ids\":" << json_integer_array(options.input)
      << ",\"model_lifecycle\":{\"load_count\":1,\"context_count\":1,"
         "\"resident_reused\":true,\"load_start_ns\":"
      << load_start << ",\"model_ready_ns\":" << model_ready << "}"
      << ",\"warmups\":{\"count\":3,\"samples\":"
      << samples_json(warmups, options, 0) << "}"
      << ",\"measured\":{\"count\":10,\"samples\":"
      << samples_json(measured, options, 3) << "}"
      << ",\"offload_evidence\":{\"gpu_offload_supported\":true,\"visible_gpu_"
         "device_count\":1"
      << ",\"selected_device\":{\"name\":" << json_escape(offload.device_name)
      << ",\"description\":" << json_escape(offload.device_description)
      << ",\"type\":\"GPU\"}"
      << ",\"requested\":{\"n_gpu_layers\":-1,\"split_mode\":\"none\",\"main_"
         "gpu\":0"
      << ",\"offload_kqv\":true,\"op_offload\":true}"
      << ",\"observed\":{\"offloaded_layers\":" << offload.offloaded_layers
      << ",\"offloadable_layers\":" << offload.offloadable_layers
      << ",\"gpu_model_buffer_mib\":"
      << json_number(offload.gpu_model_buffer_mib)
      << ",\"device_memory\":{\"free_before_bytes\":"
      << offload.memory_free_before_bytes
      << ",\"total_before_bytes\":" << offload.memory_total_before_bytes
      << ",\"free_model_ready_bytes\":" << offload.memory_free_ready_bytes
      << ",\"total_model_ready_bytes\":" << offload.memory_total_ready_bytes
      << ",\"observed_decrease_bytes\":"
      << (offload.memory_free_before_bytes - offload.memory_free_ready_bytes)
      << "}"
      << ",\"captured_log_bytes\":" << offload.captured_log_bytes << "}}"
      << ",\"audit\":{\"early_error_count\":0,\"sample_equality\":true"
      << ",\"request_memory_reset\":true,\"sampler_reset\":true,\"stop_tokens_"
         "not_fed_back\":true"
      << ",\"model_reused\":true,\"context_reused\":true,\"errors\":[]}";
  out << "}";
  return out.str();
}

void print_help(const char *executable) {
  std::cout << "usage: " << executable
            << " --model PATH --model-sha256 SHA256 --target gfx1030|gfx1201|gfx942\n"
            << "  --row-id ID --case-id CASE --input-token-ids CSV "
               "--max-new-tokens N\n"
            << "  --warmup-requests 3 --measured-requests 10 --batch-size 1 "
               "--sequences 1\n"
            << "  --n-batch 2048 --n-ubatch 512 --main-gpu 0 "
               "--benchmark-schema-version llama-phase5-v1\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 ||
                      std::strcmp(argv[1], "-h") == 0)) {
      print_help(argv[0]);
      return 0;
    }
    const Options options = parse_options(argc, argv);
    std::cout << run(options) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "llama-phase5: " << error.what() << '\n';
    return 2;
  }
}
