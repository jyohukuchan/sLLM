#ifndef SLLM_PUBLIC_RUNTIME_INTERNAL_HPP
#define SLLM_PUBLIC_RUNTIME_INTERNAL_HPP

#include "sllm/hip.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace sllm_public_runtime {

/* The public handle is an opaque token value, never a state address.  A token
 * is consumed for the lifetime of the process, including failed creation and
 * stale-handle paths.  Zero is reserved for null and for exhaustion. */
class MonotonicTokenSource final {
public:
  explicit constexpr MonotonicTokenSource(
      const uintptr_t first = static_cast<uintptr_t>(1U)) noexcept
      : next_(first) {}

  constexpr uintptr_t issue() noexcept {
    if (next_ == 0U) {
      return 0U;
    }
    const uintptr_t token = next_;
    if (token == std::numeric_limits<uintptr_t>::max()) {
      next_ = 0U;
    } else {
      ++next_;
    }
    return token;
  }

private:
  uintptr_t next_;
};

/* All instances used by one Context are mutated while that Context's
 * accounting_mutex is held.  Keeping the transition rules in this small
 * value type makes reservation, rollback, and terminal release share one
 * overflow/underflow policy. */
struct AccountingState final {
  uint64_t active_submissions = 0U;
  uint64_t completion_references = 0U;
  uint64_t child_count = 0U;
  uint64_t lifetime_guards = 0U;

  static constexpr bool can_increment(const uint64_t value) noexcept {
    return value != std::numeric_limits<uint64_t>::max();
  }

  static bool reserve_child(AccountingState &context) noexcept {
    if (!can_increment(context.child_count)) {
      return false;
    }
    ++context.child_count;
    return true;
  }

  static bool release_child(AccountingState &context) noexcept {
    if (context.child_count == 0U) {
      return false;
    }
    --context.child_count;
    return true;
  }

  static bool reserve_lifetime_guard(AccountingState &context) noexcept {
    if (!can_increment(context.lifetime_guards)) {
      return false;
    }
    ++context.lifetime_guards;
    return true;
  }

  static bool release_lifetime_guard(AccountingState &context) noexcept {
    if (context.lifetime_guards == 0U) {
      return false;
    }
    --context.lifetime_guards;
    return true;
  }

  static bool
  release_child_and_lifetime_guard(AccountingState &context) noexcept {
    if (context.child_count == 0U || context.lifetime_guards == 0U) {
      return false;
    }
    --context.child_count;
    --context.lifetime_guards;
    return true;
  }

  static bool reserve_submission(AccountingState &context,
                                 AccountingState &queue,
                                 AccountingState &buffer) noexcept {
    if (!can_increment(queue.active_submissions) ||
        !can_increment(buffer.active_submissions) ||
        !can_increment(queue.completion_references) ||
        !can_increment(buffer.completion_references) ||
        !can_increment(context.child_count) ||
        !can_increment(context.lifetime_guards)) {
      return false;
    }
    ++queue.active_submissions;
    ++buffer.active_submissions;
    ++queue.completion_references;
    ++buffer.completion_references;
    ++context.child_count;
    /* The native event guard owns this reservation until the event is
     * positively destroyed or durably orphaned.  This closes the registry
     * rollback/context-release window. */
    ++context.lifetime_guards;
    return true;
  }

  static void rmsnorm_buffer_counts(const AccountingState *const activation,
                                    const AccountingState *const raw_scale,
                                    const AccountingState *const output,
                                    uint64_t *const activation_count,
                                    uint64_t *const raw_scale_count,
                                    uint64_t *const output_count) noexcept {
    *activation_count = 1U;
    *raw_scale_count = activation == raw_scale ? 0U : 1U;
    *output_count = (output == activation || output == raw_scale) ? 0U : 1U;
    if (output == activation) {
      ++*activation_count;
    } else if (output == raw_scale) {
      ++*raw_scale_count;
    }
  }

  static bool reserve_rmsnorm_submission(AccountingState &context,
                                         AccountingState &queue,
                                         AccountingState &activation,
                                         AccountingState &raw_scale,
                                         AccountingState &output) noexcept {
    uint64_t activation_count = 0U;
    uint64_t raw_scale_count = 0U;
    uint64_t output_count = 0U;
    rmsnorm_buffer_counts(&activation, &raw_scale, &output, &activation_count,
                          &raw_scale_count, &output_count);
    const auto can_add = [](const uint64_t value, const uint64_t amount) {
      return amount <= std::numeric_limits<uint64_t>::max() - value;
    };
    if (!can_increment(queue.active_submissions) ||
        !can_increment(queue.completion_references) ||
        !can_increment(context.child_count) ||
        !can_increment(context.lifetime_guards) ||
        !can_add(activation.active_submissions, activation_count) ||
        !can_add(raw_scale.active_submissions, raw_scale_count) ||
        !can_add(output.active_submissions, output_count) ||
        !can_add(activation.completion_references, activation_count) ||
        !can_add(raw_scale.completion_references, raw_scale_count) ||
        !can_add(output.completion_references, output_count)) {
      return false;
    }
    ++queue.active_submissions;
    ++queue.completion_references;
    ++context.child_count;
    ++context.lifetime_guards;
    activation.active_submissions += activation_count;
    raw_scale.active_submissions += raw_scale_count;
    output.active_submissions += output_count;
    activation.completion_references += activation_count;
    raw_scale.completion_references += raw_scale_count;
    output.completion_references += output_count;
    return true;
  }

  static bool release_rmsnorm_active(AccountingState &queue,
                                     AccountingState &activation,
                                     AccountingState &raw_scale,
                                     AccountingState &output) noexcept {
    uint64_t activation_count = 0U;
    uint64_t raw_scale_count = 0U;
    uint64_t output_count = 0U;
    rmsnorm_buffer_counts(&activation, &raw_scale, &output, &activation_count,
                          &raw_scale_count, &output_count);
    if (queue.active_submissions == 0U ||
        activation.active_submissions < activation_count ||
        raw_scale.active_submissions < raw_scale_count ||
        output.active_submissions < output_count) {
      return false;
    }
    --queue.active_submissions;
    activation.active_submissions -= activation_count;
    raw_scale.active_submissions -= raw_scale_count;
    output.active_submissions -= output_count;
    return true;
  }

  static bool rollback_rmsnorm_submission(AccountingState &context,
                                          AccountingState &queue,
                                          AccountingState &activation,
                                          AccountingState &raw_scale,
                                          AccountingState &output) noexcept {
    uint64_t activation_count = 0U;
    uint64_t raw_scale_count = 0U;
    uint64_t output_count = 0U;
    rmsnorm_buffer_counts(&activation, &raw_scale, &output, &activation_count,
                          &raw_scale_count, &output_count);
    if (queue.active_submissions == 0U || queue.completion_references == 0U ||
        context.child_count == 0U || context.lifetime_guards == 0U ||
        activation.active_submissions < activation_count ||
        raw_scale.active_submissions < raw_scale_count ||
        output.active_submissions < output_count ||
        activation.completion_references < activation_count ||
        raw_scale.completion_references < raw_scale_count ||
        output.completion_references < output_count) {
      return false;
    }
    --queue.active_submissions;
    --queue.completion_references;
    --context.child_count;
    --context.lifetime_guards;
    activation.active_submissions -= activation_count;
    raw_scale.active_submissions -= raw_scale_count;
    output.active_submissions -= output_count;
    activation.completion_references -= activation_count;
    raw_scale.completion_references -= raw_scale_count;
    output.completion_references -= output_count;
    return true;
  }

  static bool release_rmsnorm_completion(AccountingState &context,
                                         AccountingState &queue,
                                         AccountingState &activation,
                                         AccountingState &raw_scale,
                                         AccountingState &output) noexcept {
    uint64_t activation_count = 0U;
    uint64_t raw_scale_count = 0U;
    uint64_t output_count = 0U;
    rmsnorm_buffer_counts(&activation, &raw_scale, &output, &activation_count,
                          &raw_scale_count, &output_count);
    if (queue.completion_references == 0U || context.child_count == 0U ||
        context.lifetime_guards == 0U ||
        activation.completion_references < activation_count ||
        raw_scale.completion_references < raw_scale_count ||
        output.completion_references < output_count) {
      return false;
    }
    --queue.completion_references;
    --context.child_count;
    --context.lifetime_guards;
    activation.completion_references -= activation_count;
    raw_scale.completion_references -= raw_scale_count;
    output.completion_references -= output_count;
    return true;
  }

  static bool release_active(AccountingState &queue,
                             AccountingState &buffer) noexcept {
    if (queue.active_submissions == 0U || buffer.active_submissions == 0U) {
      return false;
    }
    --queue.active_submissions;
    --buffer.active_submissions;
    return true;
  }

  static bool rollback_submission(AccountingState &context,
                                  AccountingState &queue,
                                  AccountingState &buffer) noexcept {
    if (queue.active_submissions == 0U || buffer.active_submissions == 0U ||
        queue.completion_references == 0U ||
        buffer.completion_references == 0U || context.child_count == 0U ||
        context.lifetime_guards == 0U) {
      return false;
    }
    --queue.active_submissions;
    --buffer.active_submissions;
    --queue.completion_references;
    --buffer.completion_references;
    --context.child_count;
    --context.lifetime_guards;
    return true;
  }

  static bool release_completion(AccountingState &context,
                                 AccountingState &queue,
                                 AccountingState &buffer) noexcept {
    if (queue.completion_references == 0U ||
        buffer.completion_references == 0U || context.child_count == 0U) {
      return false;
    }
    --queue.completion_references;
    --buffer.completion_references;
    --context.child_count;
    return true;
  }

  static bool
  release_completion_and_lifetime_guard(AccountingState &context,
                                        AccountingState &queue,
                                        AccountingState &buffer) noexcept {
    if (queue.completion_references == 0U ||
        buffer.completion_references == 0U || context.child_count == 0U ||
        context.lifetime_guards == 0U) {
      return false;
    }
    --queue.completion_references;
    --buffer.completion_references;
    --context.child_count;
    --context.lifetime_guards;
    return true;
  }

  /* A prepared RMSNorm plan retains one context lifetime guard and one
   * reference for each descriptor binding.  The bindings may name the same
   * buffer at disjoint intervals, so counts are intentionally per binding. */
  static bool reserve_prepared_plan(AccountingState &context,
                                    AccountingState &activation,
                                    AccountingState &raw_scale,
                                    AccountingState &output) noexcept {
    const uint64_t activation_count = 1U +
                                      (&activation == &raw_scale ? 1U : 0U) +
                                      (&activation == &output ? 1U : 0U);
    const uint64_t raw_scale_count =
        &raw_scale == &activation ? 0U : 1U + (&raw_scale == &output ? 1U : 0U);
    const uint64_t output_count =
        (&output == &activation || &output == &raw_scale) ? 0U : 1U;
    const auto can_add = [](const uint64_t value, const uint64_t amount) {
      return amount <= std::numeric_limits<uint64_t>::max() - value;
    };
    if (!can_add(context.child_count, 1U) ||
        !can_add(context.lifetime_guards, 1U) ||
        !can_add(activation.child_count, activation_count) ||
        !can_add(raw_scale.child_count, raw_scale_count) ||
        !can_add(output.child_count, output_count)) {
      return false;
    }
    ++context.child_count;
    ++context.lifetime_guards;
    activation.child_count += activation_count;
    raw_scale.child_count += raw_scale_count;
    output.child_count += output_count;
    return true;
  }

  static bool release_prepared_plan(AccountingState &context,
                                    AccountingState &activation,
                                    AccountingState &raw_scale,
                                    AccountingState &output) noexcept {
    const uint64_t activation_count = 1U +
                                      (&activation == &raw_scale ? 1U : 0U) +
                                      (&activation == &output ? 1U : 0U);
    const uint64_t raw_scale_count =
        &raw_scale == &activation ? 0U : 1U + (&raw_scale == &output ? 1U : 0U);
    const uint64_t output_count =
        (&output == &activation || &output == &raw_scale) ? 0U : 1U;
    if (context.child_count == 0U || context.lifetime_guards == 0U ||
        activation.child_count < activation_count ||
        raw_scale.child_count < raw_scale_count ||
        output.child_count < output_count) {
      return false;
    }
    --context.child_count;
    --context.lifetime_guards;
    activation.child_count -= activation_count;
    raw_scale.child_count -= raw_scale_count;
    output.child_count -= output_count;
    return true;
  }

  static bool reserve_kv_state(AccountingState &context,
                               AccountingState &key_buffer,
                               AccountingState &value_buffer) noexcept {
    if (!can_increment(context.child_count) ||
        !can_increment(context.lifetime_guards) ||
        !can_increment(key_buffer.child_count) ||
        !can_increment(value_buffer.child_count)) {
      return false;
    }
    ++context.child_count;
    ++context.lifetime_guards;
    ++key_buffer.child_count;
    ++value_buffer.child_count;
    return true;
  }

  static bool release_kv_state(AccountingState &context,
                               AccountingState &key_buffer,
                               AccountingState &value_buffer) noexcept {
    if (context.child_count == 0U || context.lifetime_guards == 0U ||
        key_buffer.child_count == 0U || value_buffer.child_count == 0U) {
      return false;
    }
    --context.child_count;
    --context.lifetime_guards;
    --key_buffer.child_count;
    --value_buffer.child_count;
    return true;
  }

  static bool reserve_kv_view(AccountingState &context,
                              AccountingState &state) noexcept {
    if (!can_increment(context.child_count) ||
        !can_increment(context.lifetime_guards) ||
        !can_increment(state.child_count)) {
      return false;
    }
    ++context.child_count;
    ++context.lifetime_guards;
    ++state.child_count;
    return true;
  }

  static bool release_kv_view(AccountingState &context,
                              AccountingState &state) noexcept {
    if (context.child_count == 0U || context.lifetime_guards == 0U ||
        state.child_count == 0U) {
      return false;
    }
    --context.child_count;
    --context.lifetime_guards;
    --state.child_count;
    return true;
  }

  static constexpr std::size_t kv_append_resource_count = 5U;

  static void
  kv_append_resource_multiplicities(AccountingState *const *const resources,
                                    uint64_t *const multiplicities) noexcept {
    for (std::size_t index = 0U; index != kv_append_resource_count; ++index) {
      multiplicities[index] = 0U;
    }
    for (std::size_t index = 0U; index != kv_append_resource_count; ++index) {
      std::size_t first = index;
      for (std::size_t prior = 0U; prior != index; ++prior) {
        if (resources[prior] == resources[index]) {
          first = prior;
          break;
        }
      }
      ++multiplicities[first];
    }
  }

  static bool reserve_kv_append(AccountingState &context,
                                AccountingState &queue, AccountingState &state,
                                AccountingState &key_input,
                                AccountingState &value_input,
                                AccountingState &key_buffer,
                                AccountingState &value_buffer) noexcept {
    AccountingState *const resources[] = {&state, &key_input, &value_input,
                                          &key_buffer, &value_buffer};
    uint64_t multiplicities[kv_append_resource_count] = {};
    kv_append_resource_multiplicities(resources, multiplicities);
    const auto can_add = [](const uint64_t value, const uint64_t amount) {
      return amount <= std::numeric_limits<uint64_t>::max() - value;
    };
    for (std::size_t index = 0U; index != kv_append_resource_count; ++index) {
      if (multiplicities[index] != 0U &&
          (!can_add(resources[index]->active_submissions,
                    multiplicities[index]) ||
           !can_add(resources[index]->completion_references,
                    multiplicities[index]))) {
        return false;
      }
    }
    if (!can_increment(queue.active_submissions) ||
        !can_increment(queue.completion_references) ||
        !can_increment(context.child_count) ||
        !can_increment(context.lifetime_guards)) {
      return false;
    }
    ++queue.active_submissions;
    ++queue.completion_references;
    ++context.child_count;
    ++context.lifetime_guards;
    for (std::size_t index = 0U; index != kv_append_resource_count; ++index) {
      if (multiplicities[index] != 0U) {
        resources[index]->active_submissions += multiplicities[index];
        resources[index]->completion_references += multiplicities[index];
      }
    }
    return true;
  }

  static bool release_kv_active(AccountingState &queue, AccountingState &state,
                                AccountingState &key_input,
                                AccountingState &value_input,
                                AccountingState &key_buffer,
                                AccountingState &value_buffer) noexcept {
    AccountingState *const resources[] = {&state, &key_input, &value_input,
                                          &key_buffer, &value_buffer};
    uint64_t multiplicities[kv_append_resource_count] = {};
    kv_append_resource_multiplicities(resources, multiplicities);
    if (queue.active_submissions == 0U) {
      return false;
    }
    for (std::size_t index = 0U; index != kv_append_resource_count; ++index) {
      if (multiplicities[index] != 0U &&
          resources[index]->active_submissions < multiplicities[index]) {
        return false;
      }
    }
    --queue.active_submissions;
    for (std::size_t index = 0U; index != kv_append_resource_count; ++index) {
      if (multiplicities[index] != 0U) {
        resources[index]->active_submissions -= multiplicities[index];
      }
    }
    return true;
  }

  static bool rollback_kv_append(AccountingState &context,
                                 AccountingState &queue, AccountingState &state,
                                 AccountingState &key_input,
                                 AccountingState &value_input,
                                 AccountingState &key_buffer,
                                 AccountingState &value_buffer) noexcept {
    AccountingState *const resources[] = {&state, &key_input, &value_input,
                                          &key_buffer, &value_buffer};
    uint64_t multiplicities[kv_append_resource_count] = {};
    kv_append_resource_multiplicities(resources, multiplicities);
    if (queue.active_submissions == 0U || queue.completion_references == 0U ||
        context.child_count == 0U || context.lifetime_guards == 0U) {
      return false;
    }
    for (std::size_t index = 0U; index != kv_append_resource_count; ++index) {
      if (multiplicities[index] != 0U &&
          (resources[index]->active_submissions < multiplicities[index] ||
           resources[index]->completion_references < multiplicities[index])) {
        return false;
      }
    }
    --queue.active_submissions;
    --queue.completion_references;
    --context.child_count;
    --context.lifetime_guards;
    for (std::size_t index = 0U; index != kv_append_resource_count; ++index) {
      if (multiplicities[index] != 0U) {
        resources[index]->active_submissions -= multiplicities[index];
        resources[index]->completion_references -= multiplicities[index];
      }
    }
    return true;
  }

  static bool release_kv_completion(
      AccountingState &context, AccountingState &queue, AccountingState &state,
      AccountingState &key_input, AccountingState &value_input,
      AccountingState &key_buffer, AccountingState &value_buffer) noexcept {
    AccountingState *const resources[] = {&state, &key_input, &value_input,
                                          &key_buffer, &value_buffer};
    uint64_t multiplicities[kv_append_resource_count] = {};
    kv_append_resource_multiplicities(resources, multiplicities);
    if (queue.completion_references == 0U || context.child_count == 0U ||
        context.lifetime_guards == 0U) {
      return false;
    }
    for (std::size_t index = 0U; index != kv_append_resource_count; ++index) {
      if (multiplicities[index] != 0U &&
          resources[index]->completion_references < multiplicities[index]) {
        return false;
      }
    }
    --queue.completion_references;
    --context.child_count;
    --context.lifetime_guards;
    for (std::size_t index = 0U; index != kv_append_resource_count; ++index) {
      if (multiplicities[index] != 0U) {
        resources[index]->completion_references -= multiplicities[index];
      }
    }
    return true;
  }

  static bool reserve_causal_attention(AccountingState &context,
                                       AccountingState &queue,
                                       AccountingState &state,
                                       AccountingState &query,
                                       AccountingState &output) noexcept {
    AccountingState *const resources[] = {&state, &query, &output};
    uint64_t multiplicities[] = {1U, 1U, 1U};
    for (std::size_t index = 0U; index != 3U; ++index) {
      for (std::size_t prior = 0U; prior != index; ++prior) {
        if (resources[prior] == resources[index]) {
          multiplicities[prior] += multiplicities[index];
          multiplicities[index] = 0U;
          break;
        }
      }
    }
    const auto can_add = [](const uint64_t value, const uint64_t amount) {
      return amount <= std::numeric_limits<uint64_t>::max() - value;
    };
    if (!can_increment(queue.active_submissions) ||
        !can_increment(queue.completion_references) ||
        !can_increment(context.child_count) ||
        !can_increment(context.lifetime_guards)) {
      return false;
    }
    for (std::size_t index = 0U; index != 3U; ++index) {
      if (multiplicities[index] != 0U &&
          (!can_add(resources[index]->active_submissions,
                    multiplicities[index]) ||
           !can_add(resources[index]->completion_references,
                    multiplicities[index]))) {
        return false;
      }
    }
    ++queue.active_submissions;
    ++queue.completion_references;
    ++context.child_count;
    ++context.lifetime_guards;
    for (std::size_t index = 0U; index != 3U; ++index) {
      resources[index]->active_submissions += multiplicities[index];
      resources[index]->completion_references += multiplicities[index];
    }
    return true;
  }

  static bool release_causal_active(AccountingState &queue,
                                    AccountingState &state,
                                    AccountingState &query,
                                    AccountingState &output) noexcept {
    AccountingState *const resources[] = {&state, &query, &output};
    uint64_t multiplicities[] = {1U, 1U, 1U};
    for (std::size_t index = 0U; index != 3U; ++index) {
      for (std::size_t prior = 0U; prior != index; ++prior) {
        if (resources[prior] == resources[index]) {
          multiplicities[prior] += multiplicities[index];
          multiplicities[index] = 0U;
          break;
        }
      }
    }
    if (queue.active_submissions == 0U) {
      return false;
    }
    for (std::size_t index = 0U; index != 3U; ++index) {
      if (multiplicities[index] != 0U &&
          resources[index]->active_submissions < multiplicities[index]) {
        return false;
      }
    }
    --queue.active_submissions;
    for (std::size_t index = 0U; index != 3U; ++index) {
      resources[index]->active_submissions -= multiplicities[index];
    }
    return true;
  }

  static bool rollback_causal_attention(AccountingState &context,
                                        AccountingState &queue,
                                        AccountingState &state,
                                        AccountingState &query,
                                        AccountingState &output) noexcept {
    AccountingState *const resources[] = {&state, &query, &output};
    uint64_t multiplicities[] = {1U, 1U, 1U};
    for (std::size_t index = 0U; index != 3U; ++index) {
      for (std::size_t prior = 0U; prior != index; ++prior) {
        if (resources[prior] == resources[index]) {
          multiplicities[prior] += multiplicities[index];
          multiplicities[index] = 0U;
          break;
        }
      }
    }
    if (queue.active_submissions == 0U || queue.completion_references == 0U ||
        context.child_count == 0U || context.lifetime_guards == 0U) {
      return false;
    }
    for (std::size_t index = 0U; index != 3U; ++index) {
      if (multiplicities[index] != 0U &&
          (resources[index]->active_submissions < multiplicities[index] ||
           resources[index]->completion_references < multiplicities[index])) {
        return false;
      }
    }
    --queue.active_submissions;
    --queue.completion_references;
    --context.child_count;
    --context.lifetime_guards;
    for (std::size_t index = 0U; index != 3U; ++index) {
      resources[index]->active_submissions -= multiplicities[index];
      resources[index]->completion_references -= multiplicities[index];
    }
    return true;
  }

  static bool release_causal_completion(AccountingState &context,
                                        AccountingState &queue,
                                        AccountingState &state,
                                        AccountingState &query,
                                        AccountingState &output) noexcept {
    AccountingState *const resources[] = {&state, &query, &output};
    uint64_t multiplicities[] = {1U, 1U, 1U};
    for (std::size_t index = 0U; index != 3U; ++index) {
      for (std::size_t prior = 0U; prior != index; ++prior) {
        if (resources[prior] == resources[index]) {
          multiplicities[prior] += multiplicities[index];
          multiplicities[index] = 0U;
          break;
        }
      }
    }
    if (queue.completion_references == 0U || context.child_count == 0U ||
        context.lifetime_guards == 0U) {
      return false;
    }
    for (std::size_t index = 0U; index != 3U; ++index) {
      if (multiplicities[index] != 0U &&
          resources[index]->completion_references < multiplicities[index]) {
        return false;
      }
    }
    --queue.completion_references;
    --context.child_count;
    --context.lifetime_guards;
    for (std::size_t index = 0U; index != 3U; ++index) {
      resources[index]->completion_references -= multiplicities[index];
    }
    return true;
  }
};

/* Completion ownership gate used by the native Completion state machine.  A
 * terminal HIP error never proves stream quiescence; only a positive event
 * query can make staging/dependencies releasable.  Event destruction is
 * accepted only after that positive proof. */
struct CompletionSafetyState final {
  enum class Phase : uint8_t {
    Initial,
    PositivelyCompleted,
    Quarantined,
    EventDestroyed,
  };

  std::atomic<uint8_t> phase{static_cast<uint8_t>(Phase::Initial)};
  /* A quarantine request is a monotonic release barrier independent of the
   * phase CAS.  It closes the bounded-retry exhaustion path without ever
   * storing over EventDestroyed.  In particular, a positively-completed
   * phase with this bit set is still conservatively non-releasable. */
  std::atomic<bool> quarantine_requested{false};

  static constexpr std::size_t quarantine_cas_attempt_bound() noexcept {
    return 16U;
  }

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
  static void
  force_quarantine_cas_failures(const uint32_t occurrences) noexcept {
    forced_quarantine_cas_failures_.store(occurrences,
                                          std::memory_order_release);
  }

  static void reset_quarantine_cas_failures() noexcept {
    forced_quarantine_cas_failures_.store(0U, std::memory_order_release);
    force_quarantine_counter_cas_contention_enabled_.store(
        false, std::memory_order_release);
  }

  /* Deterministically force the counter CAS itself to lose every attempted
   * update.  This is a host-test seam for the bounded retry path, not a
   * production fault mode. */
  static void
  force_quarantine_counter_cas_contention(const bool enabled) noexcept {
    force_quarantine_counter_cas_contention_enabled_.store(
        enabled, std::memory_order_release);
  }

  static bool consume_forced_quarantine_cas_failure_for_test() noexcept {
    return consume_forced_quarantine_cas_failure();
  }
#endif

  void observe_positive_completion() noexcept {
    if (quarantine_requested.load(std::memory_order_acquire)) {
      return;
    }
    uint8_t expected = static_cast<uint8_t>(Phase::Initial);
    (void)phase.compare_exchange_strong(
        expected, static_cast<uint8_t>(Phase::PositivelyCompleted),
        std::memory_order_acq_rel, std::memory_order_acquire);
  }

  void quarantine() noexcept {
    quarantine_requested.store(true, std::memory_order_release);
    uint8_t current = phase.load(std::memory_order_acquire);
    for (std::size_t attempt = 0U; attempt != quarantine_cas_attempt_bound();
         ++attempt) {
      if (current == static_cast<uint8_t>(Phase::Quarantined) ||
          current == static_cast<uint8_t>(Phase::EventDestroyed)) {
        return;
      }
      if (consume_forced_quarantine_cas_failure()) {
        continue;
      }
      if (phase.compare_exchange_weak(
              current, static_cast<uint8_t>(Phase::Quarantined),
              std::memory_order_acq_rel, std::memory_order_acquire)) {
        return;
      }
    }

    /* A strong CAS is still bounded and closes genuine spurious-failure
     * exhaustion.  If a concurrent event destroy won the race, its updated
     * expected value is observed and left intact.  If the test seam forces
     * this final attempt too, quarantine_requested remains the fail-closed
     * state barrier and no phase is overwritten. */
    if (current == static_cast<uint8_t>(Phase::Quarantined) ||
        current == static_cast<uint8_t>(Phase::EventDestroyed)) {
      return;
    }
    if (!consume_forced_quarantine_cas_failure()) {
      (void)phase.compare_exchange_strong(
          current, static_cast<uint8_t>(Phase::Quarantined),
          std::memory_order_acq_rel, std::memory_order_acquire);
    }
  }

  bool can_release_graph() const noexcept {
    return !quarantine_requested.load(std::memory_order_acquire) &&
           phase.load(std::memory_order_acquire) ==
               static_cast<uint8_t>(Phase::PositivelyCompleted);
  }

  bool observe_event_destroy_success() noexcept {
    if (quarantine_requested.load(std::memory_order_acquire)) {
      return false;
    }
    uint8_t expected = static_cast<uint8_t>(Phase::PositivelyCompleted);
    if (!phase.compare_exchange_strong(
            expected, static_cast<uint8_t>(Phase::EventDestroyed),
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      return false;
    }
    /* If quarantine requested before this validation, report failure while
     * preserving EventDestroyed.  The caller will therefore retain the graph
     * instead of treating this race as releasable. */
    return !quarantine_requested.load(std::memory_order_acquire);
  }

  bool event_destroyed() const noexcept {
    return phase.load(std::memory_order_acquire) ==
           static_cast<uint8_t>(Phase::EventDestroyed);
  }

private:
  static bool consume_forced_quarantine_cas_failure() noexcept {
#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
    uint32_t current =
        forced_quarantine_cas_failures_.load(std::memory_order_acquire);
    for (std::size_t attempt = 0U; attempt != quarantine_cas_attempt_bound();
         ++attempt) {
      if (current == 0U) {
        return false;
      }
      if (force_quarantine_counter_cas_contention_enabled_.load(
              std::memory_order_acquire)) {
        /* An expected value of zero cannot match a live nonzero injection
         * count.  The compare-exchange is intentionally a no-op when the
         * counter has concurrently reached zero. */
        uint32_t conflicting_expected = 0U;
        (void)forced_quarantine_cas_failures_.compare_exchange_strong(
            conflicting_expected, 0U, std::memory_order_acq_rel,
            std::memory_order_acquire);
        continue;
      }
      if (forced_quarantine_cas_failures_.compare_exchange_weak(
              current, current - 1U, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        return true;
      }
    }

    /* Failure to prove that the injected occurrence was consumed is itself a
     * forced failure.  The caller treats true as a failed phase CAS, so it
     * remains bounded and cannot accidentally report a releasable state. */
    return true;
#endif
    return false;
  }

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
  inline static std::atomic<uint32_t> forced_quarantine_cas_failures_{0U};
  inline static std::atomic<bool>
      force_quarantine_counter_cas_contention_enabled_{false};
#endif
};

/* This is an internal, non-ABI fault seam.  The production HIP runtime
 * consumes these counters at the actual query/destroy/accounting and
 * construction boundaries.  Host probes configure the same seam; they do
 * not replace the lifecycle with a toy classifier.  A zero counter means
 * "do not inject".  A successful CAS consumes one occurrence; bounded CAS
 * exhaustion conservatively reports injection without claiming consumption. */
enum class FaultPoint : uint8_t {
  CompletionQueryPending,
  CompletionQueryFatal,
  EventDestroyError,
  StreamDestroyError,
  AllocationFreeError,
  AccountingFailure,
  ContextSelectionFailure,
  NativeCreationFailure,
  ConstructionCandidateFailure,
  RegistryInsertionFailure,
  RegistryInsertionException,
  Count,
};

#if defined(SLLM_PUBLIC_RUNTIME_HOST_TEST)
class FaultInjector final {
public:
  static constexpr std::size_t cas_attempt_bound() noexcept { return 16U; }

  static void set(const FaultPoint point, const uint32_t occurrences) noexcept {
    slots_[index(point)].store(occurrences, std::memory_order_release);
  }

  /* Host-only deterministic collision seam.  It makes each counter CAS
   * attempt fail without spinning, allowing exhaustion to be tested. */
  static void force_cas_contention(const bool enabled) noexcept {
    force_cas_contention_enabled_.store(enabled, std::memory_order_release);
  }

  static bool consume(const FaultPoint point) noexcept {
    std::atomic<uint32_t> &slot = slots_[index(point)];
    uint32_t current = slot.load(std::memory_order_acquire);
    for (std::size_t attempt = 0U; attempt != cas_attempt_bound(); ++attempt) {
      if (current == 0U) {
        return false;
      }
      if (force_cas_contention_enabled_.load(std::memory_order_acquire)) {
        uint32_t conflicting_expected = 0U;
        (void)slot.compare_exchange_strong(conflicting_expected, 0U,
                                           std::memory_order_acq_rel,
                                           std::memory_order_acquire);
        continue;
      }
      if (slot.compare_exchange_weak(current, current - 1U,
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
        return true;
      }
    }

    /* Do not report a clean path when the counter could not be safely
     * consumed.  Returning true preserves the fail-injection contract and
     * causes callers to take their conservative error path. */
    return true;
  }

  static void reset() noexcept {
    for (auto &slot : slots_) {
      slot.store(0U, std::memory_order_release);
    }
    force_cas_contention_enabled_.store(false, std::memory_order_release);
  }

private:
  static constexpr std::size_t index(const FaultPoint point) noexcept {
    return static_cast<std::size_t>(point);
  }

  inline static std::atomic<uint32_t>
      slots_[static_cast<std::size_t>(FaultPoint::Count)]{};
  inline static std::atomic<bool> force_cas_contention_enabled_{false};
};
#else
/* Production code keeps the call sites identical, but has no configurable
 * fault state or test storage in the HIP archive. */
class FaultInjector final {
public:
  static constexpr void set(const FaultPoint, const uint32_t) noexcept {}
  static constexpr bool consume(const FaultPoint) noexcept { return false; }
  static constexpr void reset() noexcept {}
};
#endif

/* Process-lifetime orphan records have no public "owner capacity exhausted"
 * status.  This owner therefore has no artificial bound and never drops a
 * record; its destructor only releases record storage at process teardown.
 * The HIP runtime wraps it with its own mutex and uses it for every native
 * cleanup guard that returned an ownership-ambiguous error. */
template <typename Record> class DurableRecordOwner final {
public:
  void retain(Record record) { records_.push_back(std::move(record)); }

  std::size_t size() const noexcept { return records_.size(); }

  static constexpr bool has_bounded_capacity() noexcept { return false; }

private:
  std::vector<Record> records_;
};

inline void clear_error(sllm_error_sink_t *const sink) noexcept {
  if (sink == nullptr || sink->message == nullptr ||
      sink->message_capacity == 0U) {
    return;
  }
  sink->message[0] = '\0';
}

inline sllm_status_t write_error_n_bounded(
    sllm_error_sink_t *const sink, const sllm_status_t primary_status,
    const char *const message, const std::size_t message_length,
    const std::size_t source_length) noexcept {
  if (sink == nullptr) {
    return primary_status;
  }
  sink->message_length = static_cast<uint64_t>(message_length);
  if (sink->message_capacity == 0U) {
    return SLLM_STATUS_BUFFER_TOO_SMALL;
  }
  if (sink->message == nullptr) {
    return SLLM_STATUS_INVALID_ARGUMENT;
  }
  if (message == nullptr && message_length != 0U) {
    return SLLM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t capacity = static_cast<std::size_t>(sink->message_capacity);
  const std::size_t copied =
      std::min(message_length, std::min(capacity - 1U, source_length));
  if (copied != 0U) {
    std::memcpy(sink->message, message, copied);
  }
  sink->message[copied] = '\0';
  return message_length <= capacity - 1U ? primary_status
                                         : SLLM_STATUS_BUFFER_TOO_SMALL;
}

inline sllm_status_t write_error_n(sllm_error_sink_t *const sink,
                                   const sllm_status_t primary_status,
                                   const char *const message,
                                   const std::size_t message_length) noexcept {
  return write_error_n_bounded(sink, primary_status, message, message_length,
                               message_length);
}

inline sllm_status_t write_error(sllm_error_sink_t *const sink,
                                 const sllm_status_t primary_status,
                                 const char *const message) noexcept {
  const std::size_t message_length =
      message == nullptr ? 0U : std::strlen(message);
  return write_error_n(sink, primary_status, message, message_length);
}

inline sllm_status_t
validate_error_sink(sllm_error_sink_t *const sink) noexcept {
  if (sink == nullptr) {
    return SLLM_STATUS_OK;
  }
  if (sink->struct_size <
      offsetof(sllm_error_sink_t, abi_version) + sizeof(sink->abi_version)) {
    return SLLM_STATUS_INVALID_ARGUMENT;
  }
  if (sink->struct_size < sizeof(sllm_error_sink_t)) {
    return SLLM_STATUS_INVALID_ARGUMENT;
  }
  if (sink->abi_version != SLLM_HIP_ABI_VERSION) {
    return SLLM_STATUS_INVALID_ABI_VERSION;
  }
  if (sink->reserved[0] != 0U || sink->reserved[1] != 0U) {
    return SLLM_STATUS_RESERVED_NONZERO;
  }
  if (sink->message_capacity >
      static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return SLLM_STATUS_INVALID_ARGUMENT;
  }
  if (sink->message_capacity != 0U && sink->message == nullptr) {
    return SLLM_STATUS_INVALID_ARGUMENT;
  }
  sink->message_length = 0U;
  clear_error(sink);
  return SLLM_STATUS_OK;
}

template <typename Struct>
inline sllm_status_t validate_struct(const Struct *const value,
                                     sllm_error_sink_t *const sink,
                                     const char *const null_message) noexcept {
  if (value == nullptr) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT, null_message);
  }
  if (value->struct_size <
      offsetof(Struct, abi_version) + sizeof(value->abi_version)) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "public ABI struct_size does not include abi_version");
  }
  if (value->struct_size < sizeof(Struct)) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "public ABI struct_size is undersized");
  }
  if (value->abi_version != SLLM_HIP_ABI_VERSION) {
    return write_error(sink, SLLM_STATUS_INVALID_ABI_VERSION,
                       "public ABI version is unsupported");
  }
  return SLLM_STATUS_OK;
}

inline bool add_overflows(const uint64_t left, const uint64_t right) noexcept {
  return right > std::numeric_limits<uint64_t>::max() - left;
}

inline bool valid_arch_name(const char *const value, const std::size_t capacity,
                            std::size_t *const length) noexcept {
  if (value == nullptr || length == nullptr) {
    return false;
  }
  const void *const terminator = std::memchr(value, '\0', capacity);
  if (terminator == nullptr) {
    return false;
  }
  *length =
      static_cast<std::size_t>(static_cast<const char *>(terminator) - value);
  return *length != 0U;
}

inline void copy_fixed_string(char *const destination,
                              const std::size_t capacity,
                              const char *const source) noexcept {
  std::memset(destination, 0, capacity);
  if (source == nullptr) {
    return;
  }
  const std::size_t length = std::strlen(source);
  const std::size_t copied = length < capacity - 1U ? length : capacity - 1U;
  std::memcpy(destination, source, copied);
}

inline sllm_status_t
validate_context_create_info(const sllm_context_create_info_t *const info,
                             sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t status =
      validate_struct(info, sink, "context create info is null");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (info->flags != 0U || info->reserved[0] != 0U || info->reserved[1] != 0U ||
      info->reserved[2] != 0U || info->reserved[3] != 0U) {
    return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                       "context create reserved fields must be zero");
  }
  std::size_t length = 0U;
  if (!valid_arch_name(info->expected_gcn_arch_name, SLLM_HIP_MAX_GCN_ARCH_NAME,
                       &length)) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "context selection requires an exact gcnArchName");
  }
  (void)length;
  return SLLM_STATUS_OK;
}

inline sllm_status_t
validate_queue_create_info(const sllm_queue_create_info_t *const info,
                           sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t status =
      validate_struct(info, sink, "queue create info is null");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (info->flags != 0U || info->reserved[0] != 0U || info->reserved[1] != 0U ||
      info->reserved[2] != 0U || info->reserved[3] != 0U ||
      info->reserved[4] != 0U) {
    return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                       "queue create reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

inline sllm_status_t
validate_buffer_create_info(const sllm_buffer_create_info_t *const info,
                            sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t status =
      validate_struct(info, sink, "buffer create info is null");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (info->size_bytes == 0U ||
      info->size_bytes > SLLM_HIP_MAX_TRANSFER_BYTES * 16U) {
    return write_error(
        sink, SLLM_STATUS_INVALID_ARGUMENT,
        "buffer size is outside the bounded public runtime range");
  }
  if (info->alignment_bytes != 0U &&
      (info->alignment_bytes & (info->alignment_bytes - 1U)) != 0U) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "buffer alignment must be zero or a power of two");
  }
  if (info->alignment_bytes != 0U) {
    return write_error(sink, SLLM_STATUS_UNSUPPORTED,
                       "explicit buffer alignment is not implemented");
  }
  if (info->flags != 0U || info->reserved[0] != 0U || info->reserved[1] != 0U ||
      info->reserved[2] != 0U || info->reserved[3] != 0U ||
      info->reserved[4] != 0U) {
    return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                       "buffer create reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

inline sllm_status_t
validate_transfer_desc(const sllm_transfer_desc_t *const transfer,
                       sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t status =
      validate_struct(transfer, sink, "transfer descriptor is null");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (transfer->size_bytes == 0U ||
      transfer->size_bytes > SLLM_HIP_MAX_TRANSFER_BYTES ||
      transfer->reserved[0] != 0U || transfer->reserved[1] != 0U ||
      transfer->reserved[2] != 0U || transfer->reserved[3] != 0U) {
    return write_error(sink, SLLM_STATUS_INVALID_ARGUMENT,
                       "transfer descriptor is invalid or unbounded");
  }
  return SLLM_STATUS_OK;
}

inline sllm_status_t
validate_completion_result(const sllm_completion_result_t *const result,
                           sllm_error_sink_t *const sink) noexcept {
  const sllm_status_t status =
      validate_struct(result, sink, "completion result is null");
  if (status != SLLM_STATUS_OK) {
    return status;
  }
  if (result->reserved0 != 0U || result->reserved[0] != 0U ||
      result->reserved[1] != 0U || result->reserved[2] != 0U ||
      result->reserved[3] != 0U) {
    return write_error(sink, SLLM_STATUS_RESERVED_NONZERO,
                       "completion result reserved fields must be zero");
  }
  return SLLM_STATUS_OK;
}

} // namespace sllm_public_runtime

#endif // SLLM_PUBLIC_RUNTIME_INTERNAL_HPP
