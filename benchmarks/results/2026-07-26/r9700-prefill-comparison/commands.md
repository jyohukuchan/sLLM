# Executed commands

The credential-bearing sudo input is intentionally absent.  The one service window was launched with run-service-window.sh after the approved sudo credential priming step.  Non-secret stop/start invocations are retained under service/.

## Service wrapper and final restoration

    ./run-service-window.sh

The wrapper's credential-free `sudo -n systemctl start ullm-openai.service` attempt is recorded in service/restore.txt and returned 1 after the sudo credential expired.  The later approved restoration is represented without the credential as:

    sudo -S -p '' systemctl start ullm-openai.service

Its stdin credential is intentionally not stored.  The timestamped final state and the intervening worker-EOF observation are in service/final-recovery.md.

## Per-condition model commands

### ullm-sq8_0-f32-kv-p128

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1 ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1 ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1 ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 /tmp/ullm-prefill-clean-0216b131/target/release/ullm-sq8-r9700-phase0-profile --phase prefill --prompt-tokens 128 --repeats 5

### llama-cpp-q8_0-f32-kv-p128

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 /home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench -m /home/homelab1/datapool/ai_models/gguf/Qwen/Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf -o json -r 5 -p 128 -n 0 -b 128 -ub 128 -ctk f32 -ctv f32 -ngl 999 -sm none -mg 0 -dev ROCm0 -nkvo 0 -fa on -t 1 -mmp 1 --progress -v

### llama-cpp-q8_0-f16-kv-p128

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 /home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench -m /home/homelab1/datapool/ai_models/gguf/Qwen/Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf -o json -r 5 -p 128 -n 0 -b 128 -ub 128 -ctk f16 -ctv f16 -ngl 999 -sm none -mg 0 -dev ROCm0 -nkvo 0 -fa on -t 1 -mmp 1 --progress -v

### ullm-sq8_0-f32-kv-p512

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1 ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1 ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1 ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 /tmp/ullm-prefill-clean-0216b131/target/release/ullm-sq8-r9700-phase0-profile --phase prefill --prompt-tokens 512 --repeats 5

### llama-cpp-q8_0-f32-kv-p512

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 /home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench -m /home/homelab1/datapool/ai_models/gguf/Qwen/Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf -o json -r 5 -p 512 -n 0 -b 512 -ub 128 -ctk f32 -ctv f32 -ngl 999 -sm none -mg 0 -dev ROCm0 -nkvo 0 -fa on -t 1 -mmp 1 --progress -v

### llama-cpp-q8_0-f16-kv-p512

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 /home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench -m /home/homelab1/datapool/ai_models/gguf/Qwen/Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf -o json -r 5 -p 512 -n 0 -b 512 -ub 128 -ctk f16 -ctv f16 -ngl 999 -sm none -mg 0 -dev ROCm0 -nkvo 0 -fa on -t 1 -mmp 1 --progress -v

### ullm-sq8_0-f32-kv-p1024

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1 ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1 ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1 ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 /tmp/ullm-prefill-clean-0216b131/target/release/ullm-sq8-r9700-phase0-profile --phase prefill --prompt-tokens 1024 --repeats 5

### llama-cpp-q8_0-f32-kv-p1024

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 /home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench -m /home/homelab1/datapool/ai_models/gguf/Qwen/Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf -o json -r 5 -p 1024 -n 0 -b 1024 -ub 128 -ctk f32 -ctv f32 -ngl 999 -sm none -mg 0 -dev ROCm0 -nkvo 0 -fa on -t 1 -mmp 1 --progress -v

### llama-cpp-q8_0-f16-kv-p1024

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 /home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench -m /home/homelab1/datapool/ai_models/gguf/Qwen/Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf -o json -r 5 -p 1024 -n 0 -b 1024 -ub 128 -ctk f16 -ctv f16 -ngl 999 -sm none -mg 0 -dev ROCm0 -nkvo 0 -fa on -t 1 -mmp 1 --progress -v

### ullm-sq8_0-f32-kv-p2048

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1 ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1 ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1 ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 /tmp/ullm-prefill-clean-0216b131/target/release/ullm-sq8-r9700-phase0-profile --phase prefill --prompt-tokens 2048 --repeats 5

### llama-cpp-q8_0-f32-kv-p2048

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 /home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench -m /home/homelab1/datapool/ai_models/gguf/Qwen/Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf -o json -r 5 -p 2048 -n 0 -b 2048 -ub 128 -ctk f32 -ctv f32 -ngl 999 -sm none -mg 0 -dev ROCm0 -nkvo 0 -fa on -t 1 -mmp 1 --progress -v

### llama-cpp-q8_0-f16-kv-p2048

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 /home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench -m /home/homelab1/datapool/ai_models/gguf/Qwen/Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf -o json -r 5 -p 2048 -n 0 -b 2048 -ub 128 -ctk f16 -ctv f16 -ngl 999 -sm none -mg 0 -dev ROCm0 -nkvo 0 -fa on -t 1 -mmp 1 --progress -v

### ullm-sq8_0-f32-kv-p4095

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1 ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1 ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1 ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 /tmp/ullm-prefill-clean-0216b131/target/release/ullm-sq8-r9700-phase0-profile --phase prefill --prompt-tokens 4095 --repeats 5

### llama-cpp-q8_0-f32-kv-p4095

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 /home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench -m /home/homelab1/datapool/ai_models/gguf/Qwen/Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf -o json -r 5 -p 4095 -n 0 -b 4095 -ub 128 -ctk f32 -ctv f32 -ngl 999 -sm none -mg 0 -dev ROCm0 -nkvo 0 -fa on -t 1 -mmp 1 --progress -v

### llama-cpp-q8_0-f16-kv-p4095

    # exact executable command (credential-free)
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1 /home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench -m /home/homelab1/datapool/ai_models/gguf/Qwen/Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf -o json -r 5 -p 4095 -n 0 -b 4095 -ub 128 -ctk f16 -ctv f16 -ngl 999 -sm none -mg 0 -dev ROCm0 -nkvo 0 -fa on -t 1 -mmp 1 --progress -v
