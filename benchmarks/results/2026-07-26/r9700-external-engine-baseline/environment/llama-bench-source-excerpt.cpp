    int depth = 0; // in tokens

    std::vector<uint8_t> buf; // the llama_context state buffer
};

static bool test_prompt(llama_context * ctx, int n_prompt, int n_batch, int n_threads) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens(n_batch);

    int n_processed = 0;

    while (n_processed < n_prompt) {
        int n_tokens = std::min(n_prompt - n_processed, n_batch);
        tokens[0]    = n_processed == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
        for (int i = 1; i < n_tokens; i++) {
            tokens[i] = std::rand() % n_vocab;
        }
        int res = llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens));
        if (res != 0) {
            fprintf(stderr, "%s: failed to decode prompt batch, res = %d\n", __func__, res);
            return false;
        }
        n_processed += n_tokens;
    }

    llama_synchronize(ctx);
    return true;
}

static bool test_gen(llama_context * ctx, int n_gen, int n_threads) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    llama_token token = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;

    for (int i = 0; i < n_gen; i++) {
        int res = llama_decode(ctx, llama_batch_get_one(&token, 1));
        if (res != 0) {
            fprintf(stderr, "%s: failed to decode generation batch, res = %d\n", __func__, res);
            return false;
        }
        llama_synchronize(ctx);
        token = std::rand() % n_vocab;
    }
    return true;
}

static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
        }

        llama_attach_threadpool(ctx, threadpool, NULL);

        // warmup run
        if (!params.no_warmup) {
            if (t.n_prompt > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: warmup prompt run\n", params_idx, params_count);
                }
                //test_prompt(ctx, std::min(t.n_batch, std::min(t.n_prompt, 32)), 0, t.n_batch, t.n_threads);
                bool res = test_prompt(ctx, t.n_prompt, t.n_batch, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run prompt warmup\n", __func__);
                    llama_free(ctx);
                    llama_model_free(lmodel);
                    exit(1);
                }
            }
            if (t.n_gen > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: warmup generation run\n", params_idx, params_count);
                }
                bool res = test_gen(ctx, 1, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run gen warmup\n", __func__);
                    llama_free(ctx);
                    llama_model_free(lmodel);
                    exit(1);
                }
            }
        }

        for (int i = 0; i < params.reps; i++) {
            llama_memory_clear(llama_get_memory(ctx), false);

            if (t.n_depth > 0) {
                bool is_cached = t.n_depth == cstate.depth;

                if (is_cached) {
                    // if previously we have computed at this depth, just restore the state
                    const size_t ret = llama_state_seq_set_data(ctx, cstate.buf.data(), cstate.buf.size(), 0);
                    if (ret == 0) {
                        // if the old state is incompatible with the current context - reprocess from scratch
                        is_cached = false;
                    }
                }

                if (!is_cached) {
                    if (params.progress) {
                        fprintf(stderr, "llama-bench: benchmark %d/%zu: depth run %d/%d\n", params_idx, params_count,
                                i + 1, params.reps);
                    }
                    bool res = test_prompt(ctx, t.n_depth, t.n_batch, t.n_threads);
                    if (!res) {
                        fprintf(stderr, "%s: error: failed to run depth\n", __func__);
                        llama_free(ctx);
                        llama_model_free(lmodel);
                        exit(1);
                    }

                    // store the context state for reuse in later runs
                    cstate.depth = t.n_depth;
                    cstate.buf.resize(llama_state_seq_get_size(ctx, 0));
                    llama_state_seq_get_data(ctx, cstate.buf.data(), cstate.buf.size(), 0);
                } else {
                    if (params.progress) {
                        fprintf(stderr, "llama-bench: benchmark %d/%zu: depth run %d/%d (cached)\n", params_idx, params_count,
                                i + 1, params.reps);
                    }
                }
            }

            uint64_t t_start = get_time_ns();

            if (t.n_prompt > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: prompt run %d/%d\n", params_idx, params_count,
                            i + 1, params.reps);
                }
                bool res = test_prompt(ctx, t.n_prompt, t.n_batch, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run prompt\n", __func__);
                    llama_free(ctx);
                    llama_model_free(lmodel);
                    exit(1);
                }
            }
            if (t.n_gen > 0) {
                if (params.progress) {
                    fprintf(stderr, "llama-bench: benchmark %d/%zu: generation run %d/%d\n", params_idx, params_count,
                            i + 1, params.reps);
                }
                bool res = test_gen(ctx, t.n_gen, t.n_threads);
                if (!res) {
                    fprintf(stderr, "%s: error: failed to run gen\n", __func__);
                    llama_free(ctx);
                    llama_model_free(lmodel);
                    exit(1);
                }
            }

            uint64_t t_ns = get_time_ns() - t_start;
            t.samples_ns.push_back(t_ns);
        }

