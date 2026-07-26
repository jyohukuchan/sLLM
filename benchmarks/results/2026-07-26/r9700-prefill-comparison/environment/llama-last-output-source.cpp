--- /home/homelab1/llama.cpp-src/src/llama-batch.cpp:116-132 ---

        batch.pos = pos.data();
    }

    if (!batch.logits) {
        if (output_all) {
            // return the output for all tokens
            output.resize(batch.n_tokens, true);
        } else {
            // return the output only for the last token
            output.resize(batch.n_tokens, false);
            output[output.size() - 1] = true;
        }

        batch.logits = output.data();
    } else if (output_all) {
        bool warn = false;
