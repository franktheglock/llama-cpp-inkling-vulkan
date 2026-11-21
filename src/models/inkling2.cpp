#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    if (hparams.loop_n_pass > 1) {
        ml.get_key(LLM_KV_LOOP_GATE_RANK, hparams.loop_gate_rank);
        if (!hparams.loop_core_start || !hparams.loop_gate_rank || hparams.loop_core_start % hparams.attn_res_block_size ||
            (hparams.loop_core_end - hparams.loop_core_start) % hparams.attn_res_block_size) {
            throw std::runtime_error("inkling2: the looped core must start after the first block and span whole blocks");
        }
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    const bool loop = hparams.loop_n_pass > 1;

    for (int i = 0; i < n_layer - (int) hparams.loop_n_extra(); ++i) {
        auto & layer = layers[i];

        const int iv = hparams.loop_visit(i);

        const int64_t n_head      = hparams.n_head(iv);
        const int64_t n_head_kv   = hparams.n_head_kv(iv);
        const int64_t n_embd_head = hparams.n_embd_head_k(iv);
        const int64_t n_ff_exp    = hparams.n_ff_exp(iv);
        const int64_t n_extent    = hparams.is_swa(iv) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(iv)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(iv) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

        // message controllers of the looped core: (attn, k, v, mlp) x (down, gain, forget)
        if (loop && i >= (int) hparams.loop_core_start && i < (int) hparams.loop_core_end) {
            static const llm_tensor down[4]   = {LLM_TENSOR_LOOP_ATTN_DOWN,   LLM_TENSOR_LOOP_K_DOWN,   LLM_TENSOR_LOOP_V_DOWN,   LLM_TENSOR_LOOP_MLP_DOWN};
            static const llm_tensor gain[4]   = {LLM_TENSOR_LOOP_ATTN_GAIN,   LLM_TENSOR_LOOP_K_GAIN,   LLM_TENSOR_LOOP_V_GAIN,   LLM_TENSOR_LOOP_MLP_GAIN};
            static const llm_tensor forget[4] = {LLM_TENSOR_LOOP_ATTN_FORGET, LLM_TENSOR_LOOP_K_FORGET, LLM_TENSOR_LOOP_V_FORGET, LLM_TENSOR_LOOP_MLP_FORGET};

            const int64_t n_rank = hparams.loop_gate_rank;

            for (int m = 0; m < 4; ++m) {
                layer.loop_down[m]   = create_tensor(tn(down[m],   "weight", i), {2*n_embd, n_rank}, 0);
                layer.loop_gain[m]   = create_tensor(tn(gain[m],   "weight", i), {n_rank, n_embd}, 0);
                layer.loop_forget[m] = create_tensor(tn(forget[m], "weight", i), {n_rank, n_embd}, 0);
            }
        }
    }

    if (loop) {
        const int64_t n_query = 2*(hparams.loop_core_end - hparams.loop_core_start);

        loop_res_query    = create_tensor(tn(LLM_TENSOR_LOOP_RES_QUERY,    "weight"), {n_embd, n_query}, 0);
        loop_kv_res_query = create_tensor(tn(LLM_TENSOR_LOOP_KV_RES_QUERY, "weight"), {n_embd, n_query}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold, ggml_tensor * tau) {
    if (fused_mask & 1) {
        return ggml_inkling_clip(ctx0, cur, scale, tau, threshold);
    }
    if (tau) {
        return ggml_mul(ctx0, clip(cur, scale, threshold, nullptr), tau);
    }
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    ggml_tensor * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
    norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);

    ggml_tensor * factor = ggml_div(ctx0, ggml_fill(ctx0, norm, threshold), norm);

    return ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));
}

std::vector<ggml_tensor *> llama_model_inkling2::graph::mix_slices(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    const int64_t nq = ggml_nelements(query)/n_embd;
    std::vector<ggml_tensor *> out(nq);
    if ((fused_mask & 2) && blocks.size() <= 8) {
        std::vector<ggml_tensor *> bl(blocks.begin(), blocks.end());
        ggml_tensor * m = ggml_inkling_mix2(ctx0, query, ggml_reshape_2d(ctx0, cur, n_embd, n_tokens), bl.data(), (int) bl.size(), 1e-6f);
        for (int64_t j = 0; j < nq; ++j) {
            out[j] = ggml_view_2d(ctx0, m, n_embd, n_tokens, m->nb[1], j*m->nb[2]);
        }
        return out;
    }
    ggml_tensor * m = mix(blocks, cur, query);
    for (int64_t j = 0; j < nq; ++j) {
        out[j] = ggml_cont(ctx0, ggml_view_2d(ctx0, m, n_embd, n_tokens, m->nb[2], j*m->nb[1]));
    }
    return out;
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    ggml_tensor * states = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    for (ggml_tensor * block : blocks) {
        states = ggml_concat(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_tokens), states, 1);
    }
    states = ggml_rms_norm(ctx0, states, 1e-6f);

    ggml_tensor * queries = ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
}

// loop message controller: (1 - tanh(F h))*cur + tanh(G h)*message, h = D [rms(cur); rms(message)]; module: q, k, v, mlp
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * message, const llama_layer & layer, int module) {
    if ((fused_mask & 64) && n_tokens <= 32) {
        // small batches (decode, speculative verify): weight-streaming kernels; large batches keep the matmuls
        ggml_tensor * h = ggml_inkling_ctrl_down(ctx0, cur, message, layer.loop_down[module], 1e-6f);
        return ggml_inkling_ctrl_gate(ctx0, cur, message, h, layer.loop_forget[module], layer.loop_gain[module]);
    }
    ggml_tensor * h = ggml_concat(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), ggml_reshape_3d(ctx0, message, n_embd, 1, n_tokens), 1);
    h = ggml_reshape_2d(ctx0, ggml_rms_norm(ctx0, h, 1e-6f), 2*n_embd, n_tokens);
    h = build_lora_mm(layer.loop_down[module], h);

    if (fused_mask & 8) {
        return ggml_inkling_gate(ctx0, cur, message, build_lora_mm(layer.loop_forget[module], h), build_lora_mm(layer.loop_gain[module], h));
    }

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(layer.loop_gain[module],   h));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(layer.loop_forget[module], h));

    return ggml_add(ctx0, ggml_mul(ctx0, ggml_scale_bias(ctx0, forget, -1.0f, 1.0f), cur), ggml_mul(ctx0, gain, message));
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset,
        ggml_tensor * resid) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    if (fused_mask & 128) {
        // one kernel: conv from the history and the input, and the new conv state (plus any rollback snapshots)
        // written into the state cell (+ optional residual add)
        const size_t row_size = state->nb[1];
        const size_t base_off = inp->mctx->get_head()*row_size + offset*(n_conv - 1)*ggml_element_size(state);
        ggml_tensor * cell = cparams.n_rs_seq > 0
            ? ggml_view_3d(ctx0, state, (n_conv - 1)*n_channels, n_seqs, cparams.n_rs_seq + 1,
                    row_size, inp->mctx->get_size()*row_size, base_off)
            : ggml_view_2d(ctx0, state, (n_conv - 1)*n_channels, n_seqs, row_size, base_off);
        return ggml_inkling_conv_state_add(ctx0, past, ggml_reshape_2d(ctx0, cur, n_channels, n_tokens), weight, dynamic, cell, resid,
                1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
    }

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    if (fused_mask & 4) {
        ggml_tensor * res = ggml_inkling_conv(ctx0, x, weight, dynamic, 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
        return resid ? ggml_add(ctx0, resid, res) : res;
    }

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);

    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_3d(ctx0, x, n_seq_tokens, n_channels, n_seqs, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);

            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, n_channels/n_heads, n_heads, n_tokens);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    ggml_tensor * res = ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
    return resid ? ggml_add(ctx0, resid, res) : res;
}

llama_model_inkling2::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    {
        static const bool unfused = getenv("INKLING_UNFUSED") != nullptr;
        static const int  fmask   = getenv("INKLING_FUSED_MASK") ? atoi(getenv("INKLING_FUSED_MASK")) : 255;
        fused = !unfused;
        fused_mask = unfused ? 0 : fmask;
    }
    auto * inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    auto * attn = inp->get_attn();
    auto * recr = inp->get_recr();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    ggml_tensor * tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    std::vector<ggml_tensor *> blocks;

    // each pass of the looped core restarts from the prelude state and reads the previous pass's blocks through messages
    const int64_t n_loop_core = hparams.loop_n_pass > 1 ? hparams.loop_core_end - hparams.loop_core_start : 0;

    std::vector<ggml_tensor *> prelude_blocks;
    ggml_tensor *              prelude_cur = nullptr;
    std::vector<ggml_tensor *> bank;
    ggml_tensor *              bank_last = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        const int    ip    = hparams.loop_layer(il);

        ggml_tensor * messages[4] = {nullptr, nullptr, nullptr, nullptr};

        if (n_loop_core > 0 && ip >= (int) hparams.loop_core_start && ip < (int) hparams.loop_core_end) {
            const int pass = (il - (int) hparams.loop_core_start)/n_loop_core;

            if (ip == (int) hparams.loop_core_start) {
                if (pass == 0) {
                    prelude_blocks = blocks;
                    prelude_cur    = cur;
                } else {
                    bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                    bank_last = cur;
                    blocks    = prelude_blocks;
                    cur       = prelude_cur;
                }
            }

            if (pass > 0) {
                const int64_t core_idx = ip - (int) hparams.loop_core_start;
                if (fused && model.ink_loop_res_query) {
                    ggml_tensor * q4 = ggml_view_2d(ctx0, model.ink_loop_res_query, n_embd, 4,
                            model.ink_loop_res_query->nb[1], core_idx*model.ink_loop_res_query->nb[2]);
                    const auto m4 = mix_slices(bank, bank_last, q4);
                    messages[0] = m4[0];
                    messages[3] = m4[1];
                    messages[1] = m4[2];
                    messages[2] = m4[3];
                } else {
                    const int64_t row = 2*core_idx;

                    ggml_tensor * q_attn = ggml_view_2d(ctx0, model.loop_res_query,    n_embd, 2, model.loop_res_query->nb[1],    row*model.loop_res_query->nb[1]);
                    ggml_tensor * q_kv   = ggml_view_2d(ctx0, model.loop_kv_res_query, n_embd, 2, model.loop_kv_res_query->nb[1], row*model.loop_kv_res_query->nb[1]);

                    // rows: (attn, mlp) and (k, v)
                    const auto m_attn = mix_slices(bank, bank_last, q_attn);
                    const auto m_kv   = mix_slices(bank, bank_last, q_kv);

                    messages[0] = m_attn[0];
                    messages[3] = m_attn[1];
                    messages[1] = m_kv[0];
                    messages[2] = m_kv[1];
                }
            }
        }

        // record layer input after loop reset so DSpark extracts match upstream inkling2-loop
        res->t_layer_inp[il] = cur;

        const bool    is_swa       = hparams.is_swa(il);
        const int64_t n_head       = hparams.n_head(il);
        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_head  = hparams.n_embd_head_k(il);
        const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

        ggml_tensor * state   = recr->mctx->get_r_l(il);
        ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
        history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

        ggml_tensor * inpSA = cur;

        // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
        ggml_tensor * inputs[3] = {cur, cur, cur};
        const bool can_mix_norm = (fused_mask & 2) && blocks.size() <= 7 && n_embd % 4 == 0 && n_embd <= 4096;
        if (is_swa && !messages[0] && layer.attn_res_query && can_mix_norm) {
            std::vector<ggml_tensor *> bl(blocks.begin(), blocks.end());
            inputs[0] = ggml_inkling_mix_norm(ctx0, layer.attn_res_query, ggml_reshape_2d(ctx0, cur, n_embd, n_tokens),
                    layer.attn_norm, bl.data(), (int) bl.size(), 1e-6f, hparams.f_norm_rms_eps);
            inputs[1] = inputs[2] = inputs[0];
        } else {
            if (layer.attn_res_query) {
                const auto mixed = mix_slices(blocks, cur, layer.attn_res_query);
                for (int j = 0; j < 3; ++j) {
                    inputs[j] = mixed[is_swa ? 0 : j];
                }
            }

            if (messages[0]) {
                // the messages enter before the input norms, so SWA layers normalize three inputs
                for (int j = 0; j < 3; ++j) {
                    inputs[j] = control(inputs[j], messages[j], layer, j);
                }
                cb(inputs[0], "loop_q_inp", il);
            }

            if (is_swa && messages[0]) {
                for (int j = 0; j < 3; ++j) {
                    inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
                }
            } else if (is_swa) {
                inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
                inputs[1] = inputs[2] = inputs[0];
            } else {
                inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
                inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
                inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
            }
        }

        // self-attention
        {
            // projections; small batches use the merged [q | (k | v) | rel | gate] weights built at load time
            ggml_tensor * q_raw = nullptr, * k_raw = nullptr, * v_raw = nullptr, * r_raw = nullptr, * g_raw = nullptr;
            {
                const int64_t n_q = layer.wq->ne[1], n_k = layer.wk->ne[1], n_v = layer.wv->ne[1];
                const int64_t n_r = layer.attn_rel->ne[1], n_g = layer.wqkv_gate->ne[1];
                const bool small = fused && n_tokens <= 32;
                const bool use5  = small && layer.ink_qkvrg && inputs[1] == inputs[0] && inputs[2] == inputs[0];
                const bool use3  = small && layer.ink_qrg && !use5;
                auto slice = [&](ggml_tensor * t, int64_t row0, int64_t rows) {
                    ggml_tensor * v = ggml_view_2d(ctx0, t, rows, n_tokens, t->nb[1], row0*ggml_element_size(t));
                    return n_tokens > 1 ? ggml_cont(ctx0, v) : v;
                };
                if (use5) {
                    ggml_tensor * all = build_lora_mm(layer.ink_qkvrg, inputs[0]);
                    q_raw = slice(all, 0, n_q);
                    k_raw = slice(all, n_q, n_k);
                    v_raw = slice(all, n_q + n_k, n_v);
                    r_raw = slice(all, n_q + n_k + n_v, n_r);
                    g_raw = slice(all, n_q + n_k + n_v + n_r, n_g);
                } else if (use3) {
                    ggml_tensor * all = build_lora_mm(layer.ink_qrg, inputs[0]);
                    q_raw = slice(all, 0, n_q);
                    r_raw = slice(all, n_q, n_r);
                    g_raw = slice(all, n_q + n_r, n_g);
                }
                if (!q_raw) { q_raw = build_lora_mm(layer.wq, inputs[0]); }
                if (!k_raw) { k_raw = build_lora_mm(layer.wk, inputs[1]); }
                if (!v_raw) { v_raw = build_lora_mm(layer.wv, inputs[2]); }
                if (!r_raw) { r_raw = build_lora_mm(layer.attn_rel, inputs[0]); }
                if (!g_raw) { g_raw = build_lora_mm(layer.wqkv_gate, inputs[0]); }
            }

            auto proj_small = [&](ggml_tensor * w, ggml_tensor * x) {
                if (fused && n_tokens <= 32 && w->type == GGML_TYPE_BF16 && w->ne[0] % 4 == 0 && (!loras || loras->empty())) {
                    return ggml_inkling_proj_bf16(ctx0, w, x);
                }
                return build_lora_mm(w, x);
            };

            ggml_tensor * Qcur = q_raw;
            ggml_tensor * Rcur = r_raw;
            ggml_tensor * tau_l = is_swa ? nullptr : tau;
            Qcur = clip(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), layer.attn_q_clip, hparams.f_qk_clip, tau_l);
            Rcur = clip(ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens), layer.attn_rel_clip, hparams.f_relative_clip, tau_l);

            ggml_tensor * Kcur = nullptr;
            ggml_tensor * Vcur = nullptr;
            const auto * kv_mctx = is_swa ? attn->mctx->get_swa() : attn->mctx->get_base();
            ggml_tensor * k_rot = is_swa ? attn->self_k_rot_swa : attn->self_k_rot;
            ggml_tensor * v_rot = is_swa ? attn->self_v_rot_swa : attn->self_v_rot;
            ggml_tensor * k_idxs = is_swa ? attn->get_k_idxs_swa() : attn->get_k_idxs();
            ggml_tensor * v_idxs = is_swa ? attn->get_v_idxs_swa() : attn->get_v_idxs();
            ggml_tensor * k_raw_kv = kv_mctx->get_k_raw(ctx0, il);
            ggml_tensor * v_raw_kv = kv_mctx->get_v_raw(ctx0, il);
            const bool can_kv_store = (fused_mask & 128) && (fused_mask & 16) && n_embd_head == 128 &&
                !k_rot && !v_rot && !kv_mctx->get_v_trans() &&
                k_raw_kv->type == GGML_TYPE_Q8_0 && v_raw_kv->type == GGML_TYPE_Q8_0 &&
                k_idxs->type == GGML_TYPE_I64 && v_idxs->type == GGML_TYPE_I64;

            if (can_kv_store) {
                const int64_t n_conv = hparams.n_shortconv_l_cache;
                const int64_t n_seqs = ubatch.n_seqs;
                const size_t row_size = state->nb[1];
                const auto kv_store = [&](ggml_tensor * kv_dst, ggml_tensor * raw, ggml_tensor * w_conv, ggml_tensor * dyn,
                        int64_t offset, ggml_tensor * idxs, int mode, float param) {
                    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_embd_k_gqa, n_seqs,
                            history->nb[1], history->nb[2], offset*history->nb[1]);
                    const size_t base_off = recr->mctx->get_head()*row_size + offset*(n_conv - 1)*ggml_element_size(state);
                    ggml_tensor * cell = cparams.n_rs_seq > 0
                        ? ggml_view_3d(ctx0, state, (n_conv - 1)*n_embd_k_gqa, n_seqs, cparams.n_rs_seq + 1,
                                row_size, recr->mctx->get_size()*row_size, base_off)
                        : ggml_view_2d(ctx0, state, (n_conv - 1)*n_embd_k_gqa, n_seqs, row_size, base_off);
                    ggml_build_forward_expand(gf, ggml_inkling_kv_store(ctx0, kv_dst, past,
                            ggml_reshape_2d(ctx0, raw, n_embd_k_gqa, n_tokens), w_conv, dyn, cell, idxs,
                            (int) n_embd_head, 1.0f/sqrtf((dyn ? 2 : 1)*n_conv + 1), mode, param));
                };
                ggml_tensor * dyn_k = is_swa ? nullptr : proj_small(layer.attn_k_conv_proj, inputs[1]);
                ggml_tensor * dyn_v = is_swa ? nullptr : proj_small(layer.attn_v_conv_proj, inputs[2]);
                kv_store(k_raw_kv, k_raw, layer.attn_k_conv, dyn_k, 0,            k_idxs, 0, hparams.f_qk_clip);
                kv_store(v_raw_kv, v_raw, layer.attn_v_conv, dyn_v, n_embd_k_gqa, v_idxs, 1, hparams.f_norm_rms_eps);
            } else {
                Kcur = convolve(k_raw, layer.attn_k_conv,
                        is_swa ? nullptr : proj_small(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
                Vcur = convolve(v_raw, layer.attn_v_conv,
                        is_swa ? nullptr : proj_small(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa);
                Kcur = clip(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), nullptr, hparams.f_qk_clip);
                Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);
            }

            cb(Qcur, "Qcur", il);
            cb(Rcur, "Rcur", il);

            ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;

            const int64_t n_stream = (is_swa ? attn->self_kq_mask_swa : attn->self_kq_mask)->ne[3];
            const int64_t n_kv     = rel_pos->ne[0];

            ggml_tensor * sinks = proj_small(layer.attn_sink_proj, inputs[0]);

            if (fused_mask & 16) {
                // one kernel: input mask + gathered relative bias - per-query sink, per head, in the type flash attention takes
                ggml_tensor * mask_in = is_swa ? attn->self_kq_mask_swa : attn->self_kq_mask;
                const ggml_type mask_type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
                ggml_tensor * kq_mask;
                if (fused_mask & 32) {
                    // the bias entries are computed in the kernel from attn_rel_bias and r, no table matmul
                    kq_mask = ggml_inkling_relbias(ctx0, layer.attn_rel_bias, Rcur, sinks, rel_pos, mask_in, mask_type);
                } else {
                    ggml_tensor * table = ggml_mul_mat(ctx0, layer.attn_rel_bias, Rcur); // [n_extent, n_head, n_tokens]; rel_pos == n_extent is the zero row
                    kq_mask = ggml_inkling_relmask(ctx0, table, sinks, rel_pos, mask_in, mask_type);
                }
                cb(kq_mask, "kq_mask_rel", il);

                if (!zero_sinks) {
                    zero_sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
                }
                cur = build_attn_premasked(inp, Qcur, Kcur, Vcur, kq_mask, zero_sinks, 1.0f/sqrtf(n_embd_head), il);
            } else {

            // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x n_head
            ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);

            // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
            table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));
            table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));

            ggml_tensor * kq_b = ggml_get_rows(ctx0, table, rel_pos);
            kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, n_tokens/n_stream, n_stream);
            kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));
            cb(kq_b, "kq_b", il);

            sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);

            cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, kq_b, sinks, 1.0f/sqrtf(n_embd_head), il);
            }

            cur = fused ? ggml_inkling_mul_sig(ctx0, cur, g_raw) : ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, g_raw));
            const bool new_res_block = (ip % hparams.attn_res_block_size == 0);
            cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa,
                    new_res_block ? nullptr : inpSA);
            cb(cur, "attn_out", il);
            if (new_res_block) {
                blocks.push_back(inpSA);
            }
        }

        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        // MoE
        if (!messages[3] && can_mix_norm) {
            std::vector<ggml_tensor *> bl(blocks.begin(), blocks.end());
            cur = ggml_inkling_mix_norm(ctx0, layer.ffn_res_query, ggml_reshape_2d(ctx0, cur, n_embd, n_tokens),
                    layer.ffn_norm, bl.data(), (int) bl.size(), 1e-6f, hparams.f_norm_rms_eps);
        } else {
            cur = mix_slices(blocks, cur, layer.ffn_res_query)[0];
            if (messages[3]) {
                cur = control(cur, messages[3], layer, 3);
                cb(cur, "loop_ffn_inp", il);
            }
            cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        }

        cur = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                nullptr,
                nullptr,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert,
                hparams.n_expert_used(il),
                LLM_FFN_SITU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                il,
                nullptr,
                layer.ffn_gate_up_exps,
                layer.ffn_up_exps_s,
                layer.ffn_gate_exps_s,
                layer.ffn_down_exps_s);

        cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd, ffn_inp);
        cb(cur, "l_out", il);
    }

    if ((fused_mask & 2) && blocks.size() <= 7 && n_embd % 4 == 0 && n_embd <= 4096) {
        std::vector<ggml_tensor *> bl(blocks.begin(), blocks.end());
        cur = ggml_inkling_mix_norm(ctx0, model.output_res_query, ggml_reshape_2d(ctx0, cur, n_embd, n_tokens),
                model.output_norm, bl.data(), (int) bl.size(), 1e-6f, hparams.f_norm_rms_eps);
    } else {
        cur = mix_slices(blocks, cur, model.output_res_query)[0];
        cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    }
    res->t_h_nextn = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    ggml_mul_mat_set_prec(cur, GGML_PREC_F32);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary (view is already contiguous when ne[1] == 1)
    ggml_tensor * unpadded = ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0);
    cur = cur->ne[1] == 1 ? unpadded : ggml_cont(ctx0, unpadded);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // this fork does not auto-expand t_logits
    ggml_build_forward_expand(gf, cur);
}
