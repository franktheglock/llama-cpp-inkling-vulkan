#include "models.h"
#include "../llama-memory-recurrent.h"

llm_build_lfm_base::llm_build_lfm_base(const llm_graph_params & params) : llm_graph_context(params) {}

ggml_tensor * llm_build_lfm_base::build_lfm_dense_ffn(const llama_model & model, ggml_tensor * cur, int il) {
    GGML_ASSERT(!model.layers[il].ffn_up_b);
    GGML_ASSERT(!model.layers[il].ffn_gate_b);
    GGML_ASSERT(!model.layers[il].ffn_down_b);
    return build_ffn(cur,
        model.layers[il].ffn_up, NULL, NULL,
        model.layers[il].ffn_gate, NULL, NULL,
        model.layers[il].ffn_down, NULL, NULL,
        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
}

// same as the shortconv block in lfm2.cpp
ggml_tensor * llm_build_lfm_base::build_lfm_shortconv(const llama_model & model,
                                                      const llama_memory_recurrent_context * mctx_cur,
                                                      llm_graph_input_rs * inp_recr,
                                                      ggml_tensor * cur,
                                                      int il) {
    const uint32_t kv_head      = mctx_cur->get_head();
    const int64_t  n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t  n_seqs       = ubatch.n_seqs;
    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    GGML_ASSERT(hparams.n_shortconv_l_cache > 1);
    const uint32_t d_conv = hparams.n_shortconv_l_cache - 1;

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    auto * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
    cb(bcx, "model.layers.{}.conv.in_proj", il);

    constexpr auto n_chunks = 3;
    GGML_ASSERT(bcx->ne[0] % n_chunks == 0);
    const auto chunk_size = bcx->ne[0] / n_chunks;
    auto * b = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                            0 * chunk_size * ggml_element_size(bcx));
    auto * c = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                            1 * chunk_size * ggml_element_size(bcx));
    auto * x = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                            2 * chunk_size * ggml_element_size(bcx));

    auto * bx = ggml_transpose(ctx0, ggml_mul(ctx0, b, x));

    // read conv state
    auto * conv_state = mctx_cur->get_r_l(il);
    auto * conv_rs    = build_rs(inp_recr, conv_state, hparams.n_embd_r(), n_seqs);
    auto * conv       = ggml_reshape_3d(ctx0, conv_rs, d_conv, hparams.n_embd, n_seqs);

    bx = ggml_concat(ctx0, conv, bx, 0);
    GGML_ASSERT(bx->ne[0] > conv->ne[0]);

    // last d_conv columns is a new conv state
    auto * new_conv = ggml_view_3d(ctx0, bx, conv->ne[0], bx->ne[1], bx->ne[2], bx->nb[1], bx->nb[2],
                                   (bx->ne[0] - conv->ne[0]) * ggml_element_size(bx));
    GGML_ASSERT(ggml_are_same_shape(conv, new_conv));

    // write new conv conv state
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_conv,
                                           ggml_view_1d(ctx0, conv_state, ggml_nelements(new_conv),
                                                        kv_head * d_conv * n_embd * ggml_element_size(new_conv))));

    auto * conv_out = ggml_ssm_conv(ctx0, bx, model.layers[il].shortconv.conv);
    cb(conv_out, "model.layers.{}.conv.conv", il);

    auto * y = ggml_mul(ctx0, c, conv_out);
    y        = build_lora_mm(model.layers[il].shortconv.out_proj, y);
    cb(y, "model.layers.{}.conv.out_proj", il);
    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    return ggml_reshape_2d(ctx0, y, y->ne[0], n_seq_tokens * n_seqs);
}
