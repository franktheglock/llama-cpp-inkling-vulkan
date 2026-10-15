// K-quant (Q4_K / Q5_K) partial dot product for one 256-weight block, written as a
// template so one shader can instantiate it against several weight buffers.
//
// The includer defines, before each #include:
//   KQ_NAME   - name of the generated function
//   KQ_A      - block array (gives .dm)
//   KQ_A16    - the same buffer viewed as the packed16 struct
//   KQ_A32    - the same buffer viewed as the packed32 struct (Q4_K only)
//   KQ_Q5     - define for Q5_K, leave undefined for Q4_K
// and, once per shader, the activation accessors LFM2_X4(i) / LFM2_X2(i), which
// return the vec4 / vec2 starting at float index i (i divisible by 4 / 2).
//
// 16 threads cooperate on one block; itid (0..15) selects the 16 weights this
// thread owns. The mapping is the same as mul_mat_vec_q4_k.comp / mul_mat_vec_q5_k.comp.

float KQ_NAME(const uint ib, const uint xbase, const uint itid) {
    const uint il = itid / 4;       // 0...3
    const uint ir = itid - 4 * il;  // 0...3
    const uint v_im = il / 2;       // 0 computes 0,32 + 128,160; 1 computes 64,96 + 192,224
    const uint v_in = il % 2;

#ifdef KQ_Q5
    const uint l0 = 4 * ir + 2 * v_in;    // 0...15
#else
    const uint l0 = 4 * (2 * ir + v_in);  // 0...28
#endif
    const uint q_offset = 32 * v_im + l0;
    const uint y1_idx = xbase + 64 * v_im + l0;
    const uint y2_idx = y1_idx + 128;

    const vec2 dm = vec2(KQ_A[ib].dm);

    const uint32_t scale0_u32 = KQ_A16[ib].scales[v_im    ];
    const uint32_t scale4_u32 = KQ_A16[ib].scales[v_im + 2];
    const uint32_t scale8_u32 = KQ_A16[ib].scales[v_im + 4];

    const uint32_t scale_0_4_l = (scale4_u32 << 16) | scale0_u32;
    const uint32_t scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0) >> 2;
    const vec4 scale_0_4_l_f = vec4(unpack8(scale_0_4_l & 0x3F3F3F3F));
    const vec4 scale8_f = vec4(unpack8((((scale8_u32 << 12) | scale8_u32) & 0x0F0F0F0F) | scale_0_4_h));

    const float sc0 = scale_0_4_l_f.x;
    const float sc1 = scale_0_4_l_f.y;
    const float sc2 = scale_0_4_l_f.z;
    const float sc3 = scale_0_4_l_f.w;
    const float sc4 = scale8_f.x;
    const float sc5 = scale8_f.y;
    const float sc6 = scale8_f.z;
    const float sc7 = scale8_f.w;

#ifdef KQ_Q5
    const uint32_t qs0_u32  = uint32_t(KQ_A16[ib].qs[q_offset / 2])      | (uint32_t(KQ_A16[ib].qs[q_offset / 2 +  8]) << 16);
    const uint32_t qs64_u32 = uint32_t(KQ_A16[ib].qs[q_offset / 2 + 32]) | (uint32_t(KQ_A16[ib].qs[q_offset / 2 + 40]) << 16);

    const uint32_t qh = pack32(u16vec2(KQ_A16[ib].qh[l0 / 2], KQ_A16[ib].qh[l0 / 2 + 8]));

    const uint32_t qs0_lo4  = (qs0_u32 & 0x0F0F0F0F)         + (((qh >> (2 * v_im)) & 0x01010101) << 4);
    const uint32_t qs0_hi4  = ((qs0_u32 >> 4) & 0x0F0F0F0F)  + (((qh >> (2 * v_im)) & 0x02020202) << 3);
    const uint32_t qs64_lo4 = (qs64_u32 & 0x0F0F0F0F)        +  ((qh >> (2 * v_im)) & 0x10101010);
    const uint32_t qs64_hi4 = ((qs64_u32 >> 4) & 0x0F0F0F0F) + (((qh >> (2 * v_im)) & 0x20202020) >> 1);

    const vec2 a0 = LFM2_X2(y1_idx);
    const vec2 a1 = LFM2_X2(y1_idx + 16);
    const vec2 a2 = LFM2_X2(y1_idx + 32);
    const vec2 a3 = LFM2_X2(y1_idx + 48);
    const vec2 b0 = LFM2_X2(y2_idx);
    const vec2 b1 = LFM2_X2(y2_idx + 16);
    const vec2 b2 = LFM2_X2(y2_idx + 32);
    const vec2 b3 = LFM2_X2(y2_idx + 48);

    const vec4 by10  = vec4(a0, a1);
    const vec4 by132 = vec4(a2, a3);
    const vec4 by20  = vec4(b0, b1);
    const vec4 by232 = vec4(b2, b3);
#else
    const uint32_t qs0_u32  = KQ_A32[ib].qs[q_offset / 4];
    const uint32_t qs64_u32 = KQ_A32[ib].qs[q_offset / 4 + 16];

    const uint32_t qs0_lo4  = qs0_u32 & 0x0F0F0F0F;
    const uint32_t qs0_hi4  = (qs0_u32 >> 4) & 0x0F0F0F0F;
    const uint32_t qs64_lo4 = qs64_u32 & 0x0F0F0F0F;
    const uint32_t qs64_hi4 = (qs64_u32 >> 4) & 0x0F0F0F0F;

    const vec4 by10  = LFM2_X4(y1_idx);
    const vec4 by132 = LFM2_X4(y1_idx + 32);
    const vec4 by20  = LFM2_X4(y2_idx);
    const vec4 by232 = LFM2_X4(y2_idx + 32);
#endif

    const float sx = dot(by10,  vec4(unpack8(qs0_lo4)));
    const float sy = dot(by132, vec4(unpack8(qs0_hi4)));
    const float sz = dot(by20,  vec4(unpack8(qs64_lo4)));
    const float sw = dot(by232, vec4(unpack8(qs64_hi4)));

    const float smin =
        fma(by10.x + by10.y + by10.z + by10.w,     sc2,
        fma(by132.x + by132.y + by132.z + by132.w, sc3,
        fma(by20.x + by20.y + by20.z + by20.w,     sc6,
           (by232.x + by232.y + by232.z + by232.w) * sc7)));

    return fma(dm.x, fma(sx, sc0, fma(sy, sc1, fma(sz, sc4, sw * sc5))), -dm.y * smin);
}
