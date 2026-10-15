// Constrained-random stimulus. Each class targets a specific hard corner of the
// MAC. This is the C++ equivalent of SystemVerilog constraint blocks / sequences.
#pragma once
#include <cstdint>
#include <cmath>
#include "../model/bf16_mac_model.h"

struct Rng {                               // xorshift64*
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) { for (int i = 0; i < 8; i++) u64(); }
    uint64_t u64() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 0x2545F4914F6CDD1Dull; }
    uint32_t u32() { return (uint32_t)(u64() >> 32); }
    uint32_t below(uint32_t n) { return n ? (uint32_t)(((uint64_t)u32() * n) >> 32) : 0; }
    int range(int lo, int hi) { return lo + (int)below((uint32_t)(hi - lo + 1)); }   // inclusive
    bool chance(int pct) { return (int)below(100) < pct; }
};

struct Op { uint16_t a, b; uint32_t c; };

enum StimClass {
    S_FULL_RANDOM, S_NORMALS, S_NEAR_EXP, S_CANCEL, S_SPECIALS, S_NEAR_UNDERFLOW,
    S_NEAR_OVERFLOW, S_ROUND_CARRY, S_TIE, S_DENORM_MIX, S_FAR_EXP, S_NUM_CLASSES
};
static const char* const STIM_NAMES[S_NUM_CLASSES] = {
    "full_random", "normals", "near_exp", "cancel", "specials", "near_underflow",
    "near_overflow", "round_carry", "tie", "denorm_mix", "far_exp"
};

inline uint16_t mk_bf16(bool s, int e, int f) { return (uint16_t)(((s ? 1 : 0) << 15) | ((e & 0xFF) << 7) | (f & 0x7F)); }
inline uint32_t mk_fp32(bool s, int e, uint32_t f) { return ((uint32_t)(s ? 1 : 0) << 31) | ((uint32_t)(e & 0xFF) << 23) | (f & 0x7FFFFF); }
inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline uint16_t special_bf16(Rng& r) {
    static const uint16_t v[] = { 0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFF81,
                                  0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x7F7F, 0xFF7F,
                                  0x3F80, 0xBF80, 0x4000, 0x3F00 };
    return v[r.below(sizeof(v) / sizeof(v[0]))];
}
inline uint32_t special_fp32(Rng& r) {
    static const uint32_t v[] = { 0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
                                  0x7F800001, 0xFFBFFFFF, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,
                                  0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000 };
    return v[r.below(sizeof(v) / sizeof(v[0]))];
}

inline int ctz32(uint32_t v) { int n = 0; while (!(v & 1)) { v >>= 1; n++; } return n; }

inline Op gen_op(Rng& r, int cls) {
    Op o;
    switch (cls) {
    case S_FULL_RANDOM:
        o.a = (uint16_t)r.u32(); o.b = (uint16_t)r.u32(); o.c = r.u32(); break;

    case S_NORMALS:
        o.a = mk_bf16(r.chance(50), r.range(100, 154), r.u32());
        o.b = mk_bf16(r.chance(50), r.range(100, 154), r.u32());
        o.c = mk_fp32(r.chance(50), r.range(60, 194), r.u32()); break;

    case S_NEAR_EXP: {                         // alignment shift of -30..+30 bits
        int ea = r.range(60, 190), eb = r.range(60, 190);
        int ep = ea + eb - 127;                // biased exponent of product (before normalise)
        int ec = clampi(ep + r.range(-30, 30), 1, 254);
        o.a = mk_bf16(r.chance(50), ea, r.u32()); o.b = mk_bf16(r.chance(50), eb, r.u32());
        o.c = mk_fp32(r.chance(50), ec, r.u32()); break;
    }
    case S_CANCEL: {                           // acc ~= -(a*b): massive cancellation / exact zero
        o.a = mk_bf16(r.chance(50), r.range(20, 200), r.u32());
        o.b = mk_bf16(r.chance(50), r.range(20, 200), r.u32());
        float p = ref::bf16_to_float(o.a) * ref::bf16_to_float(o.b);    // exact when normal
        uint32_t c = ref::f2u(-p);
        int pert = r.range(0, 3);              // 0 = exact cancel, else a few ulp away
        if (pert) c = (r.chance(50)) ? c + pert : c - pert;
        if (r.chance(8)) c ^= 0x80000000u;     // sometimes same-sign (no cancellation)
        o.c = c; break;
    }
    case S_SPECIALS:
        o.a = r.chance(70) ? special_bf16(r) : (uint16_t)r.u32();
        o.b = r.chance(70) ? special_bf16(r) : (uint16_t)r.u32();
        o.c = r.chance(70) ? special_fp32(r) : r.u32(); break;

    case S_NEAR_UNDERFLOW: {                   // result exponent hovers around the FTZ threshold
        int target = r.range(-4, 5);           // desired biased exponent of the product
        int tot = clampi(target + 127, 2, 254);   // = ea + eb needed
        int ea = clampi(tot / 2 + r.range(-20, 20), 1, 254);
        int eb = clampi(tot - ea, 1, 254);
        o.a = mk_bf16(r.chance(50), ea, r.u32()); o.b = mk_bf16(r.chance(50), eb, r.u32());
        int pe = ea + eb - 127;
        int k = r.below(4);
        if (k == 0) o.c = mk_fp32(r.chance(50), 0, r.u32());              // denormal / zero acc
        else if (k == 1) o.c = mk_fp32(r.chance(50), clampi(pe + r.range(-3, 3), 1, 254), r.u32());
        else { // c ~= -(product) so that the difference lands near/below FLT_MIN
            float p = ref::bf16_to_float(o.a) * ref::bf16_to_float(o.b);
            uint32_t c = ref::f2u(-p);
            if (((c >> 23) & 0xFF) == 0) c = mk_fp32(c >> 31, 1, r.u32());
            o.c = c + r.range(0, 8) - 4;
        }
        break;
    }
    case S_NEAR_OVERFLOW: {                    // product/acc near the top of the exponent range
        int ea = r.range(100, 254), eb = r.range(100, 254);
        int pe = ea + eb - 127;
        int ec = r.chance(50) ? r.range(240, 254) : clampi(pe + r.range(-3, 3), 1, 254);
        o.a = mk_bf16(r.chance(50), ea, r.u32()); o.b = mk_bf16(r.chance(50), eb, r.u32());
        o.c = mk_fp32(r.chance(50), ec, r.chance(50) ? 0x7FFFFF : r.u32()); break;
    }
    case S_ROUND_CARRY: {                      // acc mantissa all ones: rounding can carry out
        int ea = r.range(60, 190), eb = r.range(60, 190);
        int ep = ea + eb - 127;
        int ec = clampi(ep + r.range(18, 27), 1, 254);
        o.a = mk_bf16(r.chance(50), ea, r.u32()); o.b = mk_bf16(r.chance(50), eb, r.u32());
        bool cs = r.chance(50);
        o.c = mk_fp32(cs, ec, r.chance(70) ? 0x7FFFFF : r.u32());
        if (r.chance(75)) {                    // mostly make the product sign match the acc sign
            bool psn = ((o.a >> 15) ^ (o.b >> 15)) & 1;
            if (psn != cs) o.b ^= 0x8000;
        }
        break;
    }
    case S_TIE: {                              // build operands so the product is a half-ulp of acc
        int ea = r.range(60, 190), eb = r.range(60, 190);
        o.a = mk_bf16(r.chance(50), ea, r.u32()); o.b = mk_bf16(r.chance(50), eb, r.u32());
        uint32_t mp = (uint32_t)(0x80 | (o.a & 0x7F)) * (uint32_t)(0x80 | (o.b & 0x7F));
        int ep = (ea - 127 - 7) + (eb - 127 - 7);                 // product = mp * 2^ep
        int lowbit = ep + ctz32(mp);                              // weight of product's lowest set bit
        int ec = lowbit + 1 + 150;                                // acc ulp is twice that
        if (ec < 1 || ec > 254) { o.c = r.u32(); break; }
        o.c = mk_fp32(r.chance(50), ec, r.u32() & (r.chance(50) ? 0x7FFFFF : 0x7FFFF1)); break;
    }
    case S_DENORM_MIX:
        o.a = r.chance(50) ? mk_bf16(r.chance(50), 0, r.u32()) : mk_bf16(r.chance(50), r.range(1, 254), r.u32());
        o.b = r.chance(50) ? mk_bf16(r.chance(50), 0, r.u32()) : mk_bf16(r.chance(50), r.range(1, 254), r.u32());
        o.c = r.chance(50) ? mk_fp32(r.chance(50), 0, r.u32()) : mk_fp32(r.chance(50), r.range(1, 254), r.u32()); break;

    default: {                                 // S_FAR_EXP: exponent gap well beyond the datapath width
        int ea = r.range(30, 220), eb = r.range(30, 220);
        int ep = ea + eb - 127;
        int gap = r.range(27, 120) * (r.chance(50) ? 1 : -1);
        int ec = clampi(ep + gap, 1, 254);
        o.a = mk_bf16(r.chance(50), ea, r.u32()); o.b = mk_bf16(r.chance(50), eb, r.u32());
        o.c = mk_fp32(r.chance(50), ec, r.u32()); break;
    }
    }
    return o;
}
