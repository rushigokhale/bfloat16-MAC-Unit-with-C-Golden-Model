// Checks the golden model itself before we trust it to check the RTL.
//  1. Hand-computed cases (worked out on paper from the spec).
//  2. Cross-check against the host FPU: fmaf(a, b, c) is a correctly rounded fused
//     multiply-add, which is exactly what the spec describes, except for the
//     denormal/NaN conventions, which we apply on top.
#include <cstdio>
#include <cfloat>
#include <cmath>
#include "../model/bf16_mac_model.h"
#include "stimulus.h"

static int fails = 0;
static void expect(const char* name, uint16_t a, uint16_t b, uint32_t c, uint32_t want) {
    uint32_t got = ref::mac_bits(a, b, c);
    if (got != want) { std::printf("FAIL %-28s a=%04x b=%04x c=%08x got=%08x want=%08x\n", name, a, b, c, got, want); fails++; }
}

static uint32_t flush32(uint32_t u) { return ((u >> 23) & 0xFF) == 0 ? (u & 0x80000000u) : u; }
static uint16_t flush16(uint16_t u) { return ((u >> 7) & 0xFF) == 0 ? (u & 0x8000u) : u; }

int main() {
    using namespace ref;
    // 1.5 * 2.0 + 1.0 = 4.0
    expect("1.5*2+1", 0x3FC0, 0x4000, 0x3F800000, 0x40800000);
    // zero product returns acc unchanged
    expect("0*x+acc", 0x0000, 0x3F80, 0x40490FDB, 0x40490FDB);
    // signed zero rules
    expect("-0 + -0", 0x8000, 0x3F80, 0x80000000, 0x80000000);
    expect("+0 + -0", 0x0000, 0x3F80, 0x80000000, 0x00000000);
    // exact cancellation gives +0
    expect("1*1-1", 0x3F80, 0x3F80, 0xBF800000, 0x00000000);
    // specials
    expect("inf*0", 0x7F80, 0x0000, 0x3F800000, QNAN);
    expect("inf + -inf", 0x7F80, 0x3F80, 0xFF800000, QNAN);
    expect("inf + inf", 0x7F80, 0x3F80, 0x7F800000, 0x7F800000);
    expect("-inf*2 + 5", 0xFF80, 0x4000, 0x40A00000, 0xFF800000);
    expect("nan in", 0x7F81, 0x3F80, 0x3F800000, QNAN);
    expect("nan in acc", 0x3F80, 0x3F80, 0xFFBFFFFF, QNAN);
    // denormal inputs are zero
    expect("denorm a", 0x0040, 0x3F80, 0x3F800000, 0x3F800000);
    expect("denorm acc", 0x3F80, 0x3F80, 0x00000001, 0x3F800000);
    // overflow / underflow
    expect("overflow", 0x7F7F, 0x7F7F, 0x00000000, 0x7F800000);
    expect("neg overflow", 0xFF7F, 0x7F7F, 0x00000000, 0xFF800000);
    expect("underflow ftz", 0x0080, 0x0080, 0x00000000, 0x00000000);
    expect("neg underflow ftz", 0x8080, 0x0080, 0x00000000, 0x80000000);
    // ties: 2^-12 * 2^-12 = 2^-24 is exactly half an ulp of 1.0
    expect("tie to even (down)", 0x3980, 0x3980, 0x3F800000, 0x3F800000);   // 1.0 is even -> stays
    expect("tie to even (up)",   0x3980, 0x3980, 0x3F800001, 0x3F800002);   // 1+2^-23 is odd -> up
    // just above a tie must round up: 1.0 + 2^-24 * 1.0078125 (mantissa 1.0000001b)
    expect("above tie", 0x3980, 0x3981, 0x3F800000, 0x3F800001);
    // round carry: 0x3FFFFFFF + (just over half an ulp) rounds into the next binade
    expect("round carry", 0x3980, 0x3981, 0x3FFFFFFF, 0x40000000);

    // ---- cross-check against fmaf ----------------------------------------
    Rng r(12345);
    long checked = 0, skipped = 0, bad = 0;
    for (long i = 0; i < 6000000 && bad < 10; i++) {
        Op o = gen_op(r, (int)(i % S_NUM_CLASSES));
        uint32_t want_bits = ref::mac_bits(o.a, o.b, o.c);
        float fa = bf16_to_float(flush16(o.a)), fb = bf16_to_float(flush16(o.b)), fc = u2f(flush32(o.c));
        float fr = std::fmaf(fa, fb, fc);
        uint32_t fu = f2u(fr);
        bool ok = true;
        if (std::isnan(fr)) ok = (want_bits == QNAN);
        else if (std::fabs(fr) == FLT_MIN) { skipped++; continue; }              // convention differs at the edge
        else if (std::fabs(fr) < FLT_MIN) ok = (want_bits == (fu & 0x80000000u)); // zero or flushed denormal
        else ok = (want_bits == fu);
        checked++;
        if (!ok) {
            bad++;
            std::printf("MISMATCH a=%04x b=%04x c=%08x model=%08x fmaf=%08x\n", o.a, o.b, o.c, want_bits, fu);
        }
    }
    std::printf("fmaf cross-check: %ld compared, %ld skipped (edge), %ld mismatches\n", checked, skipped, bad);
    fails += (int)bad;
    std::printf(fails ? "MODEL SELFTEST FAILED (%d)\n" : "MODEL SELFTEST PASSED\n", fails);
    return fails ? 1 : 0;
}
