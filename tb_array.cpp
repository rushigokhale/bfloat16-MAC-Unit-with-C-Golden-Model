// Testbench for bf16_systolic (N x N, weight-stationary), Verilator + C++.
//
// Expected output for a vector x and weights W:
//     y[c] = (((0 + x0*W0c) + x1*W1c) + ... + x(N-1)*W(N-1)c)
// computed with the *same golden MAC model* used for the single-MAC testbench, so
// every column must match bit-exactly. One extra check does not rely on the model at
// all: with W = identity, y[c] must simply equal x[c] (converted to fp32). That
// independently proves the array's wiring, skew and de-skew.
//
// Plusargs:  +seed=N  +vectors=N (per phase, default 3000)  +quiet
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <type_traits>
#include "Varray_top.h"
#include "verilated.h"
#if VM_COVERAGE
#include "verilated_cov.h"
#endif
#include "../model/bf16_mac_model.h"
#include "stimulus.h"

#ifndef ARRAY_N
#define ARRAY_N 4
#endif
static constexpr int N = ARRAY_N;
static constexpr int LAT = 6 * N - 3;                       // x_valid -> y_valid latency


// ---- packed-port access helpers (field widths are 16/32 and aligned, so they never straddle a word)
template <class T> typename std::enable_if<std::is_integral<T>::value>::type
put_field(T& port, int lsb, int w, uint64_t v) {
    uint64_t mask = (w >= 64) ? ~0ull : ((1ull << w) - 1);
    uint64_t cur = (uint64_t)port;
    port = (T)((cur & ~(mask << lsb)) | ((v & mask) << lsb));
}
template <size_t NW> void put_field(VlWide<NW>& port, int lsb, int w, uint64_t v) {
    uint32_t mask = (w >= 32) ? 0xFFFFFFFFu : ((1u << w) - 1);
    int word = lsb / 32, sh = lsb % 32;
    port[word] = (port[word] & ~(mask << sh)) | (((uint32_t)v & mask) << sh);
}
template <class T> typename std::enable_if<std::is_integral<T>::value, uint64_t>::type
get_field(const T& port, int lsb, int w) {
    uint64_t mask = (w >= 64) ? ~0ull : ((1ull << w) - 1);
    return ((uint64_t)port >> lsb) & mask;
}
template <size_t NW> uint64_t get_field(const VlWide<NW>& port, int lsb, int w) {
    uint32_t mask = (w >= 32) ? 0xFFFFFFFFu : ((1u << w) - 1);
    return (port[lsb / 32] >> (lsb % 32)) & mask;
}

struct Pending { uint16_t x[N]; uint32_t exp[N]; uint64_t tag; const char* label; bool identity; };

// element generators: mode 0 = well-behaved normals, 1 = mixed zeros/specials, 2 = wide exponent range
static uint16_t rand_elem(Rng& r, int mode) {
    switch (mode) {
    case 0:  return mk_bf16(r.chance(50), r.range(115, 140), r.u32());
    case 1:  if (r.chance(15)) return r.chance(50) ? 0x0000 : special_bf16(r);
             return mk_bf16(r.chance(50), r.range(100, 150), r.u32());
    default: return mk_bf16(r.chance(50), r.range(1, 254), r.u32());
    }
}

class Env {
public:
    Env(Varray_top* t, Rng& r) : top(t), rng(r) {}

    void cycle(bool valid, const uint16_t* x, const char* label, bool identity = false) {
        top->clk = 0;
        top->x_valid = valid;
        for (int i = 0; i < N; i++) put_field(top->x_in_flat, 16 * i, 16, x ? x[i] : (uint16_t)rng.u32());
        top->eval();
        top->clk = 1; top->eval();
        ++edge_no;
        if (valid) {
            Pending p; p.tag = edge_no; p.label = label; p.identity = identity;
            for (int i = 0; i < N; i++) p.x[i] = x[i];
            for (int c = 0; c < N; c++) {                    // golden: chain of golden MACs down the column
                uint32_t ps = 0;
                for (int r = 0; r < N; r++) ps = ref::mac_bits(x[r], W[r][c], ps);
                p.exp[c] = ps;
            }
            q.push_back(p);
        }
        monitor();
    }
    void idle() { cycle(false, nullptr, "idle"); }
    void drain() { for (int i = 0; i < LAT + 3; i++) idle(); }

    void write_weight(int r, int c, uint16_t v) {
        W[r][c] = v;
        top->w_we = 1; top->w_row = r; top->w_col = c; top->w_data = v;
        cycle(false, nullptr, "weight_load");
        top->w_we = 0;
    }
    void load_weights_random(int mode) {
        drain();
        for (int r = 0; r < N; r++) for (int c = 0; c < N; c++) write_weight(r, c, rand_elem(rng, mode));
    }
    void load_identity() {
        drain();
        for (int r = 0; r < N; r++) for (int c = 0; c < N; c++) write_weight(r, c, r == c ? 0x3F80 : 0x0000);
    }

    void monitor() {
        uint64_t due = edge_no - (LAT - 1);
        bool expect_valid = !q.empty() && q.front().tag == due;
        if (top->y_valid != (expect_valid ? 1 : 0)) {
            mismatches++;
            if (!first_label) { first_label = "valid_pipeline"; first_op = checked; }
            if (!quiet && mismatches <= 8) std::printf("MISMATCH[VALID] y_valid=%d at edge %llu\n", (int)top->y_valid, (unsigned long long)edge_no);
            if (top->y_valid && !q.empty() && q.front().tag < due) q.pop_front();
            return;
        }
        if (!expect_valid) return;
        Pending p = q.front(); q.pop_front();
        checked++;
        for (int c = 0; c < N; c++) {
            uint32_t got = (uint32_t)get_field(top->y_out_flat, 32 * c, 32);
            bool bad = (got != p.exp[c]);
            if (p.identity && ((p.x[c] >> 7) & 0xFF) != 0 && ((p.x[c] >> 7) & 0xFF) != 0xFF)
                bad = bad || (got != ((uint32_t)p.x[c] << 16));          // model-independent check
            if (bad) {
                mismatches++;
                if (!first_label) { first_label = p.label; first_op = checked; }
                if (!quiet && mismatches <= 8)
                    std::printf("MISMATCH[DATA] phase=%s col=%d got=%08x exp=%08x\n", p.label, c, got, p.exp[c]);
            }
            uint32_t e = (p.exp[c] >> 23) & 0xFF;
            if (e == 0xFF) (p.exp[c] & 0x7FFFFF) ? n_nan++ : n_inf++;
            else if (e == 0) n_zero++; else n_norm++;
        }
    }

    // ---- traffic helpers --------------------------------------------------
    void send_vectors(int count, int xmode, int valid_pct, const char* label, bool identity = false) {
        for (int i = 0; i < count; i++) {
            if (!rng.chance(valid_pct)) { idle(); continue; }
            uint16_t x[N];
            for (int r = 0; r < N; r++) x[r] = rand_elem(rng, xmode);
            cycle(true, x, label, identity);
        }
        drain();
    }
    void reset_in_flight() {
        uint16_t x[N];
        for (int i = 0; i < 6; i++) { for (int r = 0; r < N; r++) x[r] = rand_elem(rng, 0); cycle(true, x, "reset_test"); }
        top->rst_n = 0;
        q.clear();
        for (int i = 0; i < 3; i++) cycle(false, nullptr, "reset_hold");
        top->rst_n = 1;
        drain();
    }

    Varray_top* top; Rng& rng;
    uint16_t W[N][N] = {};
    std::deque<Pending> q;
    uint64_t edge_no = 0, checked = 0, mismatches = 0, first_op = 0;
    uint64_t n_nan = 0, n_inf = 0, n_zero = 0, n_norm = 0;
    const char* first_label = nullptr;
    bool quiet = false;
};

static uint64_t plus_u64(int argc, char** argv, const char* key, uint64_t dflt) {
    size_t n = std::strlen(key);
    for (int i = 1; i < argc; i++) if (!std::strncmp(argv[i], key, n)) return std::strtoull(argv[i] + n, nullptr, 0);
    return dflt;
}
static bool plus_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; i++) if (!std::strcmp(argv[i], key)) return true;
    return false;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::fatalOnError(false);
    uint64_t seed = plus_u64(argc, argv, "+seed=", 1);
    int nvec = (int)plus_u64(argc, argv, "+vectors=", 3000);

    Varray_top* top = new Varray_top;
    Rng rng(seed);
    Env env(top, rng);
    env.quiet = plus_flag(argc, argv, "+quiet");

    top->rst_n = 0; top->w_we = 0; top->x_valid = 0;
    for (int i = 0; i < N; i++) put_field(top->x_in_flat, 16 * i, 16, 0);
    for (int i = 0; i < 4; i++) { top->clk = 0; top->eval(); top->clk = 1; top->eval(); }
    top->rst_n = 1;

    // phase 1: identity weights -> y == x  (model-independent wiring check)
    env.load_identity();
    env.send_vectors(nvec, 0, 100, "identity_b2b", true);
    env.send_vectors(nvec, 0, 50, "identity_bubbles", true);

    // phase 2: random weight sets x several activation styles x traffic patterns
    static const int wmodes[] = { 0, 0, 1, 2, 0, 1 };
    static const int xmodes[] = { 0, 1, 2 };
    static const int pcts[]   = { 100, 60, 25 };
    for (int s = 0; s < 6; s++) {
        env.load_weights_random(wmodes[s]);
        for (int xm : xmodes) for (int pct : pcts) {
            char* lbl = (char*)std::malloc(48);
            std::snprintf(lbl, 48, "w%d_x%d_valid%d", wmodes[s], xm, pct);
            env.send_vectors(nvec / 3, xm, pct, lbl);
        }
        if (s % 2) env.reset_in_flight();
    }
    env.drain();

    int asserts = Verilated::threadContextp()->errorCount();
    std::printf("\nvectors checked  : %llu   (N=%d, latency %d cycles)\n", (unsigned long long)env.checked, N, LAT);
    std::printf("output elements  : %llu normal, %llu zero, %llu inf, %llu nan\n", (unsigned long long)env.n_norm,
                (unsigned long long)env.n_zero, (unsigned long long)env.n_inf, (unsigned long long)env.n_nan);
    std::printf("scoreboard fails : %llu\n", (unsigned long long)env.mismatches);
    std::printf("assertion fails  : %d\n", asserts);
    if (env.mismatches) std::printf("first failure    : after %llu good vectors, in phase '%s'\n",
                                    (unsigned long long)env.first_op, env.first_label);
    bool pass = env.mismatches == 0 && asserts == 0 && env.checked > 0;
    std::printf("\n%s  (seed=%llu)\n", pass ? "TEST PASSED" : "TEST FAILED", (unsigned long long)seed);

    top->final();
#if VM_COVERAGE
    VerilatedCov::write("coverage.dat");
#endif
    delete top;
    return pass ? 0 : 1;
}
