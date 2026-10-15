// Functional coverage: "did the tests actually exercise the scenarios we care about?"
// Bins are declared up front (the coverage *plan*), then hit as stimulus is applied.
// Anything declared but never hit shows up as a hole in the report.
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include "../model/bf16_mac_model.h"

class Coverage {
public:
    void declare(const std::string& group, const std::string& bin) {
        auto& g = groups_[group];
        g.order.push_back(bin);
        g.hits[bin] = 0;
    }
    void hit(const std::string& group, const std::string& bin) {
        auto& g = groups_[group];
        auto it = g.hits.find(bin);
        if (it == g.hits.end()) { g.order.push_back(bin); g.hits[bin] = 1; undeclared_++; }
        else it->second++;
    }
    // returns overall percentage
    double report(FILE* f, bool verbose) const {
        uint64_t total = 0, covered = 0;
        std::fprintf(f, "\n================ FUNCTIONAL COVERAGE ================\n");
        for (auto& kv : groups_) {
            uint64_t gt = 0, gc = 0;
            for (auto& b : kv.second.order) { gt++; if (kv.second.hits.at(b) > 0) gc++; }
            total += gt; covered += gc;
            std::fprintf(f, "%-14s %4llu / %-4llu bins  (%5.1f%%)\n", kv.first.c_str(),
                         (unsigned long long)gc, (unsigned long long)gt, gt ? 100.0 * gc / gt : 100.0);
            for (auto& b : kv.second.order) {
                uint64_t h = kv.second.hits.at(b);
                if (h == 0) std::fprintf(f, "    HOLE   %s\n", b.c_str());
                else if (verbose) std::fprintf(f, "    %8llu  %s\n", (unsigned long long)h, b.c_str());
            }
        }
        double pct = total ? 100.0 * covered / total : 100.0;
        std::fprintf(f, "TOTAL          %4llu / %-4llu bins  (%5.1f%%)\n", (unsigned long long)covered,
                     (unsigned long long)total, pct);
        return pct;
    }
private:
    struct Group { std::vector<std::string> order; std::map<std::string, uint64_t> hits; };
    std::map<std::string, Group> groups_;
    int undeclared_ = 0;
};

namespace covdef {
inline const char* cls16(uint16_t x) {
    int e = (x >> 7) & 0xFF, f = x & 0x7F;
    if (e == 0xFF) return f ? "NAN" : "INF";
    return e == 0 ? (f ? "DENORM" : "ZERO") : "NORMAL";
}
inline const char* cls32(uint32_t x) {
    int e = (x >> 23) & 0xFF; uint32_t f = x & 0x7FFFFF;
    if (e == 0xFF) return f ? "NAN" : "INF";
    return e == 0 ? (f ? "DENORM" : "ZERO") : "NORMAL";
}
inline const char* expdiff_bucket(int d) {
    if (d < -27) return "<-27";
    if (d < -2)  return "-27..-3";
    if (d < 0)   return "-2..-1";
    if (d == 0)  return "0";
    if (d <= 2)  return "1..2";
    if (d <= 27) return "3..27";
    return ">27";
}

inline void declare_all(Coverage& cov) {
    static const char* K[] = { "ZERO", "DENORM", "NORMAL", "INF", "NAN" };
    for (auto a : K) for (auto b : K) for (auto c : K)
        cov.declare("operand_cross", std::string("a=") + a + " b=" + b + " c=" + c);
    static const char* B[] = { "<-27", "-27..-3", "-2..-1", "0", "1..2", "3..27", ">27" };
    for (auto d : B) for (auto op : { "add", "sub" })
        cov.declare("exp_diff", std::string("diff ") + d + " " + op);
    for (auto r : { "nan", "inf_from_input", "overflow_pos", "overflow_neg", "zero_special_pos", "zero_special_neg",
                    "exact_cancel_pos", "flush_underflow_pos", "flush_underflow_neg", "normal_pos", "normal_neg" })
        cov.declare("result_class", r);
    for (auto r : { "exact", "inexact_down", "inexact_up", "tie_to_even_down", "tie_to_even_up", "round_carry_out" })
        cov.declare("rounding", r);
    for (auto r : { "carry_out_of_add", "no_shift", "left_1", "left_2-4", "left_5-10", "left_11+" })
        cov.declare("normalise", r);
    for (auto r : { "back_to_back_run>=8", "isolated_op", "gap_after_run", "reset_with_ops_in_flight" })
        cov.declare("pipeline", r);
}

inline void sample(Coverage& cov, uint16_t a, uint16_t b, uint32_t c, const ref::Result& r) {
    cov.hit("operand_cross", std::string("a=") + cls16(a) + " b=" + cls16(b) + " c=" + cls32(c));
    const ref::Flags& f = r.f;
    uint32_t bits = r.bits;
    bool neg = bits >> 31;
    int e = (bits >> 23) & 0xFF;
    if (f.has_exp_diff) cov.hit("exp_diff", std::string("diff ") + expdiff_bucket(f.exp_diff) + (f.eff_sub ? " sub" : " add"));

    if (e == 0xFF && (bits & 0x7FFFFF)) cov.hit("result_class", "nan");
    else if (e == 0xFF && f.overflow) cov.hit("result_class", neg ? "overflow_neg" : "overflow_pos");
    else if (e == 0xFF) cov.hit("result_class", "inf_from_input");
    else if (f.exact_cancel) cov.hit("result_class", "exact_cancel_pos");
    else if (f.underflow_flush) cov.hit("result_class", neg ? "flush_underflow_neg" : "flush_underflow_pos");
    else if (e == 0) cov.hit("result_class", neg ? "zero_special_neg" : "zero_special_pos");
    else cov.hit("result_class", neg ? "normal_neg" : "normal_pos");

    if (!f.special && !f.exact_cancel && !f.overflow && !f.underflow_flush) {
        if (f.round_carry) cov.hit("rounding", "round_carry_out");
        else if (f.tie) cov.hit("rounding", f.round_up ? "tie_to_even_up" : "tie_to_even_down");
        else if (f.inexact) cov.hit("rounding", f.round_up ? "inexact_up" : "inexact_down");
        else cov.hit("rounding", "exact");
    }
    if (f.has_exp_diff && !f.exact_cancel) {
        int s = f.norm_shift;
        if (s < 0) cov.hit("normalise", "carry_out_of_add");
        else if (s == 0) cov.hit("normalise", "no_shift");
        else if (s == 1) cov.hit("normalise", "left_1");
        else if (s <= 4) cov.hit("normalise", "left_2-4");
        else if (s <= 10) cov.hit("normalise", "left_5-10");
        else cov.hit("normalise", "left_11+");
    }
}
}  // namespace covdef
