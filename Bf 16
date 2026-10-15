// bf16 multiply-accumulate:  result = acc + a*b  (a,b bfloat16; acc,result binary32)
// 3-stage pipeline, one operation per cycle, fixed latency of 3 cycles.
// See docs/SPEC.md for the exact numeric contract.
//
//   Stage 1: decode, special-value handling, 8x8 multiply, normalise product
//   Stage 2: order operands, align the smaller one (guard/round/sticky), add or subtract
//   Stage 3: normalise (leading-zero shift), round-to-nearest-even, flush/overflow, pack
//
// Optional bug injection for the mutation-testing flow (never defined in normal builds):
//   BUG_TIE_UP        round ties up instead of to-even
//   BUG_NO_STICKY     drop the sticky bit while aligning
//   BUG_FTZ_EDGE      flush only when biased exponent < 0 (instead of <= 0)
//   BUG_OVF_EDGE      overflow only when biased exponent > 255 (instead of >= 255)
//   BUG_INF_TIMES_0   inf * 0 returns infinity instead of NaN
//   BUG_ZERO_SIGN     exact cancellation returns the sign of the larger operand
//   BUG_ROUND_CARRY   forget to bump the exponent when rounding carries out
//   BUG_DAZ           denormal acc is not treated as zero
module bf16_mac (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  logic [15:0] a,
    input  logic [15:0] b,
    input  logic [31:0] acc,
    output logic        out_valid,
    output logic [31:0] result
);
    localparam logic [31:0] QNAN = 32'h7FC0_0000;

    // =====================================================================
    // Stage 1: decode + multiply
    // =====================================================================
    logic        a_s, b_s, c_s;
    logic [7:0]  a_e, b_e, c_e;
    logic [6:0]  a_f, b_f;
    logic [22:0] c_f;
    assign {a_s, a_e, a_f} = a;
    assign {b_s, b_e, b_f} = b;
    assign {c_s, c_e, c_f} = acc;

    logic a_zero, b_zero, c_zero, a_inf, b_inf, c_inf, a_nan, b_nan, c_nan;
    assign a_zero = (a_e == 8'h00);                      // denormals count as zero (DAZ)
    assign b_zero = (b_e == 8'h00);
`ifdef BUG_DAZ
    assign c_zero = (acc[30:0] == 31'h0);
`else
    assign c_zero = (c_e == 8'h00);
`endif
    assign a_inf  = (a_e == 8'hFF) && (a_f == 7'h0);
    assign b_inf  = (b_e == 8'hFF) && (b_f == 7'h0);
    assign c_inf  = (c_e == 8'hFF) && (c_f == 23'h0);
    assign a_nan  = (a_e == 8'hFF) && (a_f != 7'h0);
    assign b_nan  = (b_e == 8'hFF) && (b_f != 7'h0);
    assign c_nan  = (c_e == 8'hFF) && (c_f != 23'h0);

    logic p_s;
    assign p_s = a_s ^ b_s;

    // 8x8 -> 16 bit exact mantissa product (hidden ones included)
    logic [15:0] prod_m;
    logic        prod_top;
    logic [23:0] p_mant;
    logic signed [11:0] p_exp;
    assign prod_m   = {1'b1, a_f} * {1'b1, b_f};
    assign prod_top = prod_m[15];
    assign p_mant   = prod_top ? {prod_m, 8'h00} : {prod_m[14:0], 9'h000};
    assign p_exp    = $signed({4'b0000, a_e}) + $signed({4'b0000, b_e}) - 12'sd127
                      + (prod_top ? 12'sd1 : 12'sd0);

    // acc as a normalised operand (zero => exponent far below everything, mantissa 0)
    logic signed [11:0] c_exp_n;
    logic [23:0]        c_mant_n;
    assign c_exp_n  = c_zero ? -12'sd1024 : $signed({4'b0000, c_e});
    assign c_mant_n = c_zero ? 24'h0 : {1'b1, c_f};

    // special results bypass the arithmetic path
    logic        special;
    logic [31:0] special_res;
    logic        nan_case, prod_inf, prod_zero;
    assign prod_zero = a_zero | b_zero;
    always_comb begin
        nan_case    = a_nan | b_nan | c_nan | (a_inf & b_zero) | (a_zero & b_inf);
        prod_inf    = a_inf | b_inf;
        special     = 1'b1;
        special_res = QNAN;
        if (nan_case) begin
`ifdef BUG_INF_TIMES_0
            if (!(a_nan | b_nan | c_nan)) special_res = {p_s, 8'hFF, 23'h0};
            else special_res = QNAN;
`else
            special_res = QNAN;
`endif
        end else if (prod_inf) begin
            if (c_inf && (c_s != p_s)) special_res = QNAN;
            else                       special_res = {p_s, 8'hFF, 23'h0};
        end else if (c_inf) begin
            special_res = {c_s, 8'hFF, 23'h0};
        end else if (prod_zero) begin
            if (c_zero) special_res = {(p_s & c_s), 31'h0};    // -0 only if both are -0
            else        special_res = acc;                     // acc passes through exactly
        end else begin
            special     = 1'b0;                                // normal arithmetic path
            special_res = 32'h0;
        end
    end

    // ---- stage 1 registers ----
    logic               s1_valid, s1_special;
    logic [31:0]        s1_special_res;
    logic               s1_ps, s1_cs;
    logic signed [11:0] s1_pexp, s1_cexp;
    logic [23:0]        s1_pmant, s1_cmant;

    always_ff @(posedge clk) begin
        if (!rst_n) s1_valid <= 1'b0;
        else        s1_valid <= in_valid;
    end
    always_ff @(posedge clk) begin
        s1_special     <= special;
        s1_special_res <= special_res;
        s1_ps          <= p_s;
        s1_cs          <= c_s;
        s1_pexp        <= p_exp;
        s1_cexp        <= c_exp_n;
        s1_pmant       <= p_mant;
        s1_cmant       <= c_mant_n;
    end

    // =====================================================================
    // Stage 2: order, align, add/subtract  (27-bit datapath: 24 mant + G,R,S)
    // =====================================================================
    logic               p_bigger;
    logic               big_s, sml_s;
    logic signed [11:0] big_e, sml_e;
    logic [23:0]        big_m, sml_m;
    logic signed [11:0] d;
    logic [26:0]        sml_ext, sml_shift, big_ext, sml_al;
    logic               sticky, eff_sub;
    logic [27:0]        sum28;

    always_comb begin
        // larger magnitude first (exponent, then mantissa), so the subtraction never goes negative
        p_bigger = (s1_pexp > s1_cexp) || ((s1_pexp == s1_cexp) && (s1_pmant >= s1_cmant));
        big_s = p_bigger ? s1_ps    : s1_cs;
        sml_s = p_bigger ? s1_cs    : s1_ps;
        big_e = p_bigger ? s1_pexp  : s1_cexp;
        sml_e = p_bigger ? s1_cexp  : s1_pexp;
        big_m = p_bigger ? s1_pmant : s1_cmant;
        sml_m = p_bigger ? s1_cmant : s1_pmant;
        d     = big_e - sml_e;                              // >= 0

        sml_ext = {sml_m, 3'b000};
        if (d >= 12'sd27) begin
            sml_shift = 27'h0;
            sticky    = (sml_m != 24'h0);
        end else begin
            sml_shift = sml_ext >> d[4:0];
            sticky    = |(sml_ext & ~(27'h7FF_FFFF << d[4:0]));
        end
`ifdef BUG_NO_STICKY
        sticky = 1'b0;
`endif
        sml_al  = {sml_shift[26:1], sml_shift[0] | sticky};
        big_ext = {big_m, 3'b000};
        eff_sub = (big_s != sml_s);
        sum28   = eff_sub ? ({1'b0, big_ext} - {1'b0, sml_al})
                          : ({1'b0, big_ext} + {1'b0, sml_al});
    end

    // ---- stage 2 registers ----
    logic               s2_valid, s2_special;
    logic [31:0]        s2_special_res;
    logic               s2_sign;
    logic signed [11:0] s2_exp;
    logic [27:0]        s2_sum;

    always_ff @(posedge clk) begin
        if (!rst_n) s2_valid <= 1'b0;
        else        s2_valid <= s1_valid;
    end
    always_ff @(posedge clk) begin
        s2_special     <= s1_special;
        s2_special_res <= s1_special_res;
        s2_sign        <= big_s;
        s2_exp         <= big_e;
        s2_sum         <= sum28;
    end

    // =====================================================================
    // Stage 3: normalise, round, pack
    // =====================================================================
    logic [4:0]         lz;                                 // leading zeros of s2_sum (28 bit)
    logic [26:0]        norm;
    logic signed [12:0] exp_n;
    /* verilator lint_off UNUSEDSIGNAL */
    logic               g_bit, s_bit, l_bit, round_up;   // s_bit/l_bit unused when BUG_TIE_UP is defined
    /* verilator lint_on UNUSEDSIGNAL */
    /* verilator lint_off UNUSEDSIGNAL */
    logic [24:0]        mant25;           // bit 23 is the hidden one, never read
    /* verilator lint_on UNUSEDSIGNAL */
    logic signed [12:0] exp_f;
    logic [31:0]        final_res;

    always_comb begin
        lz = 5'd28;
        for (int i = 0; i < 28; i++)
            if (s2_sum[i]) lz = 5'(27 - i);

        if (s2_sum[27])  norm = {s2_sum[27:2], s2_sum[1] | s2_sum[0]};   // carry out: shift right, keep sticky
        else             norm = 27'(s2_sum[26:0] << (lz - 5'd1));        // cancellation: shift left

        // exponent of the normalised value (bit 26 == leading one)
        exp_n = 13'(s2_exp) + 13'sd1 - $signed({8'b0, lz});

        g_bit = norm[2];
        s_bit = norm[1] | norm[0];
        l_bit = norm[3];
`ifdef BUG_TIE_UP
        round_up = g_bit;
`else
        round_up = g_bit & (s_bit | l_bit);
`endif
        mant25 = {1'b0, norm[26:3]} + {24'h0, round_up};
`ifdef BUG_ROUND_CARRY
        exp_f = exp_n;
`else
        exp_f = mant25[24] ? exp_n + 13'sd1 : exp_n;
`endif

        if (s2_sum == 28'h0) begin
            final_res = 32'h0000_0000;                                   // exact cancellation -> +0
`ifdef BUG_ZERO_SIGN
            final_res = {s2_sign, 31'h0};
`endif
`ifdef BUG_OVF_EDGE
        end else if (exp_f > 13'sd255) begin
`else
        end else if (exp_f >= 13'sd255) begin
`endif
            final_res = {s2_sign, 8'hFF, 23'h0};                         // overflow -> inf
`ifdef BUG_FTZ_EDGE
        end else if (exp_f < 13'sd0) begin
`else
        end else if (exp_f <= 13'sd0) begin
`endif
            final_res = {s2_sign, 31'h0};                                // underflow -> flush to zero
        end else begin
            final_res = {s2_sign, exp_f[7:0], mant25[24] ? 23'h0 : mant25[22:0]};
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) out_valid <= 1'b0;
        else        out_valid <= s2_valid;
    end
    always_ff @(posedge clk) begin
        result <= s2_special ? s2_special_res : final_res;
    end
endmodule
