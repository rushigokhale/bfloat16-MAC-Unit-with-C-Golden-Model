// SystemVerilog assertions for the bf16 MAC. Instantiated next to the DUT in mac_top.sv.
// These check protocol/timing and output invariants independently of the C++ scoreboard,
// which checks the numeric value of every result.
module mac_checker (
    input logic        clk,
    input logic        rst_n,
    input logic        in_valid,
    input logic        out_valid,
    input logic [31:0] result
);
    // fixed latency of 3: out_valid is exactly in_valid delayed by 3 cycles
    property p_latency;
        @(posedge clk) disable iff (!rst_n)
            out_valid == $past(in_valid, 3);
    endproperty
    a_latency: assert property (p_latency)
        else $error("latency/valid mismatch: out_valid != in_valid delayed by 3");

    // FTZ: a valid result is never a denormal
    property p_no_denormal;
        @(posedge clk) disable iff (!rst_n)
            out_valid |-> !((result[30:23] == 8'h00) && (result[22:0] != 23'h0));
    endproperty
    a_no_denormal: assert property (p_no_denormal)
        else $error("denormal result produced: %h", result);

    // NaN output is always the canonical quiet NaN
    property p_canonical_nan;
        @(posedge clk) disable iff (!rst_n)
            (out_valid && result[30:23] == 8'hFF && result[22:0] != 23'h0) |-> (result == 32'h7FC0_0000);
    endproperty
    a_canonical_nan: assert property (p_canonical_nan)
        else $error("non-canonical NaN: %h", result);

    // coverage points the simulator can also report
    c_out_valid: cover property (@(posedge clk) disable iff (!rst_n) out_valid);
    c_back_to_back: cover property (@(posedge clk) disable iff (!rst_n)
        out_valid && $past(out_valid, 1) && $past(out_valid, 2));
    c_bubble: cover property (@(posedge clk) disable iff (!rst_n)
        out_valid && !$past(out_valid, 1) && $past(out_valid, 2));
endmodule
