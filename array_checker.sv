// SVA for the systolic array: fixed latency of 6N-3 cycles, no spurious y_valid,
// and the FTZ / canonical-NaN invariants hold on every output element.
module array_checker #(
    parameter int N = 4
) (
    input logic        clk,
    input logic        rst_n,
    input logic        x_valid,
    input logic        y_valid,
    input logic [31:0] y_out [N]
);
    localparam int LAT = 6 * N - 3;

    // after a reset the $past history still holds pre-reset traffic, so skip the latency
    // check until LAT clean cycles have passed
    logic [7:0] age;
    always_ff @(posedge clk) begin
        if (!rst_n)         age <= 8'd0;
        else if (age < 8'(LAT)) age <= age + 8'd1;
    end

    // A reset flushes the pipeline, so $past(x_valid, LAT) is only meaningful once the whole
    // look-back window lies after the reset: wait LAT cycles after reset is released.
    int unsigned since_reset;
    always_ff @(posedge clk) begin
        if (!rst_n)              since_reset <= 0;
        else if (since_reset < LAT) since_reset <= since_reset + 1;
    end

    /* verilator lint_off TICKCOUNT */   // a 6N-3 deep $past is cheap for small N
    property p_latency;
        @(posedge clk) disable iff (!rst_n || since_reset < LAT)
            y_valid == $past(x_valid, LAT);
    endproperty
    a_latency: assert property (p_latency)
        else $error("array latency/valid mismatch: y_valid != x_valid delayed by %0d", LAT);
    /* verilator lint_on TICKCOUNT */

    for (genvar c = 0; c < N; c++) begin : g_col
        property p_no_denormal;
            @(posedge clk) disable iff (!rst_n)
                y_valid |-> !((y_out[c][30:23] == 8'h00) && (y_out[c][22:0] != 23'h0));
        endproperty
        a_no_denormal: assert property (p_no_denormal)
            else $error("column %0d produced a denormal: %h", c, y_out[c]);

        property p_canonical_nan;
            @(posedge clk) disable iff (!rst_n)
                (y_valid && y_out[c][30:23] == 8'hFF && y_out[c][22:0] != 23'h0) |-> (y_out[c] == 32'h7FC0_0000);
        endproperty
        a_canonical_nan: assert property (p_canonical_nan)
            else $error("column %0d non-canonical NaN: %h", c, y_out[c]);
    end
endmodule
