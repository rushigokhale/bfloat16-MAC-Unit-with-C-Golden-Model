// Verification wrapper for the systolic array: DUT + assertion checker.
// The DUT keeps its clean unpacked-array interface; this wrapper flattens x/y into packed
// vectors only because Verilator 5.020 mis-generates code for unpacked-array top-level ports.
//   x_in_flat[16*i +: 16] = x[i]      y_out_flat[32*c +: 32] = y[c]
module array_top #(
    parameter int N = 4
) (
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 w_we,
    input  logic [$clog2(N)-1:0] w_row,
    input  logic [$clog2(N)-1:0] w_col,
    input  logic [15:0]          w_data,
    input  logic                 x_valid,
    input  logic [N*16-1:0]      x_in_flat,
    output logic                 y_valid,
    output logic [N*32-1:0]      y_out_flat
);
    logic [15:0] x_in  [N];
    logic [31:0] y_out [N];
    for (genvar i = 0; i < N; i++) begin : g_flat
        assign x_in[i] = x_in_flat[16*i +: 16];
        assign y_out_flat[32*i +: 32] = y_out[i];
    end

    bf16_systolic #(.N(N)) dut (.*);
    array_checker #(.N(N)) chk (.*);
endmodule
