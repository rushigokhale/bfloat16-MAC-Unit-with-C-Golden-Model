// Verification wrapper: DUT + assertion checker. This is what Verilator compiles.
module mac_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  logic [15:0] a,
    input  logic [15:0] b,
    input  logic [31:0] acc,
    output logic        out_valid,
    output logic [31:0] result
);
    bf16_mac dut (.*);
    mac_checker chk (.*);
endmodule
