# bf16 MAC verification

This is a small project I made to practice design verification. It has a bfloat16 multiply-accumulate unit (MAC) and a 4x4 systolic array built out of those MACs. Most of the work went into the testbench, not the hardware.

The MAC computes `acc + a*b`, where a and b are bfloat16 and acc and the result are 32-bit floats. It rounds once at the end. It takes 3 cycles and can accept a new operation every cycle.

## Files

- `rtl/bf16_mac.sv` - the MAC, 3 pipeline stages
- `rtl/bf16_systolic.sv` - the array (weights stay in place, inputs flow right, sums flow down)
- `model/bf16_mac_model.h` - golden model in C++
- `tb/tb_mac.cpp` - testbench for the MAC
- `tb/tb_array.cpp` - testbench for the array
- `tb/stimulus.h` - random stimulus generator
- `tb/coverage.h` - functional coverage
- `tb/mac_checker.sv`, `tb/array_checker.sv` - SystemVerilog assertions
- `tb/mac_top.sv`, `tb/array_top.sv` - wrappers that connect the design and the assertions
- `tb/model_selftest.cpp` - tests the golden model itself
- `scripts/run_mutations.sh` - bug injection script
- `docs/SPEC.md` - what the design is supposed to do

## How to run

You need Verilator 5, g++, and make. I used Verilator 5.020.

    make model-test
    make run
    make array
    make bugs

`make model-test` only needs g++. `make run` and `make array` take `SEED=` if you want a different random seed.

## How it's tested

First I checked the golden model. It has some hand-worked cases, and it is also compared against the C++ `fmaf` function on about 6 million random inputs. I did this before using it to check the hardware.

The MAC testbench sends in a few kinds of stimulus:
- every combination of special values (zero, infinity, NaN, denormals, biggest and smallest numbers)
- every possible bfloat16 value on each multiplier input
- random operations in 11 different groups, each aimed at a hard case (cancellation, ties in rounding, overflow, underflow, and so on), with random gaps and some resets in the middle

Every result is compared bit for bit with the golden model, and the testbench also checks that each result shows up exactly 3 cycles later. The assertions check that the latency stays fixed, that no denormal numbers come out, and that any NaN is the standard one.

For the array, the expected output is a chain of golden MAC calls. There is one extra check that doesn't use the model: if the weights are an identity matrix, the output has to equal the input.

To make sure the testbench can actually fail, `make bugs` puts 8 different bugs into the RTL one at a time (wrong tie rounding, missing sticky bit, wrong underflow edge, and so on) and checks that each one gets caught.

## Results

With seed 1:

- MAC: about 420,000 operations checked, no failures, all 166 coverage bins hit
- Array (4x4): 37,946 vectors checked, no failures
- Bug injection: all 8 bugs caught

The output of the bug test is saved in `docs/mutation_results.txt`.

## Things to know

- The behavior is simplified. Denormal inputs are treated as zero and tiny results are flushed to zero. NaN results are always 0x7FC00000.
- The testbench is C++ with Verilator, not UVM. It has the same parts (driver, monitor, scoreboard, coverage), just not in the UVM framework.
- In `tb/array_top.sv` the array ports are flattened to plain vectors, because Verilator 5.020 generated broken code for array ports on the top module.
- Verilator 5.020 also doesn't support `##` sequences in cover properties, so those use `$past` instead.
- A newer Verilator might print different lint warnings. The build treats warnings as errors.

## Ideas for later

- Redo the testbench in SystemVerilog or UVM
- Add formal checks
- Run the RTL through Yosys or OpenROAD to get area and timing
- Support denormals
