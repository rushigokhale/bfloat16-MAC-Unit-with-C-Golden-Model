# bf16 MAC design-verification project
#   make model-test   check the golden model (hand cases + fmaf cross-check)
#   make run          build + run the MAC testbench (SEED=.. OPS=..)
#   make array        build + run the systolic-array testbench
#   make bugs         mutation test: inject bugs one at a time, confirm the testbench catches each
#   make codecov      run MAC tb with Verilator code coverage and write an annotated report
#   make all          model-test + run + array + bugs
#   make clean

SEED ?= 1
OPS  ?= 400000
N    ?= 4

VFLAGS = -Wall -Wno-DECLFILENAME --assert --coverage -O3 -j 0 \
         -CFLAGS "-O2 -std=c++17 -ffp-contract=off -DVM_COVERAGE=1"
RTL_MAC = rtl/bf16_mac.sv
MAC_SRC = $(RTL_MAC) tb/mac_checker.sv tb/mac_top.sv tb/tb_mac.cpp
ARR_SRC = $(RTL_MAC) rtl/bf16_systolic.sv tb/array_checker.sv tb/array_top.sv tb/tb_array.cpp
HDRS    = model/bf16_mac_model.h tb/stimulus.h tb/coverage.h

.PHONY: all model-test run array bugs codecov clean
all: model-test run array bugs

model-test: tb/model_selftest.cpp $(HDRS)
	g++ -O2 -std=c++17 -ffp-contract=off -Wall -o build_model_selftest tb/model_selftest.cpp
	./build_model_selftest

obj_mac/Vmac_top: $(MAC_SRC) $(HDRS)
	verilator --cc --exe --build --top-module mac_top --Mdir obj_mac $(VFLAGS) $(MAC_SRC)

run: obj_mac/Vmac_top
	./obj_mac/Vmac_top +seed=$(SEED) +ops=$(OPS)

obj_arr/Varray_top: $(ARR_SRC) $(HDRS)
	verilator --cc --exe --build --top-module array_top --Mdir obj_arr -GN=$(N) $(VFLAGS) \
	  -CFLAGS "-DARRAY_N=$(N)" $(ARR_SRC)

array: obj_arr/Varray_top
	./obj_arr/Varray_top +seed=$(SEED)

bugs:
	bash scripts/run_mutations.sh

codecov: obj_mac/Vmac_top
	cd obj_mac && ./Vmac_top +seed=$(SEED) +ops=$(OPS) +quiet > /dev/null
	verilator_coverage --annotate obj_mac/annotate --annotate-min 1 obj_mac/coverage.dat
	@echo "annotated sources in obj_mac/annotate (lines marked % were never executed)"

clean:
	rm -rf obj_mac obj_arr obj_bug_* build_model_selftest coverage.dat
