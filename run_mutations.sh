#!/usr/bin/env bash
# Mutation testing: inject one RTL bug at a time and check the testbench notices.
# A testbench that never fails proves nothing; this shows it *can* fail, and which
# stimulus class / checker catches each bug.
set -u
cd "$(dirname "$0")/.."
OPS=${OPS:-200000}
SEED=${SEED:-1}
BUGS=(BUG_TIE_UP BUG_NO_STICKY BUG_FTZ_EDGE BUG_OVF_EDGE BUG_INF_TIMES_0 BUG_ZERO_SIGN BUG_ROUND_CARRY BUG_DAZ)
DESC=(
  "round ties up instead of to-even"
  "drop sticky bit during alignment"
  "flush only if exp < 0 (should be <= 0)"
  "overflow only if exp > 255 (should be >= 255)"
  "inf * 0 returns inf instead of NaN"
  "exact cancellation returns signed zero"
  "no exponent bump when rounding carries out"
  "denormal acc not treated as zero"
)
OUT=docs/mutation_results.txt
{
echo "Mutation test results (seed=$SEED, $OPS random ops)"
echo "  full      = directed + exhaustive + random phases"
echo "  rand-only = random phase only; shows which constrained-random class finds each bug first"
printf "%-16s %-44s %-7s %9s %8s  %-10s %s\n" "bug" "description" "caught" "sb_fails" "assert" "rand-only" "first rand class to catch it"
} | tee $OUT
caught=0
for i in "${!BUGS[@]}"; do
  b=${BUGS[$i]}
  d=obj_bug_$b
  verilator --cc --exe --build --top-module mac_top --Mdir $d -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL --assert -O3 -j 0 \
    +define+$b -CFLAGS "-O2 -std=c++17 -ffp-contract=off" \
    rtl/bf16_mac.sv tb/mac_checker.sv tb/mac_top.sv tb/tb_mac.cpp > $d.build.log 2>&1 || { echo "build failed for $b (see $d.build.log)"; continue; }
  log=$d/run.log
  ./$d/Vmac_top +seed=$SEED +ops=$OPS +quiet > $log 2>&1
  sb=$(grep -E '^scoreboard fails' $log | awk '{print $4}')
  af=$(grep -E '^assertion fails' $log | awk '{print $4}')
  first=$(grep -E '^first failure' $log | sed -E "s/.*class '([^']*)'.*/\1/")
  if grep -q "TEST FAILED" $log; then res=YES; caught=$((caught+1)); else res="NO!!"; fi
  ./$d/Vmac_top +seed=$SEED +ops=$OPS +quiet +skipdirected > $d/run_rand.log 2>&1
  if grep -q "TEST FAILED" $d/run_rand.log; then rres=YES; else rres="NO"; fi
  rfirst=$(grep -E '^first failure' $d/run_rand.log | sed -E "s/.*class '([^']*)'.*/\1/")
  printf "%-16s %-44s %-7s %9s %8s  %-10s %s\n" "$b" "${DESC[$i]}" "$res" "${sb:-?}" "${af:-?}" "$rres" "${rfirst:--}" | tee -a $OUT
done
echo "caught $caught / ${#BUGS[@]} injected bugs" | tee -a $OUT
[ "$caught" -eq "${#BUGS[@]}" ]
