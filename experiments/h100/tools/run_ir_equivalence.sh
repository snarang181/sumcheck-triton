#!/usr/bin/env bash
# IR equivalence of stock vs slice-cache-patched Triton 3.7.0: same kernels, fresh caches,
# SHA-256 of ttir/ttgir/llir/ptx/cubin. Usage: run_ir_equivalence.sh LABEL PYTHON
cd "$(dirname "$0")/../../.."
source experiments/h100/env_table5.sh
label=$1; py=$2
OUT=experiments/h100/diag/ir_equivalence.jsonl
CASES_SPEC="zk_spartan_2:32:0 zk_spartan_2:32:1 zk_spartan_2:32:2 zk_witness_id_point_1:32:2 zk_witness_id_point_1:32:5
 zk_complete_add_3:32:2 zk_complete_add_3:32:6 zk_vanilla_permcheck_hp:32:2 zk_vanilla_permcheck_hp:32:5
 zk_opencheck:32:2 zk_opencheck:32:8 zk_spartan_2:64:2 zk_opencheck:64:4 zk_spartan_2:128:2
 zk_spartan_2:256:0 zk_spartan_2:256:2 zk_witness_id_point_1:256:2
 zk_jellyfish_zerocheck_hp:32:0 zk_jellyfish_zerocheck_hp:32:3 zk_jellyfish_zerocheck_hp:32:7 zk_jellyfish_zerocheck_hp:64:7"
CASES_GEN="zk_spartan_2:32:2 zk_opencheck:32:4 zk_spartan_2:256:2"
for pm in specialized generic; do
  cases=$CASES_SPEC; [ $pm = generic ] && cases=$CASES_GEN
  cache=$(mktemp -d)
  TRITON_CACHE_DIR=$cache nice -n 10 $py experiments/h100/tools/ir_hashes.py --label "$label" --point-mode $pm --cases $cases >> $OUT
  rm -rf "$cache"
done
echo "IR_EQUIVALENCE_DONE $label"
