#!/bin/bash
# The full analysis in the order it depends on (README, "Full analysis"). Each step is one command; steps in the same
# block are independent. Run from the repository root after filling in config.env and placing the inputs (README,
# "Inputs"). GPU steps: bp_train, bp91_train, rh_chain. Expect days on a large server: the Tahoe-100M extraction reads
# about 100 million cells twice.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./config.env; set +a
PY=${PYTHON:-python3}; C=code; T=$WORKSPACE/data_ext/tahoe/Tahoe-100M; SW=$WORKSPACE/rt/samewell
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
mkdir -p "$WORKSPACE/tahoe_agg" "$WORKSPACE/tahoe_program" "$WORKSPACE/rt/tahoe" "$SW"

# 1. Tahoe-100M -> per (cell line, sample, random half) aggregates; cached effect arrays; condition metadata
$PY $C/tahoe_extract.py map "$T/data" "$WORKSPACE/tahoe_agg" "${WORKERS:-16}"
$PY $C/tahoe_extract.py reduce "$WORKSPACE/tahoe_agg"
$PY $C/tahoe_program.py "$WORKSPACE/tahoe_agg" "$WORKSPACE/tahoe_program" "$WORKSPACE/genesets/MSigDB_Hallmark_2020.txt"
$PY $C/rt_tahoe_extract.py
$PY $C/rt_offset_source.py tahoe
$PY $C/rt_tech_a100.py depth

# 2. Raw-count pseudobulks on protein-coding genes (BioPert layout, heat maps, contamination)
$PY $C/pb_extract.py map "${WORKERS:-16}"
$PY $C/pb_extract.py reduce

# 3. Inputs of the label checks
$PY $C/cc_signflip.py
$PY $C/plate13_81.py
$PY $C/label81.py
$PY $C/labels84.py
$PY $C/check85.py

# 4. Same-well analyses (independent of each other once step 3 is done)
$PY $C/core82.py; $PY $C/core82.py confirmed; $PY $C/core82.py nodvia@labels85
$PY $C/perm89.py; $PY $C/stats99.py; $PY $C/str92.py; $PY $C/str92.py dmso
$PY $C/ref88.py; $PY $C/ref88b.py; $PY $C/ref99.py; $PY $C/ref99.py single; $PY $C/ref99.py tsplit
$PY $C/check83.py; $PY $C/ver99.py; $PY $C/p13strong92.py; $PY $C/p13len92.py; $PY $C/p13r99.py; $PY $C/ada_k.py
$PY $C/well99.py; $PY $C/cont99.py; $PY $C/cont99.py 2; $PY $C/corr92.py; $PY $C/ctrl99.py; $PY $C/cc99.py; $PY $C/sim79.py
$PY $C/eval82.py fewshot dprime confirmed; $PY $C/eval82.py fewshot dmso confirmed; $PY $C/eval82.py fewshot dprime confirmed swap
$PY $C/fig110.py; $PY $C/fig110.py labels; $PY $C/fig110qc.py

# 5. BioPert: reproduction of the published compound split (0.79), then the replicate-pair runs; transfer model
$PY $C/bp_data.py; $PY $C/bp_train.py A 42
$PY $C/bp91_data.py
for s in 42 1 2; do $PY $C/bp91_train.py dprime bp $s; done
$PY $C/bp91_train.py dprime ecfp; $PY $C/bp91_train.py dmso bp 42; $PY $C/bp91_train.py dmso ecfp
$PY $C/bp91_sum.py
$PY $C/mol91.py dprime; $PY $C/mol91.py dmso

# 6. Rhaister with its own code and inputs (README, "Rhaister"): clean inputs, five held-out splits few-shot and
#    zero-shot, metrics against the training plate and another plate, replication ceiling
cp $C/rh_clean.py $C/rh_metrics.py $C/config.py "$SW/rhaister/"
$PY "$SW/rhaister/rh_clean.py" build
WORKSPACE=$WORKSPACE sh $C/rh_chain.sh clean; WORKSPACE=$WORKSPACE sh $C/rh_chain.sh clean zeroshot
$PY "$SW/rhaister/rh_metrics.py"; $PY $C/rh99.py

# 7. Figure data
$PY $C/figexport89.py

# 8. Other screens
EM=$WORKSPACE/mainline_rt_samewell/emerald; mkdir -p "$EM"
$PY $C/emerald_scan.py "$EM/scan.json"
$PY $C/emerald_filter.py "$EM/scan.json" "$EM/dmso_tf"
$PY $C/emerald103.py "$EM/dmso_tf" "$EM/emerald103_results.json"
$PY $C/sw_op3.py
$PY $C/sciplex_pb.py; $PY $C/sciplex_checks.py

# 9. Collect the result files into out/data (same names as data/), threshold table, figures
mkdir -p out/data out/figures
cp "$SW"/*.json "$SW"/fig110.npz out/data/ 2>/dev/null || true
$PY $C/thr96.py out/data/figdata89.json out/data/thr96.json
$PY $C/make_figs110.py out/data out/figures
$PY $C/make_figs111.py out/data out/figures
