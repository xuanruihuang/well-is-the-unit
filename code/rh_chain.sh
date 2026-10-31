#!/bin/sh
# Usage: rh_chain.sh <orig|clean> [zeroshot]   runs splits 5..9 one after another
cd $WORKSPACE/rt/samewell/rhaister/code
export PYTHONPATH=$WORKSPACE/rt/samewell/rhaister/code:$WORKSPACE/rt/samewell/pylib_rh:$WORKSPACE/rt/samewell/pylib_bp:$WORKSPACE/pylib
export TORCHDYNAMO_DISABLE=1
for k in 5 6 7 8 9; do
  python3 -u ../rh_clean.py run $1 $k $2 > ../out/run_$1${2:+_zs}_$k.log 2>&1
done
