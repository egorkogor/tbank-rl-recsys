#!/usr/bin/env bash
# After the full val evaluation: selection rule, comparisons, decomposition, RL suboptimality vs pi*,
# five steps ahead, and task 1's booster as an external judge. Steps with an existing output are skipped.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
PY1=D:/PdM_Projects/tbank-recsys-autofeatures/.venv/Scripts/python.exe   # has lightgbm; task 1 files read only
E=../artifacts/runs/eval/val
L=../logs/analysis.log
while [ ! -f ../logs/queue_evals.done ]; do sleep 60; done
[ -f $E/selection.json ] || $PY select_point.py >> $L 2>&1
for k in lp minutes balanced skip wtg quant; do [ -f $E/compare_$k.json ] || $PY compare.py --kind $k >> $L 2>&1; done
specs=$(ls $E | grep -E '^(rl|pistar|copyscale|personal|mostpop)__.*\.npz$' | sed 's/\.npz$//; s/__/:/g')
[ -f $E/decompose.json ] || $PY decompose.py $specs >> $L 2>&1
t1=$(ls -d ../artifacts/runs/rl/*_T1/ | xargs -n1 basename | grep -v -E '^(calib|smoke)' | sed 's/$/:400/')
[ -f $E/suboptimality.json ] || $PY suboptimality.py $t1 >> $L 2>&1
choice=$($PY -c "import json;print(json.load(open('$E/selection.json'))['choice'])")
ms="sft rl:lp_b0.1_T5:400 rl:lp_b0.1_T1:400 rl:lp_b0.02_T1:400 rl:balanced_b5_T1:400"
case $choice in rl:*) ms="$ms $choice";; esac
[ -f $E/multistep.json ] || $PY eval_multistep.py $ms >> $L 2>&1
bj="sft personal copyscale:4 pistar:lp:0.1 pistar:balanced:5 pistar:wtg:0.5 rl:lp_b0.1_T1:400 rl:lp_b0.02_T1:400 rl:lp_b0_T1:400"
bj="$bj rl:minutes_b0.5_T1:400 rl:wtg_b0.5_T1:400 rl:balanced_b5_T1:400 rl:balanced_b2_T1:400 rl:mix_b0.1_T1:400"
case $choice in sft|personal|copyscale:4|pistar:lp:0.1|pistar:balanced:5|pistar:wtg:0.5) ;; *) bj="$bj $choice";; esac
# task 1's code is read-only: no bytecode is written into its repository
[ -f $E/booster_judge_b0.json ] || PYTHONDONTWRITEBYTECODE=1 TBREC_CORES=0-2 $PY1 booster_judge.py $bj >> $L 2>&1
echo done > ../logs/queue_analysis.done
