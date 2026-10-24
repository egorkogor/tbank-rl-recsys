#!/usr/bin/env bash
# After the full val evaluation: selection rule, comparisons, decomposition, RL suboptimality vs pi*,
# five steps ahead, and (last, RAM-heavy, CPU only) task 1's booster as an external judge.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
PY1=D:/PdM_Projects/tbank-recsys-autofeatures/.venv/Scripts/python.exe   # has lightgbm; task 1 files read only
while [ ! -f ../logs/queue_evals.done ]; do sleep 60; done
L=../logs/analysis.log
$PY select_point.py >> $L 2>&1
for k in lp minutes balanced skip wtg; do $PY compare.py --kind $k >> $L 2>&1; done
specs=$(ls ../artifacts/runs/eval/val/ | grep -E '^(rl|pistar|copyscale|personal|mostpop)__.*\.npz$' | sed 's/\.npz$//; s/__/:/g')
$PY decompose.py $specs >> $L 2>&1
t1=$(ls -d ../artifacts/runs/rl/*_T1/ | xargs -n1 basename | grep -v -E '^(calib|smoke)' | sed 's/$/:400/')
$PY suboptimality.py $t1 >> $L 2>&1
choice=$($PY -c "import json;print(json.load(open('../artifacts/runs/eval/val/selection.json'))['choice'])")
ms="sft rl:lp_b0.1_T5:400 rl:lp_b0.1_T1:400 rl:lp_b0.02_T1:400"
case $choice in rl:*) ms="$ms $choice";; esac
$PY eval_multistep.py $ms >> $L 2>&1
$PY1 booster_judge.py sft $(echo $specs | tr ' ' '\n' | grep -E '^rl:(lp|fair|minutes|wtg|disc|placebo)_b' | grep ':400$' | tr '\n' ' ') pistar:lp:0.02 pistar:lp:0.1 >> $L 2>&1
echo done > ../logs/queue_analysis.done
