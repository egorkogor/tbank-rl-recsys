#!/usr/bin/env bash
# The single OOT run (days 294-300) of frozen checkpoints, after the val analysis.
# freeze.json records the commit, thresholds, checkpoint hashes and the specs; OOT results are never overwritten.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
PY1=D:/PdM_Projects/tbank-recsys-autofeatures/.venv/Scripts/python.exe
L=../logs/oot.log
while [ ! -f ../logs/queue_analysis.done ]; do sleep 60; done
choice=$($PY -c "import json;print(json.load(open('../artifacts/runs/eval/val/selection.json'))['choice'])")
specs="sft ckpt:sft_nocopy_s6000.pt mostpop personal copyscale:4 pistar:lp:0.1 pistar:balanced:5 pistar:wtg:0.5"
specs="$specs rl:lp_b0.5_T1:400 rl:lp_b0.1_T1:400 rl:lp_b0.02_T1:400 rl:lp_b0_T1:400 rl:minutes_b0.5_T1:400"
specs="$specs rl:balanced_b5_T1:400 rl:balanced_b2_T1:400 rl:mix_b0.1_T1:400 rl:wtg_b0.5_T1:400 rl:placebo_b0.1_T1:400"
case " $specs " in *" $choice "*) ;; *) specs="$specs $choice";; esac
[ -f ../artifacts/freeze.json ] || $PY freeze.py $specs --rule "selection rule of docs/metrics_hypotheses_RU.md section 3 on val: choice = $choice; the other specs are references fixed before OOT" >> $L 2>&1 || exit 1
$PY eval_rm.py --stage oot --snapshot >> $L 2>&1
$PY eval_policy.py --stage oot $specs >> $L 2>&1
for k in balanced lp minutes skip; do $PY compare.py --stage oot --kind $k >> $L 2>&1; done
PYTHONDONTWRITEBYTECODE=1 TBREC_CORES=0-2 $PY1 booster_judge.py --stage oot sft personal copyscale:4 pistar:lp:0.1 pistar:balanced:5 rl:lp_b0.1_T1:400 rl:lp_b0_T1:400 rl:balanced_b5_T1:400 $choice >> $L 2>&1
echo done > ../logs/queue_oot.done
