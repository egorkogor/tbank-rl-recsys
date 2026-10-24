#!/usr/bin/env bash
# After the RL grid, one GPU process at a time: snapshot judges, then evaluations (baselines, pi*, every RL run).
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
while [ ! -f ../logs/queue_rl.done ] || [ ! -f ../logs/queue_rl2.done ]; do sleep 30; done
# snapshot-trained judges m4-m6, sequentially (RAM fits one GPU process next to task 1)
[ -f ../logs/queue_snapjudges.done ] || bash queue_snapjudges.sh
[ -f ../artifacts/runs/rm_eval_val_snapshot.json ] || $PY eval_rm.py --stage val --snapshot > ../logs/rm_eval_val_snapshot.log 2>&1
ev() { [ -f ../artifacts/runs/eval/val/$(echo "$1" | sed 's/:/__/g').json ] || $PY eval_policy.py "$@" >> ../logs/eval_val.log 2>&1; }
for s in sft ckpt:sft_nocopy_s6000.pt mostpop personal copyscale:2 copyscale:4; do ev $s; done
for r in lp ratio fair cont disc placebo; do for b in 0.02 0.1 0.5; do ev pistar:$r:$b; done; done
for d in ../artifacts/runs/rl/*/; do
  run=$(basename $d)
  case $run in calib*|smoke*) continue;; esac
  for st in 100 200 300 400; do [ -f $d/step$st.pt ] && ev rl:$run:$st; done
done
echo done > ../logs/queue_evals.done
