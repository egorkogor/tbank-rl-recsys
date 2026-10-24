#!/usr/bin/env bash
# Evaluations after the RL grid (one GPU process at a time): baselines, closed-form pi*, every RL run.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
while [ ! -f ../logs/queue_rl.done ]; do sleep 30; done
ev() { [ -f ../artifacts/runs/eval/val/$(echo "$1" | sed 's/:/__/g').json ] || $PY eval_policy.py "$@" >> ../logs/eval_val.log 2>&1; }
for s in ckpt:sft_nocopy_s6000.pt mostpop personal copyscale:2 copyscale:4; do ev $s; done
for r in lp ratio fair cont disc placebo; do for b in 0.02 0.1 0.5; do ev pistar:$r:$b; done; done
for d in ../artifacts/runs/rl/*/; do
  run=$(basename $d)
  case $run in calib*|smoke*) continue;; esac
  for st in 100 200 300 400; do [ -f $d/step$st.pt ] && ev rl:$run:$st; done
done
echo done > ../logs/queue_evals.done
