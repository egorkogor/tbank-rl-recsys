#!/usr/bin/env bash
# RL grid after the RM queue, one GPU process at a time (RAM on this laptop fits one).
# T = 1: rewards x beta; beta = 0 once (collapse demo); placebo; then T = 5 for two rewards.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
while [ ! -f ../logs/queue_after_sft.done ]; do sleep 30; done
[ -f ../artifacts/runs/rm_eval_val_snapshot.json ] || $PY eval_rm.py --stage val --snapshot > ../logs/rm_eval_val_snapshot.log 2>&1
run() {  # tag, args...
  local tag=$1; shift
  [ -f ../artifacts/runs/rl/$tag/step400.pt ] && return 0
  for try in 1 2 3; do
    $PY train_rl.py --tag $tag --lr 1e-4 --steps 400 --eval-every 100 "$@" > ../logs/rl_$tag.log 2>&1 && return 0
    sleep 60
  done
}
for beta in 0.1 0.02 0.5; do
  for r in lp ratio fair cont disc placebo; do
    run ${r}_b${beta}_T1 --reward $r --beta $beta
  done
done
run lp_b0_T1 --reward lp --beta 0
run lp_b0.1_T5 --reward lp --beta 0.1 --T 5 --bs 32
run fair_b0.1_T5 --reward fair --beta 0.1 --T 5 --bs 32
echo done > ../logs/queue_rl.done
