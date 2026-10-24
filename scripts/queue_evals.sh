#!/usr/bin/env bash
# After all RL queues, one GPU process at a time: snapshot judges m4-m6, then the full val evaluation.
# Specs run in batches (one process per batch shares the per-user-day history cache); done specs are skipped.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
while [ ! -f ../logs/queue_rl.done ] || [ ! -f ../logs/queue_rl2.done ]; do sleep 30; done
[ -f ../logs/queue_snapjudges.done ] || bash queue_snapjudges.sh
specs="sft ckpt:sft_nocopy_s6000.pt mostpop personal copyscale:2 copyscale:4"
for r in lp fair wtg; do for b in 0.02 0.1 0.5; do specs="$specs pistar:$r:$b"; done; done
for r in minutes balanced; do for b in 2 5 20; do specs="$specs pistar:$r:$b"; done; done
for d in ../artifacts/runs/rl/*/; do
  run=$(basename $d)
  case $run in calib*|smoke*) continue;; esac
  last=$(ls $d | grep -o 'step[0-9]*' | sed 's/step//' | sort -n | tail -1)
  [ -n "$last" ] && specs="$specs rl:$run:$last"
done
for run in lp_b0.1_T1 lp_b0.02_T1; do for st in 100 200 300; do specs="$specs rl:$run:$st"; done; done
todo=""
for s in $specs; do [ -f ../artifacts/runs/eval/val/$(echo "$s" | sed 's/:/__/g').json ] || todo="$todo $s"; done
echo "specs to evaluate: $(echo $todo | wc -w)" >> ../logs/eval_val.log
set -- $todo
while [ $# -gt 0 ]; do
  batch=""; for i in 1 2 3 4 5 6 7 8; do [ $# -gt 0 ] && { batch="$batch $1"; shift; }; done
  for try in 1 2; do $PY eval_policy.py $batch >> ../logs/eval_val.log 2>&1 && break; sleep 60; done
done
echo done > ../logs/queue_evals.done
