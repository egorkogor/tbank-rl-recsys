#!/usr/bin/env bash
# Snapshot-trained judges m4-m6 (parallel to the RL grid while RAM allows), then their snapshot evaluation.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
for m in 4 5 6; do
  [ -f ../artifacts/ckpt/rm_m$m.pt ] && continue
  for try in 1 2 3; do
    $PY train_rm_snapshot.py --member $m > ../logs/rm_m$m.log 2>&1 && break
    sleep 60
  done
done
$PY eval_rm.py --stage val --snapshot --members 0,4,5,6 > ../logs/rm_eval_val_snapshot_judges.log 2>&1
echo done > ../logs/queue_snapjudges.done
