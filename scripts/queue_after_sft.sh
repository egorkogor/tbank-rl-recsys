#!/usr/bin/env bash
# Wait for SFT, then: RM members 0-3, RM eval on val, full eval of SFT and the simple baselines.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
while [ ! -f ../artifacts/ckpt/sft_log.json ]; do sleep 30; done
for m in 0 1 2 3; do
  $PY train_rm.py --member $m > ../logs/rm_m$m.log 2>&1
done
$PY eval_rm.py --stage val > ../logs/rm_eval_val.log 2>&1
$PY eval_policy.py sft mostpop personal > ../logs/eval_base_val.log 2>&1
echo done > ../logs/queue_after_sft.done
