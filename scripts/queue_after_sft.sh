#!/usr/bin/env bash
# Wait for the SFT with the repeat channel, make it the reference policy (sft.pt), then: RM members 0-3,
# RM eval on val, full eval of SFT, the ablation without the repeat channel and the simple baselines.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
while [ ! -f ../artifacts/ckpt/sft_copy_log.json ]; do sleep 30; done
cp ../artifacts/ckpt/sft_copy.pt ../artifacts/ckpt/sft.pt
for m in 0 1 2 3; do
  for try in 1 2 3; do
    $PY train_rm.py --member $m > ../logs/rm_m$m.log 2>&1 && break
    sleep 60
  done
done
$PY eval_rm.py --stage val > ../logs/rm_eval_val.log 2>&1
$PY eval_policy.py sft ckpt:sft_nocopy_s6000.pt mostpop personal > ../logs/eval_base_val.log 2>&1
echo done > ../logs/queue_after_sft.done
