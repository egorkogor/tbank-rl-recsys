#!/usr/bin/env bash
# After OOT: five steps ahead with re-recommendation of generated tracks forbidden (val).
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
while [ ! -f ../logs/queue_oot.done ]; do sleep 60; done
$PY eval_multistep.py sft rl:lp_b0.1_T1:400 rl:lp_b0.1_T5:400 rl:balanced_b5_T1:400 --modes open_ban >> ../logs/extra.log 2>&1
echo done > ../logs/queue_extra.done
