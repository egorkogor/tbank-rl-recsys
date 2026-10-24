#!/usr/bin/env bash
# Owner's idea (24.10 18:00): reward = listened minutes, per track (T = 1) and per session (T = 5).
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
while [ ! -f ../logs/queue_rl.done ]; do sleep 30; done
run() {
  local tag=$1; shift
  [ -f ../artifacts/runs/rl/$tag/step400.pt ] && return 0
  for try in 1 2 3; do
    $PY train_rl.py --tag $tag --lr 3e-4 --G 16 --steps 400 --eval-every 100 "$@" > ../logs/rl_$tag.log 2>&1 && return 0
    sleep 60
  done
}
run minutes_b0.5_T1 --reward minutes --beta 0.5
run minutes_b0.1_T1 --reward minutes --beta 0.1
run minutes_b0.02_T1 --reward minutes --beta 0.02
run minutes_b0.1_T5 --reward minutes --beta 0.1 --T 5 --bs 32
# rewards from the literature search, if a queue for them was written meanwhile
[ -f queue_rl3.sh ] && bash queue_rl3.sh
echo done > ../logs/queue_rl2.done
