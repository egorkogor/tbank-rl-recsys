#!/usr/bin/env bash
# Ordinal reward closest to Yandex's public recipe (ARGUS: skip < completion above a threshold < like):
# mix = P(Listen+) + 0.5 P(like) - 0.5 P(skip). Runs after queue_rl2 (which also runs queue_rl3).
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
while [ ! -f ../logs/queue_rl2.done ]; do sleep 30; done
run() {
  local tag=$1; shift
  [ -f ../artifacts/runs/rl/$tag/step400.pt ] && return 0
  for try in 1 2 3; do
    $PY train_rl.py --tag $tag --lr 3e-4 --G 16 --steps 400 --eval-every 100 "$@" > ../logs/rl_$tag.log 2>&1 && return 0
    sleep 60
  done
}
run mix_b0.5_T1 --reward mix --beta 0.5
run mix_b0.1_T1 --reward mix --beta 0.1
run mix_b0.02_T1 --reward mix --beta 0.02
echo done > ../logs/queue_rl4.done
