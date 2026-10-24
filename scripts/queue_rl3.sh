#!/usr/bin/env bash
# Rewards from the literature search (docs/reward_formulas_review_RU.md), called at the end of queue_rl2.sh.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
run() {
  local tag=$1; shift
  [ -f ../artifacts/runs/rl/$tag/step400.pt ] || [ -f ../artifacts/runs/rl/$tag/step200.pt -a "$tag" = spotify_b0.1_T5 ] && return 0
  for try in 1 2 3; do
    $PY train_rl.py --tag $tag --lr 3e-4 --G 16 --steps 400 --eval-every 100 "$@" > ../logs/rl_$tag.log 2>&1 && return 0
    sleep 60
  done
}
run wtg_b0.5_T1 --reward wtg --beta 0.5
run wtg_b0.1_T1 --reward wtg --beta 0.1
run add_b0.5_T1 --reward add --beta 0.5
run add_b0.1_T1 --reward add --beta 0.1
run spotify_b0.1_T5 --reward spotify --beta 0.1 --T 5 --bs 32 --steps 200
# owner's balanced minutes B = (sum_k sqrt(s*_k M_k))^2 over old / new track of a known artist / new artist
# rewards in minutes have a ~10x larger spread across tracks than probabilities: minutes at beta 0.5 collapsed
# into 14-minute tracks (KL 8.8), so the minute-scale rewards get beta 2 / 5 / 20
run balanced_b20_T1 --reward balanced --beta 20
run balanced_b5_T1 --reward balanced --beta 5
run balanced_b2_T1 --reward balanced --beta 2
run minutes_b20_T1 --reward minutes --beta 20
run minutes_b5_T1 --reward minutes --beta 5
# owner's idea: minutes + completion bonus (standardised, equal weights)
run mlp_b0.5_T1 --reward mlp --beta 0.5
run mlp_b0.1_T1 --reward mlp --beta 0.1
