#!/usr/bin/env bash
# SFT (arguments passed through) with automatic resume from the last checkpoint after a crash (up to 5 restarts).
cd "$(dirname "$0")"
for n in 1 2 3 4 5 6; do
  ../.venv/Scripts/python.exe train_sft.py "$@" --resume >> ../logs/sft.log 2>&1 && exit 0
  echo "restart $n after exit code $?" >> ../logs/sft.log
  sleep 60
done
exit 1
