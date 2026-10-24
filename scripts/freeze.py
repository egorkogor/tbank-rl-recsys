"""Freeze before the single OOT run: specs, selection rule, thresholds, commit and checkpoint hashes.

Writes artifacts/freeze.json once; eval_policy.py / eval_rm.py refuse --stage oot without it.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlrec.config import ART, CKPT, RUNS  # noqa: E402
from compare import REL_THRESHOLD, REWARD_THRESHOLD  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def ckpt_of(spec: str) -> list[Path]:
    if spec.startswith("rl:"):
        _, run, step = spec.split(":")
        return [RUNS / "rl" / run / f"step{step}.pt"]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--rule", required=True, help="how the RL checkpoints were chosen on val, in words")
    a = ap.parse_args()
    out = ART / "freeze.json"
    if out.exists():
        sys.exit(f"{out} exists: the OOT protocol is already frozen")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], capture_output=True, text=True,
                           cwd=Path(__file__).parent).stdout.strip()
    if dirty:
        sys.exit("commit src/ and scripts/ before freezing:\n" + dirty)
    files = [CKPT / "sft.pt"] + [CKPT / f"rm_m{m}.pt" for m in range(4)]
    for s in a.specs:
        files += ckpt_of(s)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit, "specs": a.specs, "rule": a.rule,
           "thresholds": {"relevance_relative": REL_THRESHOLD, "reward_absolute": REWARD_THRESHOLD},
           "checkpoints": {str(f.relative_to(ART)): sha(f) for f in files}}
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps(rec, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
