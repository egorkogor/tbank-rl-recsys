"""Paths, time axis and windows (day indices), shared with task 1.

Days are counted back from the end of the Yambda-50M log: day 300 is exactly the last 24 h.
All models (SFT, RM, RL) train on events of days < TRAIN_END. val = 287-293 (every choice),
OOT = 294-300 (one run of frozen checkpoints, no retraining).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# events.parquet / items.parquet built by task 1 (prepared by tbrec.data.prepare in task 1).
DATA = Path(os.environ.get("TBANK_DATA", ROOT.parent / "tbank-recsys-autofeatures" / "data" / "processed"))
ART = ROOT / "artifacts"
SEQ = ART / "seq"
CKPT = Path(os.environ.get("RLREC_CKPT", ART / "ckpt"))
RUNS = Path(os.environ.get("RLREC_RUNS", ART / "runs"))
DEVICE = os.environ.get("RLREC_DEVICE", "cuda")

DAY = 86_400
T_END = 26_000_005
N_DAYS = 301
OFFSET = N_DAYS * DAY - T_END  # day d covers [d*DAY - OFFSET, (d+1)*DAY - OFFSET)
SEED = 20261024

TRAIN_END = 287
RL_DAYS = list(range(273, 287))
VAL_DAYS = list(range(287, 294))
OOT_DAYS = list(range(294, 301))

VOCAB = 100_000
PAD, UNK, N_SPECIAL = 0, 1, 2

# Outcome of a listen (input side feature). UNKNOWN marks generated items and outcome dropout.
SKIP, PARTIAL, LISTEN_PLUS, UNKNOWN = 0, 1, 2, 3
# Gap since the previous listen of the user, seconds -> bucket 0..7 (0 = first event).
GAP_EDGES = [60, 300, 1800, 10_800, 86_400, 7 * 86_400]
GEN_GAP = 2  # generated items are "next track in the session": gap 1-5 min
CONT_GAP = 1800  # session continues if the next listen starts within 30 min
LIKE_HORIZON = DAY
ADOPT_HORIZON = 7 * DAY  # discovery: organic listen of a newly recommended artist within 7 days


def day_start(d: int) -> int:
    return d * DAY - OFFSET
