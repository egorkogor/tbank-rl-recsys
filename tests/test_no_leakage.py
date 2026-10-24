"""Leakage and train/serve parity checks on the built artifacts/seq/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from rlrec.config import CONT_GAP, LIKE_HORIZON, N_SPECIAL, OOT_DAYS, TRAIN_END, VAL_DAYS, day_start  # noqa: E402
from rlrec.data import Seq, rm_train_mask  # noqa: E402
from rlrec.evaluate import contexts  # noqa: E402
from rlrec.model import Policy, make_batch  # noqa: E402
from rlrec.rm import COUNT_KEYS, HistoryCounts  # noqa: E402


@pytest.fixture(scope="module")
def seq():
    return Seq()


def test_context_is_strictly_before_the_day(seq):
    uds = seq.userdays(VAL_DAYS + OOT_DAYS)[::50]
    b = make_batch(seq, contexts(seq, uds), 256, "cpu")
    day = np.asarray(seq.day)
    gi = b["gi"].numpy()
    d = seq.ud["day"][uds][:, None]
    used = gi >= 0
    assert (day[np.where(used, gi, 0)][used] < np.broadcast_to(d, gi.shape)[used]).all()
    # the context ends right before the day: the last real event is the day's previous listen
    assert (gi[:, -1] == seq.ud["ctx_end"][uds] - 1).all()


def test_output_ignores_the_future(seq):
    """Shuffling every listen from the day on does not change the policy output."""
    torch.manual_seed(0)
    m = Policy(seq.n_tok, d=32, n_layer=1, n_head=2).eval()
    uds = seq.userdays(VAL_DAYS)[:16]
    ctx = contexts(seq, uds)

    class Shuffled:
        def __init__(self, base):
            self.__dict__.update(base.__dict__)
            rng = np.random.default_rng(1)
            for f in ("tok", "outcome", "org", "gap"):
                a = np.array(getattr(base, f))
                for (s, e), i in zip(ctx, uds):
                    u = seq.ud["u"][i]
                    tail = slice(e, int(seq.ptr[u + 1]))
                    a[tail] = rng.permutation(a[tail])
                setattr(self, f, a)

    with torch.no_grad():
        z0 = m.last_logits(make_batch(seq, ctx, 64, "cpu"))
        z1 = m.last_logits(make_batch(Shuffled(seq), ctx, 64, "cpu"))
    assert torch.equal(z0, z1)


def test_vocabulary_from_train_days_only(seq):
    tok, day = np.asarray(seq.tok), np.asarray(seq.day)
    cnt = np.bincount(tok[day < TRAIN_END], minlength=seq.n_tok)[N_SPECIAL:]
    assert (cnt > 0).all()
    assert (np.diff(cnt) <= 0).all()  # tokens are ranks of the train count


def test_rm_labels_close_before_cutoff(seq):
    m = rm_train_mask(seq)
    t = np.asarray(seq.t)[m]
    assert t.max() + max(CONT_GAP, LIKE_HORIZON) < day_start(TRAIN_END)
    assert (np.asarray(seq.org)[m] == 0).all()


def test_history_counts_match_logged_side_features(seq):
    """Side features at serving (HistoryCounts) equal the ones the RM was trained on (rep_item / rep_artist)."""
    rng = np.random.default_rng(0)
    uidx = seq.user_index()
    tok = np.asarray(seq.tok)
    g = rng.choice(np.flatnonzero(tok[: 5_000_000] >= N_SPECIAL), 300, replace=False)
    hc = HistoryCounts(seq)
    c = hc([(int(seq.ptr[uidx[x]]), int(x)) for x in g], tok[g][:, None])
    for k in COUNT_KEYS:
        assert (c[k][:, 0] == np.asarray(getattr(seq, k))[g]).all(), k


def test_targets_are_that_days_listen_plus(seq):
    i = seq.userdays(VAL_DAYS)[123]
    u, d = seq.ud["u"][i], seq.ud["day"][i]
    s, e = seq.ptr[u], seq.ptr[u + 1]
    day, pr, item = np.asarray(seq.day)[s:e], np.asarray(seq.pr)[s:e], np.asarray(seq.item)[s:e]
    assert len(np.unique(item[(day == d) & (pr >= 50)])) == seq.ud["n_tgt"][i]
