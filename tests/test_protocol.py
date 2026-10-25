"""Protocol checks use tiny temporary files and do not run recommendation models."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rlrec.protocol import admit_oot


@pytest.fixture
def frozen(tmp_path):
    art = tmp_path / 'artifacts'
    art.mkdir()
    ckpt = art / 'checkpoint.pt'
    ckpt.write_bytes(b'frozen toy checkpoint')
    record = {'specs': ['sft'], 'checkpoints': {'checkpoint.pt': hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16]}}
    (art / 'freeze.json').write_text(json.dumps(record), encoding='utf-8')
    outdir = art / 'eval'
    outdir.mkdir()
    return art, outdir, record


def test_unknown_spec_is_rejected(frozen):
    art, outdir, _ = frozen
    with pytest.raises(ValueError, match='absent'):
        admit_oot(art, outdir, ['new:choice'])


def test_checkpoint_drift_is_rejected(frozen):
    art, outdir, _ = frozen
    (art / 'checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        admit_oot(art, outdir, ['sft'])


def test_existing_summary_needs_no_checkpoint_and_is_untouched(frozen):
    art, outdir, _ = frozen
    summary = outdir / 'sft.json'
    summary.write_bytes(b'{"saved":true}')
    (art / 'checkpoint.pt').unlink()
    assert admit_oot(art, outdir, ['sft']) == []
    assert summary.read_bytes() == b'{"saved":true}'


def test_orphan_array_is_not_overwritten(frozen):
    art, outdir, _ = frozen
    (outdir / 'sft.npz').write_bytes(b'saved array')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        admit_oot(art, outdir, ['sft'])


def test_second_run_must_keep_label_and_checkpoint_map(frozen):
    art, outdir, record = frozen
    second = {'spec': 'pistarlen:minutes:2:2', 'checkpoints': record['checkpoints'], 'note': 'second OOT run'}
    path = art / 'freeze_post_oot.json'
    path.write_text(json.dumps(second), encoding='utf-8')
    assert admit_oot(art, outdir, [second['spec']]) == [second['spec']]
    second['note'] = 'replacement'
    path.write_text(json.dumps(second), encoding='utf-8')
    with pytest.raises(ValueError, match='retain its label'):
        admit_oot(art, outdir, [second['spec']])
