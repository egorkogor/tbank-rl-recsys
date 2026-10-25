"""Read-only admission checks before an OOT policy evaluation."""
from pathlib import Path
import hashlib
import json


def checkpoint_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()[:16]


def verify_checkpoint_map(art, checkpoint_map):
    art = Path(art).resolve()
    for relative, expected in checkpoint_map.items():
        path = (art / relative.replace('\\', '/')).resolve()
        if not path.is_relative_to(art):
            raise ValueError('Checkpoint path leaves the artifact directory')
        if checkpoint_hash(path) != expected:
            raise ValueError('Frozen checkpoint hash differs: ' + relative)


def admit_oot(art, outdir, specs):
    """Return pending specs; saved results are never regenerated or overwritten."""
    art, outdir = Path(art), Path(outdir)
    first = json.loads((art / 'freeze.json').read_text('utf-8'))
    allowed = set(first['specs'])
    second_path = art / 'freeze_post_oot.json'
    if second_path.exists():
        second = json.loads(second_path.read_text('utf-8'))
        if second['checkpoints'] != first['checkpoints'] or 'second OOT run' not in second['note']:
            raise ValueError('Second OOT manifest must retain its label and original checkpoints')
        allowed.add(second['spec'])
    unknown = set(specs) - allowed
    if unknown:
        raise ValueError('Specs absent from the frozen protocol: ' + ', '.join(sorted(unknown)))
    pending = []
    for spec in specs:
        name = spec.replace(':', '__')
        if (outdir / (name + '.json')).exists():
            continue
        if (outdir / (name + '.npz')).exists():
            raise ValueError('Saved OOT array has no summary; refusing to overwrite: ' + spec)
        pending.append(spec)
    if pending:
        verify_checkpoint_map(art, first['checkpoints'])
    return pending
