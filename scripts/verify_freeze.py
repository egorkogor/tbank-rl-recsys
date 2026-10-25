"""Verify the recorded protocol without running models or writing any artifacts."""
from pathlib import Path
import argparse
import hashlib
import json

EXPECTED = {
    'freeze.json': 'b0f011f0a865c17f05f76db3801301bdf8f64068e4c54007b83d87e5c026b83c',
    'freeze_post_oot.json': 'b8f5ba588dbefaae952f2e3688801035d9861074c2defebf2bbcc73dc7db022c',
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root, checkpoints=False):
    art = root / 'artifacts'
    records = {}
    for name, expected in EXPECTED.items():
        raw = (art / name).read_bytes()
        canonical_crlf = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        if sha(art / name) != expected and hashlib.sha256(canonical_crlf).hexdigest() != expected:
            raise ValueError('Recorded freeze file changed: ' + name)
        records[name] = json.loads((art / name).read_text('utf-8'))
    first, second = records.values()
    if first['checkpoints'] != second['checkpoints']:
        raise ValueError('Second run must use the original checkpoints')
    if second['spec'] != 'pistarlen:minutes:2:2' or 'second OOT run' not in second['note']:
        raise ValueError('Second OOT run must remain explicitly labelled')
    checked = 0
    if checkpoints:
        for relative, expected in first['checkpoints'].items():
            path = art / relative.replace('\\', '/')
            if sha(path)[:16] != expected:
                raise ValueError('Frozen checkpoint changed: ' + relative)
            checked += 1
    return {'freeze_files_unchanged': True, 'second_run_label_preserved': True,
            'checkpoints_checked': checked, 'checkpoint_verification_requested': checkpoints}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--checkpoints', action='store_true', help='Also hash the local checkpoint files')
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve(), args.checkpoints), indent=2))
