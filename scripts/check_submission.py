"""Read-only checks of the tracked submission; no data, checkpoints or inference required."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess


def check(root):
    tracked = subprocess.check_output(
        ['git', '-c', 'safe.directory=' + root.as_posix(), '-c', 'core.excludesFile=', 'ls-files'],
        cwd=root, text=True, encoding='utf-8').splitlines()
    files = set(tracked)
    problems = []
    runtime_references = []
    total_json = 0
    forbidden_keys = {'uds', 'per_userday', 'uid', 'user_id', 'item_id', 'topk', 'ctx_end', 'tgt'}

    def aggregate(value, location):
        if isinstance(value, dict):
            if forbidden_keys.intersection(value):
                problems.append(location + ': row-level keys')
            for key, child in value.items():
                aggregate(child, location + '/' + key)
        elif isinstance(value, list):
            if len(value) > 30 and not all(isinstance(v, dict) and 'spec' in v for v in value):
                problems.append(location + ': unexplained long array')
            for child in value:
                aggregate(child, location + '[]')

    for name in tracked:
        path = root / name
        if path.stat().st_size > 5_000_000:
            problems.append(name + ': exceeds 5 MB')
        if path.suffix in {'.pt', '.npz', '.npy', '.parquet'} or name.startswith(('data/', '.venv/')):
            problems.append(name + ': runtime data tracked')
        if name.startswith('artifacts/') and path.suffix == '.json':
            total_json += path.stat().st_size
            aggregate(json.loads(path.read_text('utf-8')), name)
    if total_json >= 5_000_000:
        problems.append('Aggregate JSON package exceeds 5 MB')

    readme = (root / 'README.md').read_text('utf-8')
    for num, line in enumerate(readme.splitlines(), 1):
        for token in re.findall(r'`([^`\n]+)`', line):
            if ' ' in token or not re.search(r'\.(py|sh|md|json|png|parquet|pt|npz)$', token):
                continue
            if token in files:
                continue
            if token in {'.npz', 'events.parquet', 'items.parquet'} or (
                token == 'scripts/download_yambda.py' and 'том репозитории' in line
            ):
                runtime_references.append({'line': num, 'path': token})
            else:
                problems.append(f'README:{num}: untracked path {token}')
    headings = [re.sub(r'^#+\s+', '', x) for x in readme.splitlines() if x.startswith('#')]
    anchors = {re.sub(r'[^\w\- ]', '', h.lower()).replace(' ', '-') for h in headings}
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', readme):
        if re.match(r'https?://', target):
            continue
        path, _, fragment = target.partition('#')
        if path and path not in files:
            problems.append('README: untracked Markdown link ' + target)
        if not path and fragment and fragment not in anchors:
            problems.append('README: missing heading ' + target)
    for prefix in ('8.1.', '8.2.', '8.3.', '9.', '10.', '11.', '12.', '13.'):
        if not any(h.startswith(prefix) for h in headings):
            problems.append('README: missing section ' + prefix)
    widths = []
    for num, line in enumerate(readme.splitlines(), 1):
        if line.startswith('|'):
            width = len(re.split(r'(?<!\\)\|', line))
            if widths and width != widths[-1]:
                problems.append(f'README:{num}: inconsistent table width')
            widths.append(width)
        else:
            widths = []

    def load(name):
        return json.loads((root / name).read_text('utf-8'))

    selection = load('artifacts/runs/eval/val/selection_lenguard.json')
    assert selection['choice'] == 'pistarlen:minutes:2:2'
    assert (selection['n_pass'], selection['n_candidates']) == (21, 73)
    base_val = load('artifacts/runs/eval/val/sft.json')
    threshold = 1.1 * base_val['behaviour']['len_dev_user_log']
    assert round(threshold, 3) == 0.122
    for stage in ('val', 'oot'):
        diag = load(f'artifacts/runs/eval/{stage}/diag/diag.json')
        for row in diag.values():
            r3 = row['R3_disliked_share']
            ci = r3['delta']
            expected = ('лучше' if ci['hi'] < 0 else 'хуже' if ci['lo'] > 0 else
                        'сопоставимо' if abs(ci['mean']) < .005 else 'не определено')
            assert r3['verdict'] == expected
    diag = load('artifacts/runs/eval/oot/diag/diag.json')
    original = diag['pistar:minutes:2']
    corrected = diag['pistarlen:minutes:2:2']
    assert round(original['B']['delta']['mean'], 2) == 3.99
    assert round(original['B_neutral']['delta']['mean'], 2) == -.23
    assert original['B_neutral']['delta']['hi'] < 0
    assert round(corrected['B']['delta']['mean'], 2) == 1.74
    assert round(100 * corrected['B']['delta']['mean'] / corrected['B']['base'], 1) == 10.7
    assert corrected['B_neutral']['delta']['lo'] < 0 < corrected['B_neutral']['delta']['hi']
    base = load('artifacts/runs/eval/oot/sft.json')
    winner = load('artifacts/runs/eval/oot/pistarlen__minutes__2__2.json')
    comparison = load('artifacts/runs/eval/oot/comparison_post_oot.json')
    assert comparison['v_ndcg@10'] != 'хуже'
    assert winner['relevance']['ndcg@10']['mean'] >= .98 * base['relevance']['ndcg@10']['mean']
    assert winner['relevance']['recall@100_new']['mean'] >= .9 * base['relevance']['recall@100_new']['mean']
    assert winner['r_judge_snap']['skip'] - base['r_judge_snap']['skip'] <= .005
    assert winner['r_judge_snap']['lp'] - base['r_judge_snap']['lp'] >= -.005
    assert winner['behaviour']['top_item_share'] <= 2 * base['behaviour']['top_item_share']
    assert winner['behaviour']['len_dev_user_log'] <= 1.1 * base['behaviour']['len_dev_user_log']
    freezes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'artifacts').glob('freeze*.json')}
    result = {'tracked_files': len(tracked), 'aggregate_json_bytes': total_json,
              'length_guard_val': threshold, 'freeze_sha256': freezes,
              'runtime_references': runtime_references, 'problems': problems}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    check(parser.parse_args().root.resolve())
