#!/usr/bin/env bash
# Post-OOT diagnostics (docs/post_oot_diagnostics_RU.md), after the val grid of the length-penalised pi*.
# The new winner (if not already among the OOT specs) gets a second, labelled OOT run; nothing is overwritten.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
L=../logs/post_oot.log
export PYTHONIOENCODING=utf-8
oot="sft ckpt:sft_nocopy_s6000.pt mostpop personal copyscale:4 pistar:lp:0.1 pistar:balanced:5 pistar:wtg:0.5"
oot="$oot rl:lp_b0.5_T1:400 rl:lp_b0.1_T1:400 rl:lp_b0.02_T1:400 rl:lp_b0_T1:400 rl:minutes_b0.5_T1:400"
oot="$oot rl:balanced_b5_T1:400 rl:balanced_b2_T1:400 rl:mix_b0.1_T1:400 rl:wtg_b0.5_T1:400 rl:placebo_b0.1_T1:400 pistar:minutes:2"
grid="pistarlen:minutes:2:0.5 pistarlen:minutes:2:1 pistarlen:minutes:2:2 pistarlen:minutes:2:4"
$PY select_point.py --stage val --len-guard 0.10 >> $L 2>&1 || exit 1
choice=$($PY -c "import json;print(json.load(open('../artifacts/runs/eval/val/selection_lenguard.json'))['choice'])")
echo "len-guard choice: $choice" >> $L
val="$oot $grid"
case " $val " in *" $choice "*) ;; *) val="$val $choice";; esac
$PY diagnostics.py --stage val $val >> $L 2>&1 || exit 1
case " $oot " in
  *" $choice "*) ;;
  *)
    $PY -c "
import json, subprocess, time
c = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
f = json.load(open('../artifacts/freeze.json'))
rec = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'commit': c, 'spec': '$choice',
       'rule': 'post-OOT rule of docs/post_oot_diagnostics_RU.md section 3 on val (frozen five constraints + length profile <= 1.1 x SFT)',
       'note': 'second OOT run, after the first one was unsealed; checkpoints are those of freeze.json',
       'checkpoints': f['checkpoints']}
json.dump(rec, open('../artifacts/freeze_post_oot.json', 'x'), indent=1)
" >> $L 2>&1 || exit 1
    $PY eval_policy.py --stage oot $choice >> $L 2>&1 || exit 1
    oot="$oot $choice";;
esac
$PY diagnostics.py --stage oot $oot >> $L 2>&1 || exit 1
echo done > ../logs/queue_post_oot.done
