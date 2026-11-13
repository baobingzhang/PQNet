#!/usr/bin/env bash
# PQNet and TQNet on one dataset: all four horizons x three seeds. Results go to results/<dataset>/.
# Usage: bash scripts/run_dataset.sh ETTh1 [extra train.py args, e.g. --device cuda]
set -euo pipefail
DS=${1:?usage: run_dataset.sh <dataset> [args]}; shift || true
HORIZONS=$(python -c "from pqnet.data import config; print(' '.join(map(str, config('$DS').horizons)))")
for H in $HORIZONS; do
  for SEED in 2024 2025 2026; do
    for M in pqnet tqnet; do
      python train.py --dataset "$DS" --pred_len "$H" --seed "$SEED" --model "$M" \
        --out "results/$DS/${M}_H${H}_s${SEED}.json" "$@"
    done
  done
done
python - "$DS" <<'PY'
import glob, json, sys, collections
ds = sys.argv[1]; acc = collections.defaultdict(list)
for f in glob.glob(f"results/{ds}/*.json"):
    r = json.load(open(f)); acc[(r["model"], r["pred_len"])].append(r["mse"])
for (m, h), v in sorted(acc.items()):
    print(f"{ds} {m:5s} H={h:<4d} MSE {sum(v)/len(v):.4f} over {len(v)} seeds")
PY
