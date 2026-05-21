#!/usr/bin/env bash
# Measure subgraph size + PoS(L) for a trained checkpoint, in both modes.
#
# Usage:
#   bash scripts/measure_subgraph.sh <dataset> [gpu] [max_users_for_none] [batch_for_none]
# Example:
#   bash scripts/measure_subgraph.sh last-fm 0 200 4
#
# Expects a checkpoint at results/checkpoints/<dataset>_best.pt (saved by train.py).
set -e

DATASET=${1:-last-fm}
GPU=${2:-0}
MAX_USERS_NONE=${3:-200}
BATCH_NONE=${4:-4}

CKPT="results/checkpoints/${DATASET}_best.pt"
DATA="data/${DATASET}/"

if [[ ! -f "$CKPT" ]]; then
    echo "Checkpoint $CKPT not found."
    echo "Train first: python train.py --data_path $DATA --gpu $GPU"
    exit 1
fi
if [[ ! -d "$DATA" ]]; then
    echo "Dataset directory $DATA not found."
    exit 1
fi

echo "============================================================"
echo "[1/2] mode=model  (sampling as trained)"
echo "============================================================"
python measure_subgraph.py --data_path "$DATA" --ckpt "$CKPT" --gpu "$GPU" --mode model

echo
echo "============================================================"
echo "[2/2] mode=none   (PPR top-K disabled; --max_users $MAX_USERS_NONE)"
echo "============================================================"
python measure_subgraph.py --data_path "$DATA" --ckpt "$CKPT" --gpu "$GPU" \
    --mode none --max_users "$MAX_USERS_NONE" --batch_size "$BATCH_NONE"

echo
echo "============================================================"
echo "Summary (results/${DATASET}_subgraph_*.json):"
echo "============================================================"
for m in model none; do
    f="results/${DATASET}_subgraph_${m}.json"
    if [[ -f "$f" ]]; then
        echo "--- $m ---"
        python -c "import json; d=json.load(open('$f')); \
print('  avg_nodes_per_layer:', [round(x,2) for x in d['avg_nodes_per_layer']]); \
print(f\"  PI(L)={d['PI']:.4f}  SI(L)={d['SI']:.4f}  PoS(L)={d['PoS']:.6e}\")"
    fi
done
