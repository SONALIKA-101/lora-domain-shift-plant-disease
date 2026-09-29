#!/bin/bash
# Runs the four methods (doga, lora_plain, linear_probe, full_finetune) for the
# held-out domain set in the config (data.held_out_domain).
# Usage (from the repository root): bash scripts/run_baselines.sh configs/base.yaml
#
# full_finetune is the most memory-hungry method (it unfreezes the whole
# backbone), so it runs last: if it fails, for example with an out-of-memory
# error, the earlier runs are unaffected.

CONFIG=${1:-configs/base.yaml}

echo "=== Running doga (LoRA + domain-adversarial head) ==="
python -m src.train --config "$CONFIG" --mode doga
DOGA_STATUS=$?

echo "=== Running lora_plain baseline (no domain-adversarial term) ==="
python -m src.train --config "$CONFIG" --mode lora_plain
LORA_STATUS=$?

echo "=== Running linear_probe baseline ==="
python -m src.train --config "$CONFIG" --mode linear_probe
LINEAR_STATUS=$?

echo "=== Running full_finetune baseline (most memory-hungry; may fail with out-of-memory on a small GPU) ==="
python -m src.train --config "$CONFIG" --mode full_finetune
FULLFT_STATUS=$?

echo ""
echo "=== Summary ==="
[ "$DOGA_STATUS" -eq 0 ] && echo "doga: OK" || echo "doga: FAILED (exit $DOGA_STATUS)"
[ "$LORA_STATUS" -eq 0 ] && echo "lora_plain: OK" || echo "lora_plain: FAILED (exit $LORA_STATUS)"
[ "$LINEAR_STATUS" -eq 0 ] && echo "linear_probe: OK" || echo "linear_probe: FAILED (exit $LINEAR_STATUS)"
[ "$FULLFT_STATUS" -eq 0 ] && echo "full_finetune: OK" || echo "full_finetune: FAILED (exit $FULLFT_STATUS) - often an out-of-memory error"
echo ""
echo "Checkpoints are saved under checkpoints/ and each run is logged in results/summary.csv."
