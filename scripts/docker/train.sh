#!/bin/bash

# Compute norm stats
if [ "$SKIP_STATS" != "true" ]; then
    uv run /app/scripts/compute_norm_stats.py --config-name $TRAIN_CONFIG --exp-name $EXP_NAME || exit 1
fi

# Train
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run /app/scripts/train.py $TRAIN_CONFIG --exp-name $EXP_NAME $TRAIN_ARGS || exit 1