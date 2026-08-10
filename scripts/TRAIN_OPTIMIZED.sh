#!/bin/bash
# Optimized training command for RL Vision model

# Este script vive en scripts/, pero el entrenamiento debe lanzarse desde la
# raíz del repositorio para que las rutas por defecto de models/ resuelvan bien.
cd "$(dirname "$0")/.." || exit 1

echo "Starting optimized training for RL Vision model..."
echo ""

# Default values
TOTAL_TIMESTEPS=${1:-2000000}
NUM_ENVS=${2:-4}
MAX_LEVEL=${3:-4}

echo "Configuration:"
echo "  • Total timesteps: $TOTAL_TIMESTEPS"
echo "  • Parallel envs: $NUM_ENVS"
echo "  • Max difficulty level: $MAX_LEVEL"
echo "  • Preset: optimized (fast convergence)"
echo "  • LR Schedule: cosine (better than linear)"
echo "  • Architecture: residual (improved)"
echo ""

python rl_vision/train_rl_vision.py \
  --total-timesteps $TOTAL_TIMESTEPS \
  --num-envs $NUM_ENVS \
  --max-level $MAX_LEVEL \
  --preset optimized \
  --lr-schedule cosine \
  --arch residual \
  --eval-freq 25000 \
  --checkpoint-freq 25000 \
  --n-eval-episodes 3 \
  --seed 42

echo ""
echo "Training completed!"
echo "Models saved to: models/vision/"
