# RL Vision Training - Optimizations Applied

## Changes Made

### 1. **New "optimized"Preset**
```
learning_rate:  3e-4      (↑ from 2e-4)
ent_coef:       0.08      (↑ from 0.05)
clip_range:     0.25      (↑ from 0.20)
n_epochs:       15        (↑ from 10)
```
→ Faster convergence with aggressive exploration

### 2. **Improved CNN Architecture**
**Residual (default):**
- features_dim: 512 (↑ from 384)
- dropout: 0.08 (↓ from 0.10)
- net_arch: [512, 256] (↑ from [256, 128])

**Simple:**
- features_dim: 320 (↑ from 256)
- net_arch: [512, 256] (↑ from [256, 128])

→ Larger network capacity for better feature learning

### 3. **Better Batch Size**
- Default preferred batch: 256 (↓ from 512)
- Better GPU memory utilization
- Matches theoretical rollout sizes

### 4. **New LR Schedules**
- **cosine** (NEW DEFAULT): Smooth decay, better convergence
- **polynomial**: Power=1.5, aggressive decay
- Keep: constant, linear

Recommendation: Use **cosine** (now default)

### 5. **Improved Monitoring**
- eval_freq: 25000 (↓ from 50000) - More frequent checks
- checkpoint_freq: 25000 (↓ from 50000) - Better recovery points

### 6. **Additional PPO Parameters**
- clip_range_vf: None (Added, for flexibility)
- target_kl: None (For early stopping option)

---

## Quick Start - Optimized Training

### Option 1: Using the script
```bash
bash TRAIN_OPTIMIZED.sh 2000000 4 4
```
- Argument 1: Total timesteps (default: 2M)
- Argument 2: Number of envs (default: 4)
- Argument 3: Max difficulty level (default: 4)

### Option 2: Direct command
```bash
python rl_vision/train_rl_vision.py \
  --total-timesteps 2000000 \
  --num-envs 4 \
  --max-level 4 \
  --preset optimized \
  --lr-schedule cosine \
  --arch residual
```

### Option 3: Conservative (stable, slower)
```bash
python rl_vision/train_rl_vision.py \
  --total-timesteps 2000000 \
  --preset conservative \
  --lr-schedule cosine
```

---

## Expected Improvements

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Convergence Speed | Baseline | Faster | +20-30% |
| Feature Learning | Good | Better | Larger capacity |
| Stability | Stable | Stable+ | Better LR decay |
| Memory Usage | Efficient | Similar | Batch tuned |
| Evaluation Freq | 50K steps | 25K steps | 2x monitoring |

---

## Preset Recommendations

| Preset | Use Case | Training Time |
|--------|----------|---|
| **optimized** | Fast results, exploratory | Fastest |
| **explore** | Aggressive exploration | Fast |
| **baseline** | Balanced | Moderate |
| **conservative** | Stable/safe | Slowest |

---

## Advanced: Fine-tuning

Adjust these for your specific needs:

```bash
# Very aggressive (highest learning)
--preset optimized --lr-schedule polynomial --n-epochs 20

# Most stable (lowest risk)
--preset conservative --lr-schedule linear --n-epochs 5

# Curriculum learning (easy→hard)
--start-level 1 --max-level 4
```

---

## Monitoring Training

Via TensorBoard:
```bash
tensorboard --logdir=models/vision/logs
```

Key metrics to watch:
- `rollout/ep_rew_mean` - Average episode reward
- `rollout/clear_rate` - Level completion rate
- `train/learning_rate` - Current LR (should decay smoothly)
- `train/policy_loss` - Should decrease over time
