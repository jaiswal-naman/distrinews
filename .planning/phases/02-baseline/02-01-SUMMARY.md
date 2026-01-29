# Phase 2 Summary: Baseline Training

**Status:** Complete ✓
**Completed:** 2025-01-30

---

## What Was Built

| File | Purpose |
|------|---------|
| `training/utils.py` | Helper functions (metrics, checkpointing, device detection) |
| `training/train_single.py` | Complete single-GPU training script |

---

## Key Concepts Introduced

### The Training Loop (5 Steps):
1. **Forward Pass** — Push data through model to get predictions
2. **Loss Computation** — Measure how wrong predictions are (CrossEntropyLoss)
3. **Backward Pass** — Compute gradients via backpropagation
4. **Optimizer Step** — Update weights: `weight = weight - lr * gradient`
5. **Zero Gradients** — Clear accumulated gradients for next batch

### Other Concepts:
- **Learning Rate** — How much to change weights (2e-5 for transformers)
- **Epochs vs Batches** — Epoch = full pass through data, Batch = subset
- **model.train() vs model.eval()** — Different behavior for dropout/batchnorm
- **torch.no_grad()** — Disable gradient tracking for inference
- **AdamW Optimizer** — Adam with weight decay (standard for transformers)

---

## Requirements Completed

- [x] SGT-01: Train model on single GPU
- [x] SGT-02: Log training loss per epoch
- [x] SGT-03: Log training accuracy per epoch
- [x] SGT-04: Track epoch training time
- [x] SGT-05: Save checkpoint after training

---

## How to Run

Quick test (100 samples, CPU):
```bash
python training/train_single.py --num_samples 100 --epochs 1
```

Full training (requires GPU for reasonable speed):
```bash
python training/train_single.py --epochs 3 --batch_size 32
```

Command-line options:
- `--epochs` — Number of training epochs (default: 3)
- `--batch_size` — Samples per batch (default: 32)
- `--learning_rate` — Learning rate (default: 2e-5)
- `--num_samples` — Limit samples for quick testing
- `--checkpoint_dir` — Where to save model (default: checkpoints/)

---

## What's Next

**Phase 3: Distributed Training (DDP)**
- Add distributed setup (process groups, ranks)
- Wrap model with DistributedDataParallel
- Use DistributedSampler for data sharding
- Handle logging/checkpointing on rank 0 only

This is where the real learning happens!

---
*Summary created: 2025-01-30*
