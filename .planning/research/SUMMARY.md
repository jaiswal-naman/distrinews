# Research Summary

Quick reference for DistriNews project. Read full docs in this folder for deep understanding.

---

## What We're Building

**DistriNews**: A distributed ML training system that:
1. Trains DistilBERT on AG News using PyTorch DDP (2 GPUs)
2. Deploys as a FastAPI inference service
3. Uses only free resources (Kaggle + HF Spaces)

---

## The Core Concept in 30 Seconds

**Distributed Data Parallel (DDP):**
- Same model on each GPU
- Different data on each GPU
- Gradients averaged automatically (AllReduce)
- Models stay synchronized

**Why it's faster:**
- 2 GPUs = 2x data processed per step
- Near-linear speedup

---

## Technology Choices

| Component | Choice | Why |
|-----------|--------|-----|
| Model | DistilBERT | Fast, fits on T4, transformer-based |
| Dataset | AG News | Right size (127K), 4-class, standard benchmark |
| Framework | PyTorch DDP | Industry standard, built-in |
| Backend | NCCL | Fastest for GPUs |
| Inference | FastAPI | Modern, fast, auto-docs |
| Training | Kaggle | Free 2x T4 GPUs |
| Deploy | HF Spaces | Free, git-based |

---

## Key DDP Components

```python
# Initialize (once)
dist.init_process_group(backend="nccl")

# Get identifiers
rank = int(os.environ["RANK"])           # Global ID
local_rank = int(os.environ["LOCAL_RANK"])  # GPU ID
world_size = int(os.environ["WORLD_SIZE"])  # Total GPUs

# Wrap model
model = model.to(device)
model = DDP(model, device_ids=[local_rank])

# Split data
sampler = DistributedSampler(dataset)

# Each epoch
sampler.set_epoch(epoch)

# Save (rank 0 only)
torch.save(model.module.state_dict(), "checkpoint.pt")

# Cleanup
dist.destroy_process_group()
```

---

## Critical Rules

1. **Only rank 0 prints and saves** — Avoid duplicate output
2. **Move data to device** — Match model's GPU
3. **Call set_epoch()** — Different shuffle each epoch
4. **Save model.module** — Not the DDP wrapper
5. **Always cleanup** — Prevent GPU leaks

---

## Project Structure

```
distrinews/
├── training/           # Distributed training code
│   ├── train_ddp.py   # Main script
│   ├── model.py       # DistilBERT wrapper
│   ├── dataset.py     # AG News loader
│   ├── utils.py       # Helpers
│   └── run_ddp.sh     # Launch script
├── inference/          # API server
│   ├── app.py         # FastAPI app
│   └── model_loader.py
├── checkpoints/        # Saved models
└── benchmarks/         # Performance docs
```

---

## Development Workflow

```
1. Write code locally (CPU) ──► Test logic
2. Upload to Kaggle ──────────► Train with DDP (2 GPUs)
3. Download checkpoint ───────► distilbert_agnews.pt
4. Deploy to HF Spaces ───────► Public API
```

---

## Expected Results

| Configuration | Time/Epoch | Final Accuracy |
|--------------|------------|----------------|
| 1 GPU (T4)   | ~8 min     | ~94%          |
| 2 GPU (DDP)  | ~4.5 min   | ~94%          |

**Speedup:** ~1.7-1.8x (not 2x due to communication overhead)

---

## Interview One-Liners

**What is DDP?**
> Multi-process data parallelism where each GPU has a model copy, sees different data, and gradients are synchronized via AllReduce.

**How do gradients sync?**
> DDP performs AllReduce during backward(), averaging gradients across all GPUs before optimizer step.

**Why DistributedSampler?**
> Ensures each GPU sees unique, non-overlapping data. Combined with set_epoch() for proper shuffling.

**Why only rank 0 saves?**
> All model copies are identical. Multiple processes writing one file causes corruption.

**Training vs inference distribution?**
> Training benefits from parallelism (throughput). Inference handles single requests — no parallelism needed.

---

## Common Pitfalls to Avoid

1. Forgetting `model.to(device)` before DDP wrap
2. Printing from all ranks (chaos)
3. Saving from all ranks (corruption)
4. Missing `set_epoch()` (same shuffle)
5. Not moving data to device
6. Hardcoding GPU IDs
7. Missing cleanup

---

## Files in This Research Folder

| File | Contents |
|------|----------|
| `CONCEPTS.md` | Deep dive into all DDP concepts |
| `STACK.md` | Technology choices explained |
| `ARCHITECTURE.md` | System diagrams and data flow |
| `PITFALLS.md` | Common mistakes and fixes |
| `SUMMARY.md` | This quick reference |

---

*Start with CONCEPTS.md for deep learning. Return here for quick reference.*
