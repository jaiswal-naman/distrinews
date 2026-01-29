# Distributed ML Training: The Complete Guide

This document teaches you everything you need to understand distributed machine learning training. Read this before touching any code.

---

## Table of Contents

1. [Why Distributed Training Exists](#why-distributed-training-exists)
2. [The Single GPU Baseline](#the-single-gpu-baseline)
3. [What is Data Parallelism?](#what-is-data-parallelism)
4. [How PyTorch DDP Works](#how-pytorch-ddp-works)
5. [Gradient Synchronization Explained](#gradient-synchronization-explained)
6. [The Distributed Sampler](#the-distributed-sampler)
7. [Processes, Ranks, and World Size](#processes-ranks-and-world-size)
8. [NCCL Backend](#nccl-backend)
9. [What torchrun Does](#what-torchrun-does)
10. [Training vs Inference: Why They're Different](#training-vs-inference-why-theyre-different)
11. [Interview Questions You'll Be Asked](#interview-questions-youll-be-asked)

---

## Why Distributed Training Exists

### The Problem

Imagine you have:
- A dataset with **1 million** training samples
- A model that takes **1 second** to process a batch of 32 samples
- You need to train for **10 epochs**

Math:
```
1,000,000 samples ÷ 32 samples/batch = 31,250 batches per epoch
31,250 batches × 1 second × 10 epochs = 312,500 seconds = 86.8 hours
```

**86 hours** to train your model. That's 3.5 days!

### The Solution

What if you had 2 GPUs?

Each GPU processes half the data simultaneously:
```
31,250 batches ÷ 2 GPUs = 15,625 batches per GPU
15,625 batches × 1 second × 10 epochs = 156,250 seconds = 43.4 hours
```

**~2x speedup!** With 8 GPUs, you'd be done in ~11 hours.

This is **distributed training** — using multiple GPUs to train faster.

---

## The Single GPU Baseline

Before understanding distributed training, you must understand normal training:

```
┌─────────────────────────────────────────────────────────┐
│                    SINGLE GPU TRAINING                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Dataset (all 1M samples)                              │
│          │                                              │
│          ▼                                              │
│   DataLoader (batch_size=32)                            │
│          │                                              │
│          ▼                                              │
│   ┌─────────────┐                                       │
│   │    GPU 0    │                                       │
│   │             │                                       │
│   │  1. Forward │  ← Model processes batch              │
│   │  2. Loss    │  ← Calculate how wrong we are         │
│   │  3. Backward│  ← Calculate gradients                │
│   │  4. Update  │  ← Optimizer updates weights          │
│   │             │                                       │
│   └─────────────┘                                       │
│                                                         │
│   Repeat for all 31,250 batches...                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Key concept:** In single GPU training, ONE GPU sees ALL the data and computes ALL the gradients.

---

## What is Data Parallelism?

Data Parallelism means: **Split the DATA, not the model.**

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PARALLELISM                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Dataset (1M samples)                                          │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────────────────────────────┐                      │
│   │       DistributedSampler             │                      │
│   │  Splits data: GPU0 gets samples 0,2,4...                    │
│   │               GPU1 gets samples 1,3,5...                    │
│   └──────────────────────────────────────┘                      │
│          │                    │                                 │
│          ▼                    ▼                                 │
│   ┌─────────────┐      ┌─────────────┐                          │
│   │    GPU 0    │      │    GPU 1    │                          │
│   │             │      │             │                          │
│   │  Model      │      │  Model      │  ← SAME model on both!   │
│   │  (copy)     │      │  (copy)     │                          │
│   │             │      │             │                          │
│   │  Processes  │      │  Processes  │                          │
│   │  samples    │      │  samples    │                          │
│   │  0,2,4...   │      │  1,3,5...   │                          │
│   │             │      │             │                          │
│   └──────┬──────┘      └──────┬──────┘                          │
│          │                    │                                 │
│          │    GRADIENTS       │                                 │
│          └────────┬───────────┘                                 │
│                   ▼                                             │
│          ┌───────────────┐                                      │
│          │   AllReduce   │  ← Average gradients from all GPUs   │
│          └───────────────┘                                      │
│                   │                                             │
│                   ▼                                             │
│          Both GPUs update with SAME averaged gradients          │
│          Models stay synchronized!                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why this works:

1. **Each GPU has the SAME model** (identical weights)
2. **Each GPU sees DIFFERENT data** (no overlap)
3. **Gradients are averaged** across all GPUs
4. **Same update applied** to all model copies
5. **Models stay in sync** throughout training

**The magic:** Mathematically, training on 2 GPUs with batch_size=32 each is equivalent to training on 1 GPU with batch_size=64. Same gradients, same model, just faster.

---

## How PyTorch DDP Works

DDP = **D**istributed **D**ata **P**arallel

### Step-by-step breakdown:

```python
# 1. Initialize the process group (all GPUs learn about each other)
torch.distributed.init_process_group(backend="nccl")

# 2. Create model and move to this GPU
model = MyModel().to(device)

# 3. Wrap model with DDP (this is the magic)
model = DistributedDataParallel(model, device_ids=[local_rank])

# 4. Use DistributedSampler (ensures no data overlap)
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)

# 5. Training loop (looks almost normal!)
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Important for shuffling
    for batch in dataloader:
        loss = model(batch)
        loss.backward()       # ← DDP automatically syncs gradients here!
        optimizer.step()
```

### What DDP does automatically:

| When | What DDP Does | Why |
|------|---------------|-----|
| Model creation | Broadcasts weights from rank 0 to all GPUs | Start synchronized |
| loss.backward() | AllReduce gradients across all GPUs | Everyone gets same gradients |
| optimizer.step() | Each GPU updates independently | Same gradients = same weights |

---

## Gradient Synchronization Explained

This is the **most important concept** for interviews.

### The Problem Without Sync

```
GPU 0 sees batch A → computes gradients_A
GPU 1 sees batch B → computes gradients_B

If each GPU updates independently:
  GPU 0 weights = weights - lr * gradients_A
  GPU 1 weights = weights - lr * gradients_B

DISASTER! Models diverge! They're no longer the same!
```

### The Solution: AllReduce

```
┌─────────────────────────────────────────────────────────────────┐
│                      AllReduce Operation                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   GPU 0: gradients = [0.1, 0.2, 0.3]                            │
│   GPU 1: gradients = [0.3, 0.4, 0.1]                            │
│                                                                 │
│                     AllReduce (average)                         │
│                            │                                    │
│                            ▼                                    │
│                                                                 │
│   GPU 0: gradients = [0.2, 0.3, 0.2]  ← SAME!                   │
│   GPU 1: gradients = [0.2, 0.3, 0.2]  ← SAME!                   │
│                                                                 │
│   Now both GPUs apply the SAME gradient update.                 │
│   Models stay identical.                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Interview Answer:

> "In DDP, when backward() is called, PyTorch automatically performs an AllReduce operation across all GPUs. This averages the gradients from each GPU's local batch, ensuring every GPU has identical gradients. When the optimizer steps, all model copies update identically, maintaining synchronization throughout training."

---

## The Distributed Sampler

### Problem Without DistributedSampler

```
Normal DataLoader on 2 GPUs:

GPU 0: batch 1, batch 2, batch 3...
GPU 1: batch 1, batch 2, batch 3...

SAME batches! Wasted computation!
```

### Solution: DistributedSampler

```python
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,  # Total number of GPUs
    rank=rank,                 # This GPU's ID
    shuffle=True
)
```

```
With DistributedSampler on 2 GPUs:

Dataset indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

GPU 0 (rank 0): [0, 2, 4, 6, 8]  ← Even indices
GPU 1 (rank 1): [1, 3, 5, 7, 9]  ← Odd indices

No overlap! Each GPU sees unique data.
```

### Why set_epoch() Matters

```python
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # CRITICAL!
```

Without `set_epoch()`:
- Same shuffle every epoch
- GPU 0 always sees [0, 2, 4, 6, 8]
- GPU 1 always sees [1, 3, 5, 7, 9]
- Some data combinations never trained together

With `set_epoch()`:
- Different shuffle each epoch
- Epoch 0: GPU0=[0,4,2,8,6], GPU1=[1,5,3,9,7]
- Epoch 1: GPU0=[3,7,1,5,9], GPU1=[2,6,0,4,8]
- Better generalization!

---

## Processes, Ranks, and World Size

### Vocabulary

| Term | Meaning | Example |
|------|---------|---------|
| **Process** | One instance of your training script | 2 GPUs = 2 processes |
| **Rank** | Unique ID for each process | 0, 1, 2... |
| **Local Rank** | GPU ID on this machine | 0 or 1 for 2-GPU machine |
| **World Size** | Total number of processes | 2 for 2-GPU training |
| **Rank 0** | The "main" process | Does logging, checkpointing |

### Visual

```
┌─────────────────────────────────────────────────────────┐
│                    Single Machine                        │
│                                                         │
│   Process 0          Process 1                          │
│   ┌─────────┐       ┌─────────┐                         │
│   │ Rank: 0 │       │ Rank: 1 │                         │
│   │ Local:0 │       │ Local:1 │                         │
│   │  GPU 0  │       │  GPU 1  │                         │
│   └─────────┘       └─────────┘                         │
│                                                         │
│   World Size = 2                                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### In Code

```python
import os
import torch.distributed as dist

# These are set by torchrun automatically
rank = int(os.environ["RANK"])              # 0 or 1
local_rank = int(os.environ["LOCAL_RANK"])  # 0 or 1
world_size = int(os.environ["WORLD_SIZE"])  # 2

# Only rank 0 should print/save
if rank == 0:
    print("Training started!")
    save_checkpoint(model)
```

---

## NCCL Backend

NCCL = **N**VIDIA **C**ollective **C**ommunication **L**ibrary

### What it does:

NCCL handles GPU-to-GPU communication. When gradients need to be synchronized, NCCL:

1. Uses fast GPU interconnects (NVLink if available)
2. Efficiently transfers data between GPUs
3. Performs AllReduce operations in parallel with computation

### In Code

```python
torch.distributed.init_process_group(
    backend="nccl",    # Use NCCL for GPU communication
    init_method="env://",  # Get config from environment variables
)
```

### Other Backends

| Backend | Use Case |
|---------|----------|
| **nccl** | GPU training (fastest) |
| **gloo** | CPU training or CPU-GPU mixed |
| **mpi** | HPC clusters |

**For this project:** We use NCCL because we're training on GPUs.

---

## What torchrun Does

`torchrun` is PyTorch's official distributed launcher.

### Without torchrun (manual)

You'd have to:
1. Start process 0 manually, set RANK=0, LOCAL_RANK=0, etc.
2. Start process 1 manually, set RANK=1, LOCAL_RANK=1, etc.
3. Make sure they can find each other
4. Handle errors and cleanup

### With torchrun (easy)

```bash
torchrun --nproc_per_node=2 train.py
```

torchrun automatically:
1. Spawns 2 processes (one per GPU)
2. Sets environment variables (RANK, LOCAL_RANK, WORLD_SIZE)
3. Sets up communication between processes
4. Handles cleanup if one process fails

### Environment Variables Set by torchrun

| Variable | Meaning |
|----------|---------|
| RANK | Global rank (0, 1, 2...) |
| LOCAL_RANK | Rank on this machine (0 or 1 for 2-GPU) |
| WORLD_SIZE | Total processes |
| MASTER_ADDR | IP of rank 0 machine |
| MASTER_PORT | Port for communication |

---

## Training vs Inference: Why They're Different

### Training: Distributed is Faster

```
Training:
- Process MILLIONS of samples
- Need to see all data multiple times (epochs)
- Computation-bound: more GPUs = faster

Use DDP: Split data across GPUs, train in parallel
```

### Inference: Distributed is Overkill

```
Inference:
- Process ONE sample at a time (user request)
- No epochs, no gradients, no synchronization
- Often I/O bound (network latency), not compute-bound

Use single model: Load once, serve requests
```

### Why We DON'T Distribute Inference

1. **No gradients** = No need for synchronization
2. **Single request** = Can't split one sample across GPUs
3. **Latency matters** = Adding GPU communication adds delay
4. **Simpler deployment** = One container, one model, easier ops

### Our Architecture

```
TRAINING (distributed)          INFERENCE (centralized)
┌─────────────────────┐        ┌─────────────────────┐
│   GPU 0   GPU 1     │        │                     │
│   ┌───┐   ┌───┐     │        │   ┌───────────┐     │
│   │DDP│   │DDP│     │        │   │  FastAPI  │     │
│   │   │◄─►│   │     │        │   │   Server  │     │
│   └─┬─┘   └─┬─┘     │        │   └─────┬─────┘     │
│     │       │       │        │         │           │
│     └───┬───┘       │        │         ▼           │
│         ▼           │        │   ┌───────────┐     │
│  ┌─────────────┐    │        │   │   Model   │     │
│  │ checkpoint  │────┼───────►│   │ (single)  │     │
│  └─────────────┘    │        │   └───────────┘     │
└─────────────────────┘        └─────────────────────┘

Saves one checkpoint ──────────► Loads one checkpoint
```

---

## Interview Questions You'll Be Asked

### Q1: "What is the difference between DataParallel and DistributedDataParallel?"

**Answer:**
> DataParallel (DP) uses a single process with multiple threads. It has a bottleneck because gradients must be gathered to one GPU. DistributedDataParallel (DDP) uses multiple processes, one per GPU, with AllReduce for gradient sync. DDP is faster because it overlaps gradient computation with communication and doesn't have the single-GPU bottleneck.

### Q2: "How does gradient synchronization work in DDP?"

**Answer:**
> During backward(), DDP performs AllReduce to average gradients across all GPUs. This happens automatically when loss.backward() is called. The gradients are synchronized before optimizer.step(), ensuring all model replicas receive identical gradient updates and stay synchronized.

### Q3: "Why do you need DistributedSampler?"

**Answer:**
> Without DistributedSampler, each GPU would load the same batches, wasting computation. DistributedSampler partitions the dataset so each GPU sees unique, non-overlapping samples. Combined with set_epoch() for shuffling, it ensures efficient data utilization across GPUs.

### Q4: "What happens if one GPU is slower than the other?"

**Answer:**
> DDP uses synchronous training — all GPUs must complete their forward and backward passes before gradients are synchronized. If one GPU is slower (straggler), all GPUs wait. This maintains correctness but can reduce efficiency. Solutions include better load balancing or asynchronous methods (though DDP is synchronous).

### Q5: "How do you save and load checkpoints in distributed training?"

**Answer:**
> Only rank 0 should save checkpoints to avoid file conflicts. The model's state_dict is saved (not the DDP wrapper). On loading, all ranks load the same checkpoint. You can use dist.barrier() to ensure rank 0 finishes saving before other ranks proceed.

---

## Summary: The Key Concepts

| Concept | One-liner |
|---------|-----------|
| **Data Parallelism** | Same model, different data, synchronized gradients |
| **DDP** | PyTorch's efficient multi-process data parallelism |
| **AllReduce** | Operation that averages gradients across all GPUs |
| **DistributedSampler** | Ensures each GPU sees unique data |
| **Rank** | Unique ID for each training process |
| **World Size** | Total number of training processes |
| **torchrun** | Launcher that sets up distributed environment |
| **NCCL** | Fast GPU-to-GPU communication backend |

---

*Read this document until you can explain each concept without looking. Then move to the code.*
