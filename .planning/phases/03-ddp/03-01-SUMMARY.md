# Phase 3 Summary: Distributed Data Parallel Training

**Status:** Complete ✓
**Completed:** 2025-01-30

---

## What Was Built

| File | Purpose |
|------|---------|
| `training/train_ddp.py` | Full DDP training script with all concepts explained |
| `training/run_ddp.sh` | torchrun launcher for Linux/Mac |
| `training/run_ddp.bat` | torchrun launcher for Windows |

---

## Key DDP Concepts Introduced

### 1. Process Group Initialization
```python
dist.init_process_group(backend="nccl")
```
- Creates communication channels between GPUs
- All processes must call this before any DDP operations
- Uses NCCL (NVIDIA's fast GPU communication library)

### 2. Ranks and World Size
```python
rank = int(os.environ["RANK"])           # Global ID (0, 1, 2...)
local_rank = int(os.environ["LOCAL_RANK"])  # GPU on this machine
world_size = int(os.environ["WORLD_SIZE"])  # Total GPUs
```
- Each process has unique rank
- LOCAL_RANK determines which GPU to use
- Rank 0 is the "main" process (logging, checkpointing)

### 3. DistributedDataParallel Wrapper
```python
model = DDP(model, device_ids=[local_rank])
```
- Broadcasts weights from rank 0 to all ranks
- Registers hooks for automatic gradient synchronization
- During backward(), gradients are averaged via AllReduce

### 4. DistributedSampler
```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
```
- Ensures each GPU sees different data
- GPU 0: samples [0, 2, 4, 6, ...]
- GPU 1: samples [1, 3, 5, 7, ...]

### 5. set_epoch() for Shuffling
```python
sampler.set_epoch(epoch)  # CRITICAL!
```
- Different shuffle each epoch
- Without this: same data split every epoch
- Uses epoch as random seed

### 6. Rank 0 Only Operations
```python
if rank == 0:
    print("...")
    save_checkpoint(...)
```
- Avoid duplicate output
- Avoid file write conflicts

### 7. Clean Shutdown
```python
dist.destroy_process_group()
```
- Release GPU resources
- Close network connections
- Always in try/finally block

---

## Requirements Completed

- [x] DDP-01: Initialize process group with NCCL backend
- [x] DDP-02: Each process uses assigned GPU (LOCAL_RANK)
- [x] DDP-03: Model wrapped with DistributedDataParallel
- [x] DDP-04: Data sharded with DistributedSampler
- [x] DDP-05: sampler.set_epoch() called every epoch
- [x] DDP-06: Gradients synchronized automatically
- [x] DDP-07: Only rank 0 logs to console
- [x] DDP-08: Only rank 0 saves checkpoint
- [x] DDP-09: Proper cleanup with destroy_process_group
- [x] CHK-01: Save model state_dict (not DDP wrapper)
- [x] CHK-02: Save tokenizer alongside model
- [x] CHK-03: Checkpoint saved to checkpoints/ directory
- [x] CHK-04: Checkpoint filename: distilbert_agnews.pt
- [x] LCH-01: Shell script to launch DDP training
- [x] LCH-02: Uses torchrun with --nproc_per_node
- [x] LCH-03: Configurable number of GPUs

---

## How to Run

### On Kaggle (with 2 GPUs):
```bash
# In a Kaggle notebook cell:
!torchrun --nproc_per_node=2 --standalone training/train_ddp.py --epochs 3
```

### On Windows (testing with CPU):
```bash
# Single process (CPU testing)
python training/train_ddp.py --epochs 1 --num_samples 100
```

### On Linux/Mac (with GPUs):
```bash
./training/run_ddp.sh 2  # 2 GPUs
```

---

## The DDP Flow Visualized

```
┌────────────────────────────────────────────────────────────────┐
│                    torchrun --nproc_per_node=2                  │
│                              │                                  │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                  │
│       ┌─────────────┐                 ┌─────────────┐          │
│       │  Process 0  │                 │  Process 1  │          │
│       │  RANK=0     │                 │  RANK=1     │          │
│       │  GPU 0      │                 │  GPU 1      │          │
│       └──────┬──────┘                 └──────┬──────┘          │
│              │                               │                  │
│              │    init_process_group()       │                  │
│              └───────────────┬───────────────┘                  │
│                              ▼                                  │
│                    ┌─────────────────┐                          │
│                    │  Process Group  │                          │
│                    │   (connected)   │                          │
│                    └────────┬────────┘                          │
│                              │                                  │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                  │
│       ┌─────────────┐                 ┌─────────────┐          │
│       │ DDP(model)  │ ◄── AllReduce ──► │ DDP(model)  │          │
│       │ Sampler:    │    (gradients)    │ Sampler:    │          │
│       │ [0,2,4,6..] │                   │ [1,3,5,7..] │          │
│       └──────┬──────┘                 └─────────────┘          │
│              │                                                  │
│              ▼                                                  │
│       Save checkpoint                                           │
│       (rank 0 only)                                            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## What's Next

**Phase 4: Inference API**
- Load the trained checkpoint
- Create FastAPI server
- POST /predict endpoint

The hard part is done! Inference is straightforward.

---
*Summary created: 2025-01-30*
