# Common Pitfalls in Distributed Training

Learn from others' mistakes. These are the most common errors when learning DDP.

---

## Pitfall #1: Forgetting to Set Device

### The Bug
```python
# WRONG: Model stays on CPU
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = DDP(model)  # Error: DDP needs model on GPU
```

### The Fix
```python
# CORRECT: Move to GPU before wrapping with DDP
device = torch.device(f"cuda:{local_rank}")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = model.to(device)  # ← Move to GPU first
model = DDP(model, device_ids=[local_rank])
```

### Why This Happens
DDP wraps the model for distributed training. If the model isn't on a GPU, NCCL can't communicate gradients between GPUs.

---

## Pitfall #2: Printing from All Ranks

### The Bug
```python
# WRONG: Every GPU prints, output is chaos
print(f"Epoch {epoch}, Loss: {loss.item()}")

# Output:
# Epoch 0, Loss: 2.34  (from GPU 0)
# Epoch 0, Loss: 2.41  (from GPU 1)
# Epoch 0, Loss: 2.34  (from GPU 0)  ← Interleaved mess
# Epoch 0, Loss: 2.41  (from GPU 1)
```

### The Fix
```python
# CORRECT: Only rank 0 prints
if rank == 0:
    print(f"Epoch {epoch}, Loss: {loss.item()}")
```

### Why This Happens
Each process runs independently. Without rank checking, you get duplicate (and interleaved) output.

---

## Pitfall #3: Saving Checkpoint from All Ranks

### The Bug
```python
# WRONG: All GPUs try to save, causes file corruption
torch.save(model.state_dict(), "checkpoint.pt")
```

### The Fix
```python
# CORRECT: Only rank 0 saves
if rank == 0:
    torch.save(model.module.state_dict(), "checkpoint.pt")
```

### Why This Happens
Multiple processes writing to the same file = corruption. Also, all model copies are identical, so only one needs to save.

### Note the `.module`
When model is wrapped with DDP:
- `model` = DDP wrapper
- `model.module` = actual model

Save `model.module.state_dict()`, not `model.state_dict()`.

---

## Pitfall #4: Forgetting set_epoch()

### The Bug
```python
# WRONG: Same shuffling every epoch
sampler = DistributedSampler(dataset)
for epoch in range(epochs):
    for batch in dataloader:  # Same order every epoch!
        train(batch)
```

### The Fix
```python
# CORRECT: Different shuffle each epoch
sampler = DistributedSampler(dataset)
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # ← This changes the shuffle
    for batch in dataloader:
        train(batch)
```

### Why This Happens
DistributedSampler uses the epoch number as a random seed. Without setting it, you get the same "random" order every epoch.

---

## Pitfall #5: Data Not Moving to Device

### The Bug
```python
# WRONG: Data stays on CPU
for input_ids, labels in dataloader:
    output = model(input_ids)  # Error: model on GPU, data on CPU
```

### The Fix
```python
# CORRECT: Move data to same device as model
for input_ids, labels in dataloader:
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    output = model(input_ids)
```

### Why This Happens
DataLoader returns CPU tensors by default. Model is on GPU. They must be on the same device.

---

## Pitfall #6: Hardcoding GPU IDs

### The Bug
```python
# WRONG: Hardcoded GPU 0
model = model.to("cuda:0")
```

### The Fix
```python
# CORRECT: Use local_rank from environment
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
model = model.to(device)
```

### Why This Happens
In DDP, each process needs to use a different GPU. `LOCAL_RANK` tells each process which GPU it owns.

---

## Pitfall #7: Not Cleaning Up

### The Bug
```python
# WRONG: No cleanup, processes hang on next run
def main():
    dist.init_process_group(backend="nccl")
    train()
    # Missing cleanup!
```

### The Fix
```python
# CORRECT: Always cleanup
def main():
    dist.init_process_group(backend="nccl")
    try:
        train()
    finally:
        dist.destroy_process_group()  # ← Clean shutdown
```

### Why This Happens
NCCL creates GPU resources that must be freed. Without cleanup, you may hit GPU memory errors on subsequent runs.

---

## Pitfall #8: Wrong Loss Reduction

### The Bug
```python
# WRONG: Sum reduction + gradient sync = wrong gradients
criterion = nn.CrossEntropyLoss(reduction='sum')
loss = criterion(output, labels)
loss.backward()  # Gradients are 2x too large with 2 GPUs!
```

### The Fix
```python
# CORRECT: Mean reduction (default)
criterion = nn.CrossEntropyLoss(reduction='mean')  # or just CrossEntropyLoss()
loss = criterion(output, labels)
loss.backward()  # Gradients are averaged correctly
```

### Why This Happens
DDP averages gradients across GPUs. If you use sum reduction, each GPU sums its batch, then DDP averages those sums — giving you wrong values.

---

## Pitfall #9: Batch Size Confusion

### The Bug
```python
# Confusion about effective batch size
# Per-GPU batch size: 32
# With 2 GPUs, effective batch size: ???
```

### The Reality
```
Per-GPU batch size: 32
Number of GPUs: 2
Effective batch size: 32 × 2 = 64

If you want effective batch size 64:
  - Use batch_size=32 per GPU

If you want effective batch size 32:
  - Use batch_size=16 per GPU
```

### Why This Matters
Learning rate often scales with batch size. If you double your GPUs without adjusting learning rate, training might be unstable.

**Rule of thumb:** If you double GPUs, you might need to double learning rate (linear scaling).

---

## Pitfall #10: Loading Checkpoint Wrong

### The Bug
```python
# WRONG: Loading full state dict into non-DDP model
model = MyModel()
model.load_state_dict(torch.load("checkpoint.pt"))  # Error: keys don't match
```

### The Fix (Option A: Load before DDP wrap)
```python
# Load into base model, then wrap with DDP
model = MyModel()
model.load_state_dict(torch.load("checkpoint.pt"))
model = model.to(device)
model = DDP(model, device_ids=[local_rank])
```

### The Fix (Option B: Save from module)
```python
# When saving, save from .module
if rank == 0:
    torch.save(model.module.state_dict(), "checkpoint.pt")

# When loading, load into base model
model = MyModel()
model.load_state_dict(torch.load("checkpoint.pt"))
```

### Why This Happens
DDP wraps the model, adding a `.module` prefix to all keys. Saving `model.state_dict()` includes this prefix; saving `model.module.state_dict()` doesn't.

---

## Pitfall #11: Barrier Misuse

### The Bug
```python
# WRONG: Barrier inside conditional
if rank == 0:
    save_checkpoint()
    dist.barrier()  # Only rank 0 reaches this = DEADLOCK!
```

### The Fix
```python
# CORRECT: All ranks reach barrier
if rank == 0:
    save_checkpoint()
dist.barrier()  # All ranks wait here
```

### Why This Happens
`dist.barrier()` waits for ALL processes. If only some processes reach it, the others wait forever.

---

## Pitfall #12: Gradient Accumulation Errors

### The Bug
```python
# WRONG: zero_grad after backward, not before
loss.backward()
optimizer.zero_grad()  # Clears the gradients you just computed!
optimizer.step()  # Steps with zero gradients!
```

### The Fix
```python
# CORRECT: Standard order
optimizer.zero_grad()  # Clear old gradients
loss.backward()        # Compute new gradients
optimizer.step()       # Update weights
```

---

## Debugging Checklist

When something goes wrong in DDP, check these in order:

```
□ Is init_process_group called before any DDP operations?
□ Is each process using its assigned GPU (LOCAL_RANK)?
□ Is the model moved to GPU before wrapping with DDP?
□ Is data moved to the same device as the model?
□ Is DistributedSampler used?
□ Is set_epoch() called every epoch?
□ Are logs/saves only on rank 0?
□ Is destroy_process_group() called at the end?
□ Is loss reduction set to 'mean'?
□ Is .module used when saving checkpoint?
```

---

## Quick Reference: Correct DDP Template

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def main():
    # 1. Setup distributed
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")

    # 2. Create model on correct GPU
    model = MyModel().to(device)
    model = DDP(model, device_ids=[local_rank])

    # 3. Create data with DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # 4. Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important!
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()  # DDP syncs gradients automatically
            optimizer.step()

        # 5. Only rank 0 logs/saves
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            torch.save(model.module.state_dict(), "checkpoint.pt")

    # 6. Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

---

*Bookmark this page. You WILL hit these pitfalls. Everyone does.*
