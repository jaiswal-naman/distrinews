"""
training/train_ddp.py - Distributed Data Parallel Training Script

THIS IS THE MOST IMPORTANT FILE IN THE PROJECT.

This script trains the news classifier using PyTorch's DistributedDataParallel (DDP)
across multiple GPUs. Every DDP concept is explained in detail.

=============================================================================
REVIEW: WHAT IS DISTRIBUTED DATA PARALLEL?
=============================================================================

Read .planning/research/CONCEPTS.md for the full explanation.
Here's a quick recap:

DDP = Same model on each GPU, different data, synchronized gradients.

    ┌─────────────┐      ┌─────────────┐
    │    GPU 0    │      │    GPU 1    │
    │             │      │             │
    │  Model      │      │  Model      │  ← SAME weights
    │  (copy)     │      │  (copy)     │
    │             │      │             │
    │  Batch A    │      │  Batch B    │  ← DIFFERENT data
    │             │      │             │
    └──────┬──────┘      └──────┬──────┘
           │                    │
           │   GRADIENTS        │
           └────────┬───────────┘
                    ▼
           ┌───────────────┐
           │   AllReduce   │  ← AVERAGE gradients
           └───────────────┘
                    │
                    ▼
           Both GPUs update with SAME gradients
           Models stay identical!

WHY IT'S FASTER:
- 2 GPUs process 2x the data in the same time
- Near-linear speedup (1.7-1.8x with 2 GPUs)

=============================================================================
HOW THIS SCRIPT DIFFERS FROM train_single.py
=============================================================================

SINGLE GPU (train_single.py):
    model = NewsClassifier()
    dataloader = DataLoader(dataset, shuffle=True)

DISTRIBUTED (train_ddp.py):
    dist.init_process_group(backend="nccl")  # NEW: Initialize DDP
    model = NewsClassifier()
    model = DDP(model, device_ids=[local_rank])  # NEW: Wrap with DDP
    sampler = DistributedSampler(dataset)  # NEW: Shard data
    dataloader = DataLoader(dataset, sampler=sampler)

The training loop is ALMOST identical. DDP handles gradient sync automatically.

=============================================================================
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import NewsClassifier
from training.dataset import load_agnews, get_tokenizer, AGNewsDataset
from training.utils import (
    AverageMeter,
    compute_accuracy,
    save_checkpoint,
    format_time
)


# =============================================================================
# DDP SETUP FUNCTIONS
# =============================================================================

def setup_distributed():
    """
    Initialize the distributed environment.

    WHAT THIS FUNCTION DOES:
    -----------------------
    1. Read environment variables set by torchrun
    2. Initialize the process group (all GPUs learn about each other)
    3. Set the device for this process

    RETURNS:
    --------
    tuple: (rank, local_rank, world_size, device)

    ==========================================================================
    CONCEPT: ENVIRONMENT VARIABLES
    ==========================================================================

    torchrun sets these environment variables BEFORE your script runs:

    RANK:        Global process ID (0, 1, 2, ... across all machines)
    LOCAL_RANK:  GPU ID on THIS machine (0 or 1 for 2-GPU machine)
    WORLD_SIZE:  Total number of processes (= total GPUs)
    MASTER_ADDR: IP address of rank 0 (for communication)
    MASTER_PORT: Port for communication

    Example with 2 GPUs on 1 machine:
        Process 0: RANK=0, LOCAL_RANK=0, WORLD_SIZE=2
        Process 1: RANK=1, LOCAL_RANK=1, WORLD_SIZE=2

    ==========================================================================
    CONCEPT: PROCESS GROUP
    ==========================================================================

    init_process_group() creates a "group" where all processes can communicate.

    Think of it like a group chat:
    - All GPUs join the chat
    - They can send messages to each other (gradients)
    - They can synchronize (wait for everyone)

    PARAMETERS:
    - backend="nccl": Use NVIDIA's fast GPU communication library
    - init_method="env://": Read connection info from environment variables

    After this call, ALL processes can communicate with each other.

    ==========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Read environment variables
    # -------------------------------------------------------------------------
    # These are set by torchrun automatically
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # -------------------------------------------------------------------------
    # Step 2: Initialize process group
    # -------------------------------------------------------------------------
    # This is the FIRST DDP call. It MUST happen before any other DDP operations.
    #
    # WHAT HAPPENS INSIDE:
    # 1. This process connects to MASTER_ADDR:MASTER_PORT
    # 2. It registers itself with rank and world_size
    # 3. Once all processes connect, the group is ready
    # 4. NCCL creates GPU-to-GPU communication channels
    #
    # If any process fails to connect, ALL processes hang!
    # That's why proper error handling is important.
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",      # NVIDIA Collective Communication Library
            init_method="env://",  # Read MASTER_ADDR, MASTER_PORT from env
            world_size=world_size,
            rank=rank
        )
        print(f"[Rank {rank}] Initialized process group (world_size={world_size})")
    else:
        print("Running in single-GPU mode (no distributed)")

    # -------------------------------------------------------------------------
    # Step 3: Set device for this process
    # -------------------------------------------------------------------------
    # Each process MUST use a different GPU.
    # LOCAL_RANK tells us which GPU this process owns.
    #
    # IMPORTANT: We use LOCAL_RANK (not RANK) because:
    # - RANK is global across machines (0, 1, 2, 3, ...)
    # - LOCAL_RANK is per-machine GPU index (0 or 1)
    #
    # On a single machine with 2 GPUs:
    #   Process 0: LOCAL_RANK=0 → cuda:0
    #   Process 1: LOCAL_RANK=1 → cuda:1
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)  # Set default CUDA device
    else:
        device = torch.device("cpu")
        print(f"[Rank {rank}] WARNING: CUDA not available, using CPU")

    return rank, local_rank, world_size, device


def cleanup_distributed():
    """
    Clean up the distributed environment.

    WHAT THIS FUNCTION DOES:
    -----------------------
    Destroys the process group, releasing all resources.

    WHY IT'S IMPORTANT:
    ------------------
    - Releases GPU memory used for communication
    - Closes network connections
    - Allows clean shutdown

    If you don't call this:
    - Subsequent DDP runs might fail
    - GPU memory might leak
    - Processes might hang

    WHEN TO CALL:
    -------------
    At the very end of training, after all work is done.
    Use try/finally to ensure it's always called!
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """
    Check if this is the main process (rank 0).

    WHY WE NEED THIS:
    ----------------
    In DDP, all processes run the same code.
    But some things should only happen ONCE:
    - Printing to console (avoid duplicate lines)
    - Saving checkpoints (avoid file conflicts)
    - Logging to TensorBoard
    - Downloading datasets

    By convention, rank 0 is the "main" process that does these things.

    EXAMPLE:
    --------
    # Without this check:
    print("Training started")  # Prints twice with 2 GPUs!

    # With this check:
    if is_main_process(rank):
        print("Training started")  # Prints once
    """
    return rank == 0


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train news classifier with DDP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size PER GPU (effective = batch_size * num_gpus)")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)

    # Data parameters
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit training samples (for quick testing)")

    # Output parameters
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    return parser.parse_args()


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    rank: int,
    world_size: int
) -> tuple:
    """
    Train for one epoch with DDP.

    ==========================================================================
    KEY DIFFERENCE FROM SINGLE-GPU TRAINING:
    ==========================================================================

    The training loop is ALMOST THE SAME. The magic happens automatically:

    Single GPU:
        loss.backward()  # Compute gradients for this batch

    DDP:
        loss.backward()  # Compute gradients AND synchronize across GPUs!

    DDP hooks into backward() and automatically:
    1. Computes local gradients (same as single GPU)
    2. AllReduces gradients across all GPUs
    3. Averages the gradients
    4. All GPUs now have identical gradients

    You don't write any extra code! DDP handles it.

    ==========================================================================
    """
    model.train()

    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("accuracy")

    # Only show progress bar on rank 0 (avoid duplicate bars)
    if is_main_process(rank):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", ncols=100)
    else:
        pbar = dataloader  # No progress bar for other ranks

    for batch in pbar:
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_size = input_ids.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (same as single GPU)
        logits = model(input_ids, attention_mask)

        # Compute loss (same as single GPU)
        loss = criterion(logits, labels)

        # =====================================================================
        # BACKWARD PASS - WHERE THE DDP MAGIC HAPPENS
        # =====================================================================
        # This single line does A LOT more in DDP:
        #
        # 1. Compute local gradients (same as single GPU)
        # 2. AUTOMATICALLY start AllReduce communication
        # 3. Average gradients across all GPUs
        # 4. All GPUs end up with identical averaged gradients
        #
        # HOW IT WORKS:
        # DDP registers "hooks" on each parameter. When backward() computes
        # a gradient, the hook triggers AllReduce. This happens in parallel
        # with gradient computation for efficiency.
        #
        # WHY AVERAGE?
        # Each GPU computed gradients from different data. Averaging them
        # is equivalent to computing gradients on a batch that's world_size
        # times larger. This is mathematically correct for SGD.
        # =====================================================================
        loss.backward()

        # Optimizer step (same as single GPU)
        # Because gradients are now identical across GPUs,
        # all GPUs make the same update to weights.
        # Models stay synchronized!
        optimizer.step()

        # Track metrics
        loss_meter.update(loss.item(), batch_size)
        acc = compute_accuracy(logits.detach(), labels)
        acc_meter.update(acc, batch_size)

        # Update progress bar (only rank 0)
        if is_main_process(rank):
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "acc": f"{acc_meter.avg:.2f}%"
            })

    return loss_meter.avg, acc_meter.avg


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rank: int
) -> tuple:
    """
    Evaluate the model.

    NOTE: In a full implementation, you'd want to gather predictions
    from all GPUs for proper evaluation. For simplicity, we evaluate
    only on rank 0's portion of the data.
    """
    model.eval()

    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("accuracy")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            batch_size = input_ids.size(0)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss_meter.update(loss.item(), batch_size)
            acc = compute_accuracy(logits, labels)
            acc_meter.update(acc, batch_size)

    return loss_meter.avg, acc_meter.avg


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main training function with DDP.

    ==========================================================================
    DDP TRAINING FLOW
    ==========================================================================

    1. SETUP
       - Parse arguments
       - Initialize process group (all GPUs connect)
       - Create model and wrap with DDP
       - Create DistributedSampler

    2. TRAINING LOOP
       - For each epoch:
         - sampler.set_epoch(epoch)  ← Important for shuffling!
         - Train all batches
         - Evaluate
         - Save checkpoint (rank 0 only)

    3. CLEANUP
       - Destroy process group

    ==========================================================================
    """
    # Parse arguments
    args = parse_args()

    # =========================================================================
    # STEP 1: Initialize distributed environment
    # =========================================================================
    rank, local_rank, world_size, device = setup_distributed()

    # Print config (only rank 0)
    if is_main_process(rank):
        print("\n" + "=" * 60)
        print("DDP TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"World size (GPUs):    {world_size}")
        print(f"Batch size per GPU:   {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Learning rate:        {args.learning_rate}")
        print(f"Epochs:               {args.epochs}")
        print(f"Device:               {device}")
        print("=" * 60 + "\n")

    # =========================================================================
    # STEP 2: Load data
    # =========================================================================
    if is_main_process(rank):
        print("Loading data...")

    tokenizer = get_tokenizer()

    # Load datasets
    train_data = load_agnews("train")
    test_data = load_agnews("test")

    # Limit samples for testing
    if args.num_samples is not None:
        train_data = train_data[:args.num_samples]
        test_data = test_data[:min(args.num_samples // 4, len(test_data))]

    # Create datasets
    train_dataset = AGNewsDataset(train_data, tokenizer, args.max_length)
    test_dataset = AGNewsDataset(test_data, tokenizer, args.max_length)

    # =========================================================================
    # STEP 3: Create DistributedSampler
    # =========================================================================
    #
    # THIS IS CRITICAL FOR DDP!
    #
    # Without DistributedSampler:
    #   GPU 0: batches [0, 1, 2, 3, ...]
    #   GPU 1: batches [0, 1, 2, 3, ...]  ← SAME DATA! Wasted!
    #
    # With DistributedSampler:
    #   GPU 0: batches [0, 2, 4, 6, ...]  ← Even indices
    #   GPU 1: batches [1, 3, 5, 7, ...]  ← Odd indices
    #
    # Each GPU sees UNIQUE data. Together they cover the full dataset.
    #
    # PARAMETERS:
    # - dataset: The dataset to sample from
    # - num_replicas: Number of GPUs (world_size)
    # - rank: This GPU's ID
    # - shuffle: Whether to shuffle (True for training)
    # =========================================================================
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True  # Shuffle for training
    )

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False  # Don't shuffle for evaluation
    )

    # Create dataloaders
    # NOTE: shuffle=False because DistributedSampler handles shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,  # Use distributed sampler
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True
    )

    if is_main_process(rank):
        print(f"Train samples: {len(train_dataset):,} (each GPU sees {len(train_dataset)//world_size:,})")
        print(f"Test samples:  {len(test_dataset):,}")

    # =========================================================================
    # STEP 4: Create model and wrap with DDP
    # =========================================================================
    if is_main_process(rank):
        print("\nInitializing model...")

    # Create model
    model = NewsClassifier(num_labels=4)

    # Move to device BEFORE wrapping with DDP
    # This is important! DDP expects model already on the correct device.
    model = model.to(device)

    # =========================================================================
    # WRAP MODEL WITH DistributedDataParallel
    # =========================================================================
    #
    # THIS IS THE KEY DDP OPERATION!
    #
    # DDP wraps your model and adds:
    # 1. Gradient synchronization hooks
    # 2. Communication buffers
    # 3. Bucket management for efficient AllReduce
    #
    # PARAMETERS:
    # - model: Your model (already on GPU)
    # - device_ids: Which GPU(s) this process uses [local_rank]
    #
    # WHAT HAPPENS:
    # 1. DDP broadcasts weights from rank 0 to all other ranks
    #    → All models start with identical weights
    # 2. DDP registers hooks on each parameter
    #    → When backward() is called, hooks trigger AllReduce
    # 3. Gradients are synchronized automatically during backward()
    #
    # AFTER THIS:
    # - model.forward() works normally
    # - model.backward() synchronizes gradients automatically
    # - The actual model is at model.module
    # =========================================================================
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        if is_main_process(rank):
            print(f"Model wrapped with DDP (device_ids=[{local_rank}])")
    else:
        if is_main_process(rank):
            print("Single GPU mode - no DDP wrapper")

    # Print model info (only rank 0)
    if is_main_process(rank):
        # Get actual model (unwrap DDP if needed)
        actual_model = model.module if hasattr(model, 'module') else model
        num_params = sum(p.numel() for p in actual_model.parameters())
        print(f"Model parameters: {num_params:,}")

    # =========================================================================
    # STEP 5: Create optimizer and loss function
    # =========================================================================
    criterion = nn.CrossEntropyLoss()

    # Note: Optimizer is created AFTER DDP wrapping
    # This ensures it sees the DDP-wrapped parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )

    # =========================================================================
    # STEP 6: Training loop
    # =========================================================================
    if is_main_process(rank):
        print("\n" + "=" * 60)
        print("STARTING DDP TRAINING")
        print("=" * 60 + "\n")

    best_accuracy = 0.0
    total_start_time = time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            # =================================================================
            # CRITICAL: Set epoch for sampler
            # =================================================================
            # This is ESSENTIAL for proper shuffling in DDP!
            #
            # Without set_epoch():
            #   - Sampler uses same "random" order every epoch
            #   - GPU 0 always gets samples [0, 2, 4, ...]
            #   - GPU 1 always gets samples [1, 3, 5, ...]
            #   - Some sample pairs never train together
            #
            # With set_epoch(epoch):
            #   - Different shuffle each epoch
            #   - Epoch 1: GPU0=[0,4,2,...], GPU1=[1,5,3,...]
            #   - Epoch 2: GPU0=[3,7,1,...], GPU1=[2,6,0,...]
            #   - Better generalization!
            #
            # The epoch number is used as a random seed.
            # =================================================================
            train_sampler.set_epoch(epoch)

            # Train
            train_loss, train_acc = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                num_epochs=args.epochs,
                rank=rank,
                world_size=world_size
            )

            # Evaluate
            test_loss, test_acc = evaluate(
                model=model,
                dataloader=test_loader,
                criterion=criterion,
                device=device,
                rank=rank
            )

            epoch_time = time.time() - epoch_start_time

            # Print summary (only rank 0)
            if is_main_process(rank):
                print(f"\nEpoch [{epoch}/{args.epochs}] - {format_time(epoch_time)}")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")

            # Track best accuracy
            if test_acc > best_accuracy:
                best_accuracy = test_acc

            # =================================================================
            # BARRIER: Synchronize all processes
            # =================================================================
            # This ensures all GPUs finish the epoch before any proceeds.
            # Without this, rank 0 might start saving while others train.
            #
            # barrier() = "everyone wait here until all arrive"
            # =================================================================
            if world_size > 1:
                dist.barrier()

        # =====================================================================
        # STEP 7: Save checkpoint (ONLY RANK 0)
        # =====================================================================
        # CRITICAL: Only one process should save!
        #
        # If all processes save:
        # 1. File corruption (multiple writes to same file)
        # 2. Wasted disk I/O
        # 3. Race conditions
        #
        # Also note: We save model.module (the actual model),
        # not model (the DDP wrapper).
        # =====================================================================
        if is_main_process(rank):
            total_time = time.time() - total_start_time

            print("\n" + "=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            print(f"Total time: {format_time(total_time)}")
            print(f"Best test accuracy: {best_accuracy:.2f}%")
            print(f"Time per epoch: {format_time(total_time / args.epochs)}")

            # Save checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, "distilbert_agnews.pt")

            # Get actual model (unwrap DDP)
            actual_model = model.module if hasattr(model, 'module') else model

            save_checkpoint(
                model=actual_model,
                optimizer=optimizer,
                epoch=args.epochs,
                loss=train_loss,
                accuracy=best_accuracy,
                filepath=checkpoint_path,
                is_ddp=False  # We already unwrapped, so is_ddp=False
            )

            print(f"\nCheckpoint saved to: {checkpoint_path}")

    finally:
        # =====================================================================
        # STEP 8: Cleanup
        # =====================================================================
        # ALWAYS clean up, even if training fails!
        # try/finally ensures this runs no matter what.
        # =====================================================================
        cleanup_distributed()
        if is_main_process(rank):
            print("\nDistributed training cleanup complete.")


if __name__ == "__main__":
    main()
