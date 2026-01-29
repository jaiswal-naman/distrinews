"""
training/utils.py - Training Utilities

Helper functions for training: metrics, checkpointing, logging.

=============================================================================
WHY SEPARATE UTILITIES?
=============================================================================

Good software engineering separates concerns:
- train_single.py: Training logic (what to train)
- train_ddp.py: Distributed logic (how to distribute)
- utils.py: Common utilities (shared by both)

This way, we don't duplicate code between single-GPU and DDP training.

=============================================================================
"""

import os
import time
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


def get_device(local_rank: Optional[int] = None) -> torch.device:
    """
    Get the best available device for training.

    PARAMETERS:
    -----------
    local_rank : int, optional
        For distributed training, the GPU index to use.
        If None, auto-detect best available device.

    RETURNS:
    --------
    torch.device: The device to use (cuda:X or cpu)

    HOW THIS WORKS:
    ---------------
    Priority order:
    1. If local_rank provided → use that specific GPU
    2. If CUDA available → use GPU 0
    3. Otherwise → use CPU

    WHY CHECK CUDA?
    ---------------
    CUDA = NVIDIA's GPU computing platform.
    torch.cuda.is_available() returns True if:
    - You have an NVIDIA GPU
    - CUDA drivers are installed
    - PyTorch was built with CUDA support
    """
    if local_rank is not None:
        # Distributed training: use specified GPU
        return torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        # Single-GPU training: use first GPU
        return torch.device("cuda:0")
    else:
        # No GPU: use CPU
        return torch.device("cpu")


class AverageMeter:
    """
    Computes and stores the average and current value.

    WHY THIS CLASS?
    ---------------
    During training, we compute loss for each batch.
    We want to track the AVERAGE loss across all batches in an epoch.

    Example:
        meter = AverageMeter("loss")
        for batch in batches:
            loss = compute_loss(batch)
            meter.update(loss.item(), batch_size)
        print(f"Average loss: {meter.avg}")

    WHAT IT TRACKS:
    ---------------
    - val: Most recent value
    - sum: Sum of all values (weighted by count)
    - count: Total number of samples
    - avg: Average value (sum / count)

    WHY WEIGHTED AVERAGE?
    --------------------
    Batches might have different sizes (especially the last batch).
    If batch 1 has 32 samples and batch 2 has 16 samples:
        Simple average: (loss1 + loss2) / 2  ← WRONG
        Weighted average: (loss1*32 + loss2*16) / 48  ← CORRECT
    """

    def __init__(self, name: str):
        """
        Initialize the meter.

        PARAMETERS:
        -----------
        name : str
            Name of the metric (for display purposes)
        """
        self.name = name
        self.reset()

    def reset(self):
        """Reset all statistics to zero."""
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val: float, n: int = 1):
        """
        Update with a new value.

        PARAMETERS:
        -----------
        val : float
            The new value to add
        n : int
            Weight/count for this value (usually batch_size)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    PARAMETERS:
    -----------
    logits : torch.Tensor
        Model predictions (raw scores), shape (batch_size, num_classes)

    labels : torch.Tensor
        Ground truth labels, shape (batch_size,)

    RETURNS:
    --------
    float: Accuracy as a percentage (0-100)

    HOW IT WORKS:
    ------------
    1. Find predicted class: argmax of logits
    2. Compare to true labels
    3. Count correct / total

    EXAMPLE:
    --------
    logits = [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]  # 2 samples, 3 classes
    labels = [2, 0]  # True classes

    predictions = [2, 0]  # argmax of each row
    correct = [True, True]  # predictions == labels
    accuracy = 2/2 = 100%

    WHY .detach()?
    -------------
    Detach removes the tensor from the computation graph.
    We don't need gradients for accuracy — it's just a metric.
    """
    # Get predicted class (highest logit)
    predictions = torch.argmax(logits, dim=1)

    # Compare to true labels
    correct = (predictions == labels).sum().item()
    total = labels.size(0)

    # Return percentage
    return (correct / total) * 100


def format_time(seconds: float) -> str:
    """
    Format seconds into a readable string.

    PARAMETERS:
    -----------
    seconds : float
        Time in seconds

    RETURNS:
    --------
    str: Formatted time string

    EXAMPLES:
    ---------
    format_time(65) → "1m 5s"
    format_time(3665) → "1h 1m 5s"
    format_time(45.5) → "45s"
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    filepath: str,
    is_ddp: bool = False
) -> None:
    """
    Save a training checkpoint.

    PARAMETERS:
    -----------
    model : nn.Module
        The model to save

    optimizer : torch.optim.Optimizer
        The optimizer (to resume training)

    epoch : int
        Current epoch number

    loss : float
        Current loss value

    accuracy : float
        Current accuracy

    filepath : str
        Where to save the checkpoint

    is_ddp : bool
        If True, model is wrapped with DDP, need to save model.module

    WHAT WE SAVE:
    ------------
    - model_state_dict: All model weights
    - optimizer_state_dict: Optimizer state (momentum, etc.)
    - epoch: To resume training
    - loss: For reference
    - accuracy: For reference

    WHY SAVE OPTIMIZER STATE?
    ------------------------
    Optimizers like Adam track momentum for each parameter.
    Without saving this, resuming training would be like starting fresh.

    WHY model.module FOR DDP?
    ------------------------
    When model is wrapped with DistributedDataParallel:
        model = DDP(model)

    The actual model is at model.module.
    We save model.module.state_dict() to get clean weights
    that can be loaded without DDP wrapper.
    """
    # Get the actual model (handle DDP wrapper)
    model_to_save = model.module if is_ddp else model

    # Create checkpoint directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Build checkpoint dict
    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy,
    }

    # Save
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu")
) -> Dict:
    """
    Load a training checkpoint.

    PARAMETERS:
    -----------
    filepath : str
        Path to checkpoint file

    model : nn.Module
        Model to load weights into

    optimizer : torch.optim.Optimizer, optional
        Optimizer to load state into (for resuming training)

    device : torch.device
        Device to load tensors to

    RETURNS:
    --------
    dict: Checkpoint metadata (epoch, loss, accuracy)

    WHY map_location?
    ----------------
    If checkpoint was saved on GPU but you're loading on CPU:
        torch.load("checkpoint.pt")  # Error!
        torch.load("checkpoint.pt", map_location="cpu")  # Works!

    map_location tells PyTorch where to put the loaded tensors.
    """
    print(f"Loading checkpoint: {filepath}")

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0),
        "accuracy": checkpoint.get("accuracy", 0),
    }


def print_training_header(
    model_name: str,
    dataset_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    num_train_samples: int,
    num_gpus: int = 1
) -> None:
    """
    Print a nice header at the start of training.

    This gives a clear summary of the training configuration.
    """
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model:          {model_name}")
    print(f"Dataset:        {dataset_name}")
    print(f"Train samples:  {num_train_samples:,}")
    print(f"Epochs:         {num_epochs}")
    print(f"Batch size:     {batch_size}")
    print(f"Learning rate:  {learning_rate}")
    print(f"Device:         {device}")
    print(f"GPUs:           {num_gpus}")
    print("=" * 60 + "\n")


def print_epoch_summary(
    epoch: int,
    num_epochs: int,
    train_loss: float,
    train_acc: float,
    epoch_time: float,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None
) -> None:
    """
    Print a summary at the end of each epoch.
    """
    time_str = format_time(epoch_time)

    print(f"\nEpoch [{epoch}/{num_epochs}] - {time_str}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

    if val_loss is not None and val_acc is not None:
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")


# =============================================================================
# TESTING CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Training Utilities")
    print("=" * 60)

    # Test device detection
    print("\n1. Testing device detection...")
    device = get_device()
    print(f"   Detected device: {device}")

    # Test AverageMeter
    print("\n2. Testing AverageMeter...")
    meter = AverageMeter("test_loss")
    meter.update(1.0, n=10)  # 10 samples with loss 1.0
    meter.update(2.0, n=20)  # 20 samples with loss 2.0
    # Expected average: (1.0*10 + 2.0*20) / 30 = 50/30 = 1.667
    print(f"   Average: {meter.avg:.3f} (expected: 1.667)")
    assert abs(meter.avg - 1.667) < 0.01, "AverageMeter incorrect"
    print("   ✓ AverageMeter works!")

    # Test accuracy computation
    print("\n3. Testing compute_accuracy...")
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])  # predictions: 1, 0, 1
    labels = torch.tensor([1, 0, 0])  # true: 1, 0, 0
    acc = compute_accuracy(logits, labels)
    # predictions [1, 0, 1] vs labels [1, 0, 0] → 2/3 correct = 66.67%
    print(f"   Accuracy: {acc:.2f}% (expected: 66.67%)")
    assert abs(acc - 66.67) < 0.1, "compute_accuracy incorrect"
    print("   ✓ compute_accuracy works!")

    # Test time formatting
    print("\n4. Testing format_time...")
    print(f"   45 seconds → {format_time(45)}")
    print(f"   125 seconds → {format_time(125)}")
    print(f"   3725 seconds → {format_time(3725)}")
    print("   ✓ format_time works!")

    # Test checkpoint save/load
    print("\n5. Testing checkpoint save/load...")
    import tempfile
    from model import NewsClassifier

    # Create a simple model
    model = NewsClassifier()
    optimizer = torch.optim.Adam(model.parameters())

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_checkpoint.pt")
        save_checkpoint(model, optimizer, epoch=5, loss=0.5, accuracy=85.0, filepath=filepath)

        # Load checkpoint
        model2 = NewsClassifier()
        optimizer2 = torch.optim.Adam(model2.parameters())
        info = load_checkpoint(filepath, model2, optimizer2)

        print(f"   Loaded epoch: {info['epoch']}")
        print(f"   Loaded accuracy: {info['accuracy']}%")
        assert info["epoch"] == 5, "Checkpoint epoch incorrect"
        assert info["accuracy"] == 85.0, "Checkpoint accuracy incorrect"
    print("   ✓ Checkpoint save/load works!")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
