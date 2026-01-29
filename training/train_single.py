"""
training/train_single.py - Single-GPU Training Script

This script trains the news classifier on a single GPU (or CPU).
It's our BASELINE before adding distributed training.

=============================================================================
CONCEPT: The Training Loop
=============================================================================

Training a neural network is an iterative process:

    for epoch in range(num_epochs):           # Repeat multiple times
        for batch in dataloader:               # Process data in chunks
            predictions = model(batch)         # 1. FORWARD: Make predictions
            loss = criterion(predictions, labels)  # 2. LOSS: How wrong are we?
            loss.backward()                    # 3. BACKWARD: Compute gradients
            optimizer.step()                   # 4. UPDATE: Adjust weights
            optimizer.zero_grad()              # 5. RESET: Clear old gradients

Let's break down each step:

=============================================================================
STEP 1: FORWARD PASS
=============================================================================

The forward pass pushes data through the network:

    Input: "Sports news about football" (tokenized)
    → Embedding layer: converts tokens to vectors
    → Transformer layers: processes context
    → Classification head: produces 4 scores
    Output: [0.1, 0.8, 0.05, 0.05] (logits for each class)

The model "predicts" based on current weights.

=============================================================================
STEP 2: LOSS COMPUTATION
=============================================================================

Loss measures "how wrong" the model is.

For classification, we use Cross-Entropy Loss:
- True label: "Sports" (class 1)
- Predicted: [0.1, 0.8, 0.05, 0.05] (highest is class 1)
- Loss: -log(0.8) ≈ 0.22 (low loss = good prediction)

If model predicted [0.8, 0.1, 0.05, 0.05] (class 0):
- Loss: -log(0.1) ≈ 2.30 (high loss = bad prediction)

Lower loss = better predictions.

=============================================================================
STEP 3: BACKWARD PASS (Backpropagation)
=============================================================================

This is where the magic happens!

loss.backward() computes GRADIENTS for every weight in the network.

WHAT IS A GRADIENT?
A gradient tells us: "If I increase this weight slightly, will loss go up or down?"

- Positive gradient: Increasing weight increases loss → decrease it
- Negative gradient: Increasing weight decreases loss → increase it
- Zero gradient: Weight doesn't affect loss → don't change

PyTorch computes gradients automatically using the chain rule.
This is called "automatic differentiation" or "autograd".

=============================================================================
STEP 4: OPTIMIZER STEP
=============================================================================

The optimizer updates weights based on gradients:

    new_weight = old_weight - learning_rate * gradient

LEARNING RATE:
- Too high: Weights change too much, training unstable
- Too low: Weights change too little, training slow
- Just right: Smooth convergence to good weights

Common learning rates: 1e-5 to 1e-3 (0.00001 to 0.001)

ADAM OPTIMIZER:
Adam is smarter than basic gradient descent:
- Adapts learning rate per parameter
- Uses momentum (considers past gradients)
- Works well for transformers

=============================================================================
STEP 5: ZERO GRADIENTS
=============================================================================

PyTorch ACCUMULATES gradients by default.
If you don't zero them, gradients from previous batches add up!

    optimizer.zero_grad()  # Clear accumulated gradients

This MUST happen before each backward pass.

=============================================================================
CONCEPT: Epochs vs Batches
=============================================================================

- BATCH: A subset of data processed together (e.g., 32 samples)
- EPOCH: One complete pass through ALL training data

If you have 10,000 samples and batch_size=100:
- 1 epoch = 100 batches (10,000 / 100)
- 10 epochs = 1,000 batches total

WHY MULTIPLE EPOCHS?
One pass isn't enough. The model needs to see data multiple times
to learn patterns. Usually 3-10 epochs for fine-tuning.

=============================================================================
CONCEPT: model.train() vs model.eval()
=============================================================================

Neural networks behave differently during training vs inference:

model.train():
- Dropout is ACTIVE (randomly zeros some neurons)
- BatchNorm uses batch statistics

model.eval():
- Dropout is DISABLED (use all neurons)
- BatchNorm uses running statistics

Always set the mode before forward pass!

=============================================================================
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import NewsClassifier
from training.dataset import load_agnews, get_tokenizer, AGNewsDataset, create_dataloader
from training.utils import (
    get_device,
    AverageMeter,
    compute_accuracy,
    save_checkpoint,
    print_training_header,
    print_epoch_summary,
    format_time
)


def parse_args():
    """
    Parse command-line arguments.

    WHY COMMAND-LINE ARGUMENTS?
    --------------------------
    Instead of editing code to change settings, we pass them at runtime:
        python train_single.py --epochs 5 --batch_size 32

    This makes it easy to run experiments with different configurations.
    """
    parser = argparse.ArgumentParser(
        description="Train news classifier on single GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization"
    )

    # Data parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit training samples (for quick testing)"
    )

    # Output parameters
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )

    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int
) -> tuple:
    """
    Train for one epoch.

    PARAMETERS:
    -----------
    model : nn.Module
        The model to train

    dataloader : DataLoader
        Training data loader

    criterion : nn.Module
        Loss function (CrossEntropyLoss)

    optimizer : torch.optim.Optimizer
        Optimizer (Adam)

    device : torch.device
        Device to train on

    epoch : int
        Current epoch number (for display)

    num_epochs : int
        Total epochs (for display)

    RETURNS:
    --------
    tuple: (average_loss, average_accuracy)

    THE TRAINING LOOP EXPLAINED:
    ---------------------------
    This function implements the core training loop.
    Each step is explained inline with comments.
    """
    # =========================================================================
    # STEP 0: Set model to training mode
    # =========================================================================
    # This enables dropout and batch normalization training behavior.
    # CRITICAL: Always call model.train() before training!
    model.train()

    # =========================================================================
    # SETUP: Initialize metrics tracking
    # =========================================================================
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("accuracy")

    # Progress bar for visual feedback
    # tqdm wraps the dataloader and shows progress
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{num_epochs}",
        leave=True,
        ncols=100
    )

    # =========================================================================
    # MAIN LOOP: Process each batch
    # =========================================================================
    for batch in pbar:
        # ---------------------------------------------------------------------
        # STEP 1: Move data to device
        # ---------------------------------------------------------------------
        # Data comes from DataLoader on CPU. Move to GPU for faster processing.
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Get batch size (might be smaller for last batch)
        batch_size = input_ids.size(0)

        # ---------------------------------------------------------------------
        # STEP 2: Zero gradients from previous batch
        # ---------------------------------------------------------------------
        # IMPORTANT: Must do this BEFORE forward pass!
        # Otherwise gradients accumulate across batches.
        optimizer.zero_grad()

        # ---------------------------------------------------------------------
        # STEP 3: Forward pass
        # ---------------------------------------------------------------------
        # Model takes tokens and attention mask, returns logits.
        # logits shape: (batch_size, 4) - one score per class
        logits = model(input_ids, attention_mask)

        # ---------------------------------------------------------------------
        # STEP 4: Compute loss
        # ---------------------------------------------------------------------
        # CrossEntropyLoss compares predicted logits to true labels.
        # It handles softmax internally, so we pass raw logits.
        loss = criterion(logits, labels)

        # ---------------------------------------------------------------------
        # STEP 5: Backward pass (compute gradients)
        # ---------------------------------------------------------------------
        # This is where PyTorch's autograd magic happens.
        # It computes gradients for ALL parameters in the model.
        # Gradients are stored in param.grad for each parameter.
        loss.backward()

        # ---------------------------------------------------------------------
        # STEP 6: Optimizer step (update weights)
        # ---------------------------------------------------------------------
        # This updates all parameters using their gradients:
        #   param = param - lr * param.grad
        # Adam optimizer also considers momentum and adaptive learning rates.
        optimizer.step()

        # ---------------------------------------------------------------------
        # STEP 7: Track metrics
        # ---------------------------------------------------------------------
        # .item() converts single-element tensor to Python float
        # We detach accuracy computation from gradient tracking
        loss_meter.update(loss.item(), batch_size)
        acc = compute_accuracy(logits.detach(), labels)
        acc_meter.update(acc, batch_size)

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{acc_meter.avg:.2f}%"
        })

    return loss_meter.avg, acc_meter.avg


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    Evaluate the model on validation/test data.

    This is similar to training but:
    - No gradient computation (faster, less memory)
    - No weight updates
    - model.eval() mode

    PARAMETERS & RETURNS: Same as train_one_epoch
    """
    # =========================================================================
    # Set model to evaluation mode
    # =========================================================================
    # Disables dropout, uses running stats for batch norm
    model.eval()

    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("accuracy")

    # =========================================================================
    # torch.no_grad() context
    # =========================================================================
    # Disables gradient computation for everything inside.
    # Benefits:
    # - Faster (no need to track operations)
    # - Less memory (no gradient tensors stored)
    # Use this for inference/evaluation!
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False, ncols=100):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            batch_size = input_ids.size(0)

            # Forward pass only (no backward)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # Track metrics
            loss_meter.update(loss.item(), batch_size)
            acc = compute_accuracy(logits, labels)
            acc_meter.update(acc, batch_size)

    return loss_meter.avg, acc_meter.avg


def main():
    """
    Main training function.

    This orchestrates the entire training process:
    1. Setup (args, device, data, model)
    2. Training loop (multiple epochs)
    3. Save final model
    """
    # =========================================================================
    # SETUP
    # =========================================================================

    # Parse command-line arguments
    args = parse_args()

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    print("\nLoading data...")

    # Load tokenizer
    tokenizer = get_tokenizer()

    # Load AG News dataset
    train_data = load_agnews("train")
    test_data = load_agnews("test")

    # Limit samples for quick testing
    if args.num_samples is not None:
        train_data = train_data[:args.num_samples]
        test_data = test_data[:min(args.num_samples // 4, len(test_data))]
        print(f"Limited to {len(train_data)} train, {len(test_data)} test samples")

    # Create datasets
    train_dataset = AGNewsDataset(train_data, tokenizer, args.max_length)
    test_dataset = AGNewsDataset(test_data, tokenizer, args.max_length)

    # Create dataloaders
    # -------------------------------------------------------------------------
    # BATCH SIZE EXPLANATION:
    # - Larger batch = faster training (more parallelism)
    # - Larger batch = more memory usage
    # - Larger batch = sometimes worse generalization
    # 32 is a good default. Increase if you have more GPU memory.
    # -------------------------------------------------------------------------
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Randomize order each epoch
        num_workers=0  # 0 for debugging, 2-4 for speed
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle evaluation data
        num_workers=0
    )

    # =========================================================================
    # MODEL SETUP
    # =========================================================================
    print("\nInitializing model...")

    # Create model
    model = NewsClassifier(num_labels=4)

    # Move model to device
    # -------------------------------------------------------------------------
    # IMPORTANT: Model must be on same device as data!
    # If model is on GPU and data is on CPU, you get an error.
    # -------------------------------------------------------------------------
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # =========================================================================
    # LOSS FUNCTION
    # =========================================================================
    # CrossEntropyLoss is standard for classification:
    # - Combines softmax + negative log likelihood
    # - Input: logits (raw scores), Target: class indices
    # - Output: scalar loss value
    # -------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    # =========================================================================
    # OPTIMIZER
    # =========================================================================
    # Adam optimizer with weight decay (regularization)
    # -------------------------------------------------------------------------
    # WHY ADAM?
    # - Adapts learning rate per parameter
    # - Works well with transformers
    # - Less sensitive to learning rate choice
    #
    # WHY THIS LEARNING RATE?
    # - 2e-5 = 0.00002 is standard for fine-tuning transformers
    # - Higher (1e-4) might be unstable
    # - Lower (1e-6) would be too slow
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01  # L2 regularization to prevent overfitting
    )

    # =========================================================================
    # PRINT CONFIGURATION
    # =========================================================================
    print_training_header(
        model_name="DistilBERT",
        dataset_name="AG News",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        num_train_samples=len(train_dataset),
        num_gpus=1
    )

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("Starting training...")

    best_accuracy = 0.0
    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=args.epochs
        )

        # Evaluate on test set
        test_loss, test_acc = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device
        )

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print summary
        print_epoch_summary(
            epoch=epoch,
            num_epochs=args.epochs,
            train_loss=train_loss,
            train_acc=train_acc,
            epoch_time=epoch_time,
            val_loss=test_loss,
            val_acc=test_acc
        )

        # Track best accuracy
        if test_acc > best_accuracy:
            best_accuracy = test_acc

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    total_time = time.time() - total_start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {format_time(total_time)}")
    print(f"Best test accuracy: {best_accuracy:.2f}%")

    # =========================================================================
    # SAVE CHECKPOINT
    # =========================================================================
    checkpoint_path = os.path.join(args.checkpoint_dir, "distilbert_agnews.pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        loss=train_loss,
        accuracy=best_accuracy,
        filepath=checkpoint_path,
        is_ddp=False
    )

    print(f"\nModel saved to: {checkpoint_path}")
    print("You can now use this checkpoint for inference or distributed training.")


if __name__ == "__main__":
    main()
