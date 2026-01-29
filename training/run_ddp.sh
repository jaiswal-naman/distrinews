#!/bin/bash
# =============================================================================
# run_ddp.sh - Launch Distributed Training with torchrun
# =============================================================================
#
# USAGE:
#   ./run_ddp.sh              # Use 2 GPUs (default)
#   ./run_ddp.sh 1            # Use 1 GPU
#   ./run_ddp.sh 4            # Use 4 GPUs
#
# =============================================================================
# WHAT IS torchrun?
# =============================================================================
#
# torchrun is PyTorch's official distributed launcher. It:
#
# 1. SPAWNS PROCESSES
#    - Runs your script N times (once per GPU)
#    - Each runs as a separate Python process
#
# 2. SETS ENVIRONMENT VARIABLES
#    For each process, torchrun sets:
#    - RANK: Global process ID (0, 1, 2, ...)
#    - LOCAL_RANK: GPU ID on this machine (0, 1, ...)
#    - WORLD_SIZE: Total number of processes
#    - MASTER_ADDR: IP of rank 0 (localhost for single machine)
#    - MASTER_PORT: Port for communication
#
# 3. HANDLES FAILURES
#    - If one process crashes, torchrun kills others
#    - Clean shutdown instead of hanging processes
#
# =============================================================================
# WHY NOT JUST RUN PYTHON TWICE?
# =============================================================================
#
# You COULD manually start processes:
#   RANK=0 LOCAL_RANK=0 WORLD_SIZE=2 python train_ddp.py &
#   RANK=1 LOCAL_RANK=1 WORLD_SIZE=2 python train_ddp.py &
#
# But this is error-prone:
# - Easy to misconfigure environment variables
# - Hard to handle failures
# - Doesn't scale to multiple machines
#
# torchrun handles all of this automatically.
#
# =============================================================================

# Number of GPUs (default: 2)
NUM_GPUS=${1:-2}

# Training parameters (can be overridden)
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-2e-5}

echo "=============================================="
echo "Launching DDP Training"
echo "=============================================="
echo "GPUs:          $NUM_GPUS"
echo "Epochs:        $EPOCHS"
echo "Batch size:    $BATCH_SIZE (per GPU)"
echo "Effective BS:  $((BATCH_SIZE * NUM_GPUS))"
echo "Learning rate: $LEARNING_RATE"
echo "=============================================="
echo ""

# =============================================================================
# LAUNCH TRAINING
# =============================================================================
#
# torchrun arguments:
#   --nproc_per_node=N    Number of processes (GPUs) per machine
#   --standalone          Single machine mode (no multi-node setup)
#   --nnodes=1            Number of machines (1 for single machine)
#
# Your script receives these as environment variables,
# NOT as command-line arguments.
# =============================================================================

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --standalone \
    training/train_ddp.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE

# =============================================================================
# EXIT CODE
# =============================================================================
# torchrun returns non-zero if any process failed.
# This lets you chain commands: ./run_ddp.sh && echo "Success!"
# =============================================================================

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Training completed successfully!"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "Training FAILED with exit code $exit_code"
    echo "=============================================="
fi

exit $exit_code
