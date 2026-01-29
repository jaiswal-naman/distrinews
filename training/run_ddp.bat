@echo off
REM =============================================================================
REM run_ddp.bat - Launch Distributed Training on Windows
REM =============================================================================
REM
REM USAGE:
REM   run_ddp.bat              Use 2 GPUs (default)
REM   run_ddp.bat 1            Use 1 GPU
REM   run_ddp.bat 4            Use 4 GPUs
REM
REM =============================================================================

REM Number of GPUs (default: 2)
SET NUM_GPUS=%1
IF "%NUM_GPUS%"=="" SET NUM_GPUS=2

REM Training parameters
IF "%EPOCHS%"=="" SET EPOCHS=3
IF "%BATCH_SIZE%"=="" SET BATCH_SIZE=32
IF "%LEARNING_RATE%"=="" SET LEARNING_RATE=2e-5

echo ==============================================
echo Launching DDP Training
echo ==============================================
echo GPUs:          %NUM_GPUS%
echo Epochs:        %EPOCHS%
echo Batch size:    %BATCH_SIZE% (per GPU)
echo Learning rate: %LEARNING_RATE%
echo ==============================================
echo.

REM Launch training with torchrun
torchrun ^
    --nproc_per_node=%NUM_GPUS% ^
    --standalone ^
    training/train_ddp.py ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --learning_rate %LEARNING_RATE%

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo ==============================================
    echo Training completed successfully!
    echo ==============================================
) ELSE (
    echo.
    echo ==============================================
    echo Training FAILED with exit code %ERRORLEVEL%
    echo ==============================================
)
