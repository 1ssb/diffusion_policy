#!/bin/bash

# Training script for PushT Diffusion Policy
# Runs with proper logging and timestamp-based naming
# Runs in background with nohup so you can log off

cd /home/rudra/projects/imitation_games/diffusion_policy

# Create timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="training_${TIMESTAMP}.log"

echo "Starting training at $(date)"
echo "Log file: $LOGFILE"
echo "You can monitor with: tail -f $LOGFILE"
echo ""

nohup conda run -n robodiff python train.py \
  --config-dir=. \
  --config-name=image_pusht_diffusion_policy_cnn.yaml \
  training.seed=42 \
  training.device=cuda:0 \
  dataloader.num_workers=0 \
  dataloader.pin_memory=false \
  dataloader.persistent_workers=false \
  val_dataloader.num_workers=0 \
  val_dataloader.pin_memory=false \
  val_dataloader.persistent_workers=false \
  logging.name="\${now:%Y%m%d_%H%M%S}_pusht_diffusion" \
  hydra.run.dir="data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}" \
  > "$LOGFILE" 2>&1 &

# Save PID
echo $! > training.pid
echo "Training started with PID: $(cat training.pid)"
echo "To stop training: kill \$(cat training.pid)"
