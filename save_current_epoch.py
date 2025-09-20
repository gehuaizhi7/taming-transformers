#!/usr/bin/env python3
"""
Save current training state manually
"""

import torch
import os
import glob
from omegaconf import OmegaConf

def save_current_state():
    """Save the current training state"""
    print("Saving current training state...")
    
    # Find the most recent checkpoint directory
    checkpoint_dirs = glob.glob("checkpoints_variational*")
    if not checkpoint_dirs:
        print("No checkpoint directory found. Creating one...")
        os.makedirs("checkpoints_variational", exist_ok=True)
        checkpoint_dir = "checkpoints_variational"
    else:
        checkpoint_dir = checkpoint_dirs[0]
    
    print(f"Using checkpoint directory: {checkpoint_dir}")
    
    # Try to load the model and save current state
    try:
        # Load config
        config = OmegaConf.load("configs/unconditional_transformer.yaml")
        
        # Import and create model
        from taming.models.cond_transformer import Net2NetTransformer
        model = Net2NetTransformer(**config.model.params)
        
        # Load any existing checkpoint
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            print(f"Loading latest checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 1)
        else:
            print("No existing checkpoint found. Using initial model state.")
            epoch = 1
        
        # Save current state as epoch 1
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': 0.0,  # Placeholder
            'val_loss': 0.0,  # Placeholder
            'eta_fast': 1e-3,
            'eta_slow': 1e-4,
            'T_fast': 5
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'manual_epoch_{epoch}.pt')
        torch.save(checkpoint_data, checkpoint_path, _use_new_zipfile_serialization=True)
        
        print(f"✅ Saved current state to: {checkpoint_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving state: {e}")
        return False

def main():
    print("Manual Checkpoint Saver")
    print("=" * 30)
    
    if save_current_state():
        print("\n✅ Current training state saved!")
        print("You can now:")
        print("1. Continue training and it will save at epoch 5")
        print("2. Or stop and restart with --save_every 1")
    else:
        print("\n❌ Failed to save current state")

if __name__ == "__main__":
    main()






