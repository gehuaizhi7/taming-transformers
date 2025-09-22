#!/usr/bin/env python3
"""
Variational training script for Taming Transformers
Implements the fast/slow learning algorithm for inference and generation
"""

import os
import argparse

# Set GPU IDs BEFORE importing torch
if '--gpu_ids' in os.sys.argv:
    gpu_idx = os.sys.argv.index('--gpu_ids')
    if gpu_idx + 1 < len(os.sys.argv):
        gpu_ids = os.sys.argv[gpu_idx + 1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"Using GPUs: {gpu_ids}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import glob
import shutil

from taming.models.cond_transformer import Net2NetTransformer
from taming.data.utils import custom_collate

class VariationalInference:
    """Fast-learning variational params for one-z-per-layer: (B, L, z_dim)."""
    def __init__(self, L, z_dim, device, dtype=torch.float32, logvar_clip=(-20.0, 20.0)):
        self.L = L
        self.z_dim = z_dim
        self.device = device
        self.dtype = dtype
        self.logvar_clip = logvar_clip

    def initialize_params(self, batch_size):
        """mu, log_var as leaf tensors (optimized by Adam in the fast loop)."""
        # Initialize with small random values to encourage exploration
        mu = torch.randn(batch_size, self.L, self.z_dim, device=self.device, dtype=self.dtype) * 0.1
        mu.requires_grad_(True)
        
        log_var = torch.ones(batch_size, self.L, self.z_dim, device=self.device, dtype=self.dtype) * 0.1
        log_var.requires_grad_(True)
        
        return mu, log_var

    def sample_z(self, mu, log_var, detach=False):
        """Reparameterized sample from q; optionally detach grads (for slow step)."""
        if self.logvar_clip is not None:
            log_var = torch.clamp(log_var, *self.logvar_clip)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z.detach() if detach else z

    def kl_divergence(self, mu, log_var, reduction='mean'):
        """
        KL(q||p) with p=N(0,I). reduction in {'mean','sum','none'}:
          - 'none': returns (B,) per sample
          - 'sum' : sum over batch
          - 'mean': mean over batch (default)
        """
        if self.logvar_clip is not None:
            log_var = torch.clamp(log_var, *self.logvar_clip)
        # (B, L, z_dim)
        kl_elem = 0.5 * (torch.exp(log_var) + mu**2 - 1.0 - log_var)
        # sum over z_dim and layers -> (B,)
        kl_per_sample = kl_elem.sum(dim=(1, 2))
        if reduction == 'none':
            return kl_per_sample
        elif reduction == 'sum':
            return kl_per_sample.sum()
        else:
            return kl_per_sample.mean()



def load_model_from_config(config_path, ckpt_path=None, is_resume_checkpoint=False):
    """Load model from config file"""
    config = OmegaConf.load(config_path)
    model_config = config.model
    
    # Instantiate the model
    model = Net2NetTransformer(**model_config.params)
    
    if ckpt_path and not is_resume_checkpoint:
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    return model


def load_data_from_config(config_path, batch_size=None):
    """Load data from config file"""
    config = OmegaConf.load(config_path)
    data_config = config.data
    
    if batch_size:
        data_config.params.batch_size = batch_size
    
    # Instantiate data module
    from main import instantiate_from_config
    data_module = instantiate_from_config(data_config)
    data_module.prepare_data()
    data_module.setup()
    
    return data_module


def train_batch_variational(model, batch, variational, eta_fast, T_fast, generator_optimizer, device, beta=0.1):
    """Train a single batch using the variational algorithm"""
    model.train()
    # ---- 0) Data & tokens (do ONCE per batch) ----
    x = batch['image'].to(device)
    with torch.no_grad():
        _, z_indices = model.encode_to_z(x)   # (B, S)
        cond_tokens = None
        if hasattr(model, "encode_to_c"):
            _, cond_tokens = model.encode_to_c(x)

    # For unconditional training, use full sequence
    y_in  = z_indices  # full sequence as input
    y_tgt = z_indices  # full sequence as target
    
    # Teacher forcing alignment: logits at position t predict token at position t+1
    # logits shape: (B, S, V), target should be (B, S) starting from position 1
    y_tgt_shifted = y_tgt[:, 1:]  # Remove first token: [token_1, token_2, ..., token_S]
    B = y_in.size(0)

    # ---- 1) Fast variables: mu, log_var for L layers ----
    num_layers = model.transformer.L
    mu, log_var = variational.initialize_params(batch_size=B)
    optimizer_fast = torch.optim.Adam([mu, log_var], lr=eta_fast)

    # ---- 2) Fast loop: optimize (mu, log_var) with ELBO ----
    for t in range(T_fast):
        optimizer_fast.zero_grad()

        z_per_layer = variational.sample_z(mu, log_var)             # (B, L, z_dim)
        # For unconditional training, pass the original image and the same image as conditioning
        logits, _ = model(x, c=x, z=z_per_layer)  # (B, S, V)
        
        # Teacher forcing: remove last logit position to align with shifted target
        logits = logits[:, :-1, :]  # (B, S-1, V) - remove last position

        V = logits.size(-1)
        recon_loss = F.cross_entropy(logits.reshape(-1, V), y_tgt_shifted.reshape(-1))
        kl_loss = variational.kl_divergence(mu, log_var)
        
        # Add KL weight (β) to balance reconstruction and KL loss
        loss = recon_loss + beta * kl_loss   # β-VAE ELBO (negated)

        loss.backward()
        optimizer_fast.step()

        if (t % 2) == 0:
            print(f"fast {t+1}/{T_fast} | total {loss.item():.4f}  recon {recon_loss.item():.4f}  KL {kl_loss.item():.4f} (β={beta})")

    # ---- 3) Slow step: optimize model beta with recon only ----
    generator_optimizer.zero_grad()

    # Sample z_final with detached gradients (no gradients for mu, log_var)
    with torch.no_grad():
        mu_d, log_var_d = mu.detach(), log_var.detach()
        z_final = variational.sample_z(mu_d, log_var_d)

    # Model forward pass WITHOUT no_grad() - we need gradients for the generator!
    logits, _ = model(x, c=x, z=z_final)
    
    # Teacher forcing: remove last logit position to align with shifted target
    logits = logits[:, :-1, :]  # (B, S-1, V) - remove last position
    
    V = logits.size(-1)
    generator_loss = F.cross_entropy(logits.reshape(-1, V), y_tgt_shifted.reshape(-1))
    generator_loss.backward()
    generator_optimizer.step()

    return (
        float(loss.item()),          # total ELBO at end of fast loop
        float(recon_loss.item()),    # recon at end of fast loop
        float(kl_loss.item()),       # KL at end of fast loop
        float(generator_loss.item()) # slow-step recon loss
    )


def train_epoch(model, dataloader, variational, eta_fast, T_fast, generator_optimizer, device, epoch, beta=0.1,
                kl_anneal_mode: str = "linear", kl_anneal_steps: int = 20000, global_step_start: int = 0):
    """Train for one epoch using variational algorithm"""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_generator = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Compute annealed beta based on global step
        global_step = global_step_start + batch_idx
        if kl_anneal_mode == "linear" and kl_anneal_steps > 0:
            beta_eff = min(beta * (global_step / kl_anneal_steps), beta)
        elif kl_anneal_mode == "sigmoid" and kl_anneal_steps > 0:
            import math
            progress = min(max(global_step / kl_anneal_steps, 0.0), 1.0)
            # smooth sigmoid from 0->beta
            beta_eff = beta * (1.0 / (1.0 + math.exp(-12.0 * (progress - 0.5))))
        else:
            beta_eff = beta
        print(f"Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}")
        
        # Move data to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Train batch with variational algorithm
        batch_loss, recon_loss, kl_loss, generator_loss = train_batch_variational(
            model, batch, variational, eta_fast, T_fast, generator_optimizer, device, beta_eff
        )
        
        total_loss += batch_loss
        total_recon += recon_loss
        total_kl += kl_loss
        total_generator += generator_loss
        num_batches += 1
        
        print(f"  Batch Loss: {batch_loss:.4f}, Recon: {recon_loss:.4f}, KL: {kl_loss:.4f}, Generator: {generator_loss:.4f} (β_eff={beta_eff:.4f})")
    
    return (total_loss / num_batches, total_recon / num_batches, 
            total_kl / num_batches, total_generator / num_batches)


def validate_epoch(model, dataloader, device, epoch):
    """Validate for one epoch (simplified for variational training)"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device and apply format conversion
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Apply the same data format conversion as in training
            x = batch['image']
            # Debug: print tensor shape to understand the issue
            print(f"Validation batch shape: {x.shape}")
            
            # The data should already be in CHW format from the data loader
            # But let's ensure it's correct: (B, C, H, W) where C=3
            if x.dim() == 4:
                if x.size(1) == 3:  # Already in CHW format
                    pass  # No conversion needed
                elif x.size(-1) == 3:  # HWC format
                    x = x.permute(0, 3, 1, 2)  # Convert to CHW
                    batch['image'] = x
                else:
                    print(f"Unexpected tensor shape: {x.shape}")
                    # Handle the specific case where we get [B, H, C, W] format
                    if x.size(2) == 3:  # Channels are in position 2
                        x = x.permute(0, 2, 1, 3)  # Convert [B, H, C, W] to [B, C, H, W]
                        batch['image'] = x
                        print(f"Converted to CHW format: {x.shape}")
                    else:
                        print(f"Cannot convert unexpected shape: {x.shape}")
                        continue  # Skip this batch
            
            # Double-check the final format
            print(f"Final validation batch shape: {batch['image'].shape}")
            
            # Simple validation - call model directly to avoid get_input conversion
            x = batch['image'].to(device)
            c = batch['image'].to(device)  # For unconditional training, c = x
            
            # Call model forward directly
            logits, target = model(x, c)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def cleanup_old_checkpoints(save_dir, keep_last_n=3):
    """Clean up old checkpoint files, keeping only the most recent N checkpoints"""
    try:
        # Get all checkpoint files
        checkpoint_files = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*.pt"))
        best_model_files = glob.glob(os.path.join(save_dir, "best_model_epoch_*.pt"))
        
        # Sort by modification time (newest first)
        all_files = checkpoint_files + best_model_files
        all_files.sort(key=os.path.getmtime, reverse=True)
        
        # Keep only the most recent N files
        files_to_delete = all_files[keep_last_n:]
        
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted old checkpoint: {os.path.basename(file_path)}")
            except OSError as e:
                print(f"Warning: Could not delete {file_path}: {e}")
        
        if files_to_delete:
            print(f"Cleaned up {len(files_to_delete)} old checkpoint files")
            
    except Exception as e:
        print(f"Warning: Error during checkpoint cleanup: {e}")


def get_checkpoint_size_mb(file_path):
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except OSError:
        return 0


def save_optimized_checkpoint(checkpoint_data, save_dir, epoch, is_best=False, max_checkpoints=3, compress=True):
    """Save checkpoint with automatic cleanup of old files"""
    if is_best:
        checkpoint_path = os.path.join(save_dir, f'best_model_epoch_{epoch}.pt')
    else:
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    
    # Save the checkpoint with compression if requested
    if compress:
        torch.save(checkpoint_data, checkpoint_path, _use_new_zipfile_serialization=True)
    else:
        torch.save(checkpoint_data, checkpoint_path)
    
    # Get file size
    file_size_mb = get_checkpoint_size_mb(checkpoint_path)
    print(f"Saved checkpoint: {os.path.basename(checkpoint_path)} ({file_size_mb:.1f} MB)")
    
    # Clean up old checkpoints
    cleanup_old_checkpoints(save_dir, keep_last_n=max_checkpoints)
    
    return checkpoint_path


def create_lightweight_checkpoint(model, optimizer, epoch, loss, val_loss, eta_fast, eta_slow, T_fast):
    """Create a lightweight checkpoint with only essential data"""
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_loss': val_loss,
        'eta_fast': eta_fast,
        'eta_slow': eta_slow,
        'T_fast': T_fast
    }


def show_disk_usage(save_dir):
    """Show disk usage for checkpoint directory"""
    try:
        total_size = 0
        checkpoint_files = glob.glob(os.path.join(save_dir, "*.pt"))
        
        for file_path in checkpoint_files:
            total_size += get_checkpoint_size_mb(file_path)
        
        print(f"Checkpoint directory usage: {total_size:.1f} MB ({len(checkpoint_files)} files)")
        
        # Show individual file sizes
        if checkpoint_files:
            print("Current checkpoints:")
            for file_path in sorted(checkpoint_files, key=os.path.getmtime, reverse=True):
                size_mb = get_checkpoint_size_mb(file_path)
                filename = os.path.basename(file_path)
                print(f"  {filename}: {size_mb:.1f} MB")
                
    except Exception as e:
        print(f"Warning: Could not calculate disk usage: {e}")


def main():
    parser = argparse.ArgumentParser(description='Variational Training Script')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--eta_fast', type=float, default=1e-3, help='Fast learning rate for inference')
    parser.add_argument('--eta_slow', type=float, default=1e-4, help='Slow learning rate for generator')
    parser.add_argument('--T_fast', type=int, default=5, help='Number of fast learning steps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated GPU IDs (e.g., 5,6,7)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_variational', help='Directory to save checkpoints')
    parser.add_argument('--max_checkpoints', type=int, default=3, help='Maximum number of checkpoints to keep (default: 3)')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--save_best_only', action='store_true', help='Only save best model checkpoints (saves disk space)')
    parser.add_argument('--compress_checkpoints', action='store_true', default=True, help='Use compression for checkpoints (default: True)')
    parser.add_argument('--no_compress', action='store_true', help='Disable checkpoint compression')
    parser.add_argument('--beta', type=float, default=0.1, help='KL loss weight (β) for β-VAE (default: 0.1)')
    parser.add_argument('--kl_anneal_mode', type=str, default='linear', choices=['linear','sigmoid','none','cyclical'], help='KL annealing schedule')
    parser.add_argument('--kl_anneal_steps', type=int, default=20000, help='Steps to reach target β during annealing')
    parser.add_argument('--kl_cycle_steps', type=int, default=0, help='If >0 and mode=cyclical, cycle length for β annealing')
    
    args = parser.parse_args()
    
    # Load config first to get hyperparameters
    config = OmegaConf.load(args.config)
    
    # Get all variational parameters from config, with command line override
    eta_fast = config.variational.get('eta_fast', 1e-3)
    eta_slow = config.variational.get('eta_slow', 1e-4) 
    T_fast = config.variational.get('T_fast', 5)
    beta = config.variational.get('beta', 0.1)
    
    # Command line arguments override config (only if explicitly provided)
    # Check if arguments were explicitly provided by looking at sys.argv
    import sys
    if '--eta_fast' in sys.argv:
        eta_fast = args.eta_fast
    if '--eta_slow' in sys.argv:
        eta_slow = args.eta_slow
    if '--T_fast' in sys.argv:
        T_fast = args.T_fast
    if '--beta' in sys.argv:
        beta = args.beta
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Variational Training Algorithm")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"η_fast (fast learning rate): {eta_fast}")
    print(f"η_slow (slow learning rate): {eta_slow}")
    print(f"T_fast (fast learning steps): {T_fast}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model_from_config(args.config, is_resume_checkpoint=bool(args.ckpt))
    model = model.to(args.device)
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Load data
    print("Loading data...")
    data_module = load_data_from_config(args.config, args.batch_size)
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    
    # Get model parameters for variational inference
    L = model.transformer.L  # Number of z vectors (L)
    
    # Get z_dim from config
    z_dim = config.model.params.transformer_config.params.z_dim
    
    # Initialize variational inference
    print("Initializing variational inference...")
    variational = VariationalInference(L, z_dim, args.device)
    
    # Setup generator optimizer (slow learning)
    print("Setting up generator optimizer...")
    generator_params = []
    for name, param in model.named_parameters():
        if 'transformer' in name:
            generator_params.append(param)
    
    generator_optimizer = optim.AdamW(generator_params, lr=eta_slow)
    
    print(f"Model L (layers): {L}")
    print(f"Z dimension: {z_dim}")
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")
    print(f"KL weight (β): {beta}")
    
    # Load checkpoint if provided
    start_epoch = 0
    best_loss = float('inf')
    
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"Loading checkpoint from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=args.device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        
        print(f"Resumed from epoch {start_epoch} with loss {best_loss:.4f}")
        print(f"Previous training parameters: η_fast={checkpoint['eta_fast']}, η_slow={checkpoint['eta_slow']}, T_fast={checkpoint['T_fast']}")
    
    # Training loop
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train epoch
        # Handle cyclical annealing option by mapping global_step into cycles
        anneal_mode = args.kl_anneal_mode
        if anneal_mode == 'none':
            anneal_mode_eff = 'none'
            kl_steps_eff = args.kl_anneal_steps
            global_step_for_epoch = global_step
        elif anneal_mode == 'cyclical' and args.kl_cycle_steps > 0:
            # Map position inside cycle and reuse linear schedule over the cycle
            cycle_pos = global_step % args.kl_cycle_steps
            anneal_mode_eff = 'linear'
            kl_steps_eff = max(1, min(args.kl_anneal_steps, args.kl_cycle_steps))
            global_step_for_epoch = cycle_pos
        else:
            anneal_mode_eff = anneal_mode
            kl_steps_eff = args.kl_anneal_steps
            global_step_for_epoch = global_step

        avg_loss, avg_recon, avg_kl, avg_generator = train_epoch(
            model, train_dataloader, variational, eta_fast, T_fast,
            generator_optimizer, args.device, epoch+1, beta,
            kl_anneal_mode=anneal_mode_eff, kl_anneal_steps=kl_steps_eff, global_step_start=global_step_for_epoch
        )

        global_step += len(train_dataloader)
        
        # Validate
        val_loss = validate_epoch(model, val_dataloader, args.device, epoch+1)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Reconstruction: {avg_recon:.4f}")
        print(f"  Average KL: {avg_kl:.4f}")
        print(f"  Average Generator: {avg_generator:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        
        # Show disk usage
        show_disk_usage(args.save_dir)
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Prepare lightweight checkpoint data
        checkpoint_data = create_lightweight_checkpoint(
            model, generator_optimizer, epoch+1, avg_loss, val_loss,
            args.eta_fast, args.eta_slow, args.T_fast
        )
        
        # Determine compression setting
        use_compression = args.compress_checkpoints and not args.no_compress
        
        # Save best model if improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_optimized_checkpoint(
                checkpoint_data, args.save_dir, epoch+1, 
                is_best=True, max_checkpoints=args.max_checkpoints, compress=use_compression
            )
        
        # Save regular checkpoint based on frequency and settings
        should_save_regular = False
        if args.save_best_only:
            # Only save if it's the best model (already handled above)
            pass
        else:
            # Save every N epochs
            should_save_regular = (epoch + 1) % args.save_every == 0
        
        if should_save_regular:
            save_optimized_checkpoint(
                checkpoint_data, args.save_dir, epoch+1, 
                is_best=False, max_checkpoints=args.max_checkpoints, compress=use_compression
            )


if __name__ == '__main__':
    main()
