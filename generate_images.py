#!/usr/bin/env python3
"""
Image generation script for CustomGPT model
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from main import instantiate_from_config
from PIL import Image
import os

def load_model_from_config(config_path, ckpt_path=None):
    """Load model from config and checkpoint"""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    
    if ckpt_path:
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Checkpoint loaded successfully")
    
    return model

def sample_z(model, device, z_dim, L):
    """Sample random latent vectors z"""
    # Sample z for each layer (L vectors total)
    z = torch.randn(1, L, z_dim, device=device)  # Shape: (1, L, z_dim)
    return z

def generate_tokens(model, z, max_length=256, temperature=1.0, top_k=None):
    """Generate token sequence using the model"""
    model.eval()
    
    with torch.no_grad():
        # Start with SOS token (0)
        sequence = torch.tensor([[0]], device=z.device)  # Shape: (1, 1)
        
        for _ in range(max_length):
            # Forward pass through transformer only
            logits, _ = model.transformer(sequence, z=z)
            
            # Get logits for the last position
            logits = logits[:, -1, :] / temperature  # Shape: (1, vocab_size)
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            sequence = torch.cat([sequence, next_token], dim=1)
    
    return sequence

def tokens_to_image(model, tokens):
    """Convert tokens to image using VQGAN decoder"""
    model.eval()
    
    with torch.no_grad():
        # Remove SOS token if present
        if tokens[0, 0] == 0:  # SOS token
            tokens = tokens[:, 1:]
        
        # Get the shape for decoding (we need to know the spatial dimensions)
        # For 256x256 images, the tokens should be 16x16 = 256 tokens
        # But we need to reshape them properly
        seq_len = tokens.shape[1]
        
        # Assume square images: sqrt(seq_len) x sqrt(seq_len)
        # For 256 tokens, that's 16x16
        spatial_size = int(np.sqrt(seq_len))
        if spatial_size * spatial_size != seq_len:
            # Pad to nearest square
            target_len = spatial_size * spatial_size
            if seq_len < target_len:
                # Pad with zeros
                pad_len = target_len - seq_len
                tokens = torch.cat([tokens, torch.zeros(1, pad_len, device=tokens.device, dtype=tokens.dtype)], dim=1)
            else:
                # Truncate
                tokens = tokens[:, :target_len]
        
        # Reshape to spatial dimensions
        tokens = tokens.reshape(1, spatial_size, spatial_size)
        
        # Decode tokens to image
        # We need to provide the zshape for the decoder
        zshape = (1, 256, spatial_size, spatial_size)  # (batch, channels, height, width)
        image = model.decode_to_img(tokens, zshape)
        
        # Convert to PIL Image
        image = image.squeeze(0).cpu()
        image = torch.clamp(image, 0, 1)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        
        return Image.fromarray(image)

def main():
    parser = argparse.ArgumentParser(description="Generate images using CustomGPT")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, default="checkpoints_variational/best_model_epoch_75.pt", help="Path to checkpoint file")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model_from_config(args.config, args.ckpt)
    model = model.to(args.device)
    
    # Get model parameters
    z_dim = model.transformer.z_dim
    L = model.transformer.L
    print(f"Model parameters: z_dim={z_dim}, L={L}")
    
    # Generate samples
    print(f"Generating {args.num_samples} images...")
    
    for i in range(args.num_samples):
        print(f"Generating image {i+1}/{args.num_samples}")
        
        # Sample random latent vectors
        z = sample_z(model, args.device, z_dim, L)
        
        # Generate token sequence
        tokens = generate_tokens(
            model, z, 
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        print(f"Generated {tokens.shape[1]} tokens: {tokens[0, :10].tolist()}...")
        
        # Convert tokens to image
        try:
            image = tokens_to_image(model, tokens)
            
            # Save image
            output_path = os.path.join(args.output_dir, f"generated_{i+1:03d}.png")
            image.save(output_path)
            print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"Error generating image {i+1}: {e}")
            continue
    
    print("Image generation complete!")

if __name__ == "__main__":
    main()
