#!/usr/bin/env python3
"""
Quick image generation test for Tiny ImageNet using pretrained ImageNet VQGAN
"""

import torch
import numpy as np
from omegaconf import OmegaConf
from main import instantiate_from_config
from PIL import Image
import os

def quick_generate_tiny_imagenet_pretrained_vqgan():
    """Quick test of image generation on Tiny ImageNet with pretrained VQGAN"""
    
    print("Loading model...")
    config = OmegaConf.load("configs/tiny_imagenet_transformer.yaml")
    model = instantiate_from_config(config.model)
    
    # Load the trained checkpoint
    checkpoint_path = "checkpoints_variational_cyclical/best_model_epoch_1.pt"
    print(f"Loading trained checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"z_dim={model.transformer.z_dim}, L={model.transformer.L}")
    
    # Create output directory
    os.makedirs("quick_generated_tiny_imagenet_pretrained_vqgan", exist_ok=True)
    
    # Grid parameters
    grid_size = 4  # 4x4 grid = 16 images
    image_size = 256  # Size of each individual image
    
    print(f"\nGenerating {grid_size}x{grid_size} grid of images...")
    
    try:
        with torch.no_grad():
            # Generate all images in a batch for efficiency
            batch_size = grid_size * grid_size
            print(f"Generating batch of {batch_size} images...")
            
            # For unconditional generation, we need to create proper conditioning tokens
            # The model expects c to be token indices, not z vectors
            
            # Create conditioning sequences for all images in batch
            c_indices = torch.tensor([[0]] * batch_size, dtype=torch.long, device=model.device)  # SOS token for each
            
            # Start with empty sequences for unconditional generation
            z_start_indices = torch.zeros((batch_size, 0), dtype=torch.long, device=model.device)
            
            # Use the model's sample method for proper generation
            # For 256x256 images with pretrained VQGAN, we need 16x16 = 256 tokens
            # Latent z-driven sampling: pass z=None to draw z ~ N(0, I) internally
            generated_tokens = model.sample_with_latent(
                z_start_indices,
                c_indices,
                steps=256,  # 16x16 latent tokens
                z=None,     # sample z ~ N(0, I)
                temperature=1.0,
                sample=True,
                top_k=100,
                callback=lambda k: print(f"  Generated {k} tokens") if k % 50 == 0 else None
            )
            
            print(f"Generated {generated_tokens.shape[1]} tokens total for {batch_size} images")
            
            # Decode all images at once
            zshape = (batch_size, 256, 16, 16)  # (batch, channels, height, width) for 256x256 -> 16x16 latent
            images = model.decode_to_img(generated_tokens, zshape)
            
            # Convert to numpy arrays
            images = images.cpu()
            # Clamp to [0, 1] range and convert to [0, 255]
            images = torch.clamp((images + 1.0) / 2.0, 0, 1)  # VQGAN outputs [-1, 1], convert to [0, 1]
            images = images.permute(0, 2, 3, 1).numpy()  # (batch, height, width, channels)
            images = (images * 255).astype(np.uint8)
            
            # Create grid
            print("Creating grid layout...")
            grid_width = grid_size * image_size
            grid_height = grid_size * image_size
            grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if idx < len(images):
                        start_y = i * image_size
                        end_y = start_y + image_size
                        start_x = j * image_size
                        end_x = start_x + image_size
                        grid_image[start_y:end_y, start_x:end_x] = images[idx]
            
            # Save grid image
            output_path = "quick_generated_tiny_imagenet_pretrained_vqgan/grid_4x4.png"
            Image.fromarray(grid_image).save(output_path)
            print(f"Saved grid: {output_path}")
            
            # Also save individual images for reference
            for i in range(min(batch_size, 16)):  # Save first 16 individual images
                individual_path = f"quick_generated_tiny_imagenet_pretrained_vqgan/individual_{i+1:03d}.png"
                Image.fromarray(images[i]).save(individual_path)
            
            print(f"Saved {min(batch_size, 16)} individual images for reference")
                
    except Exception as e:
        print(f"Error generating grid: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nQuick generation test complete!")

if __name__ == "__main__":
    quick_generate_tiny_imagenet_pretrained_vqgan()



