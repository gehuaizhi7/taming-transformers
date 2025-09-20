#!/usr/bin/env python3
"""
Quick image generation test
"""

import torch
import numpy as np
from omegaconf import OmegaConf
from main import instantiate_from_config
from PIL import Image
import os

def quick_generate():
    """Quick test of image generation"""
    
    print("Loading model...")
    config = OmegaConf.load("configs/unconditional_transformer.yaml")
    model = instantiate_from_config(config.model)
    
    # Load the trained checkpoint
    checkpoint_path = "checkpoints_variational/best_model_epoch_19.pt"
    print(f"Loading trained checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"z_dim={model.transformer.z_dim}, L={model.transformer.L}")
    
    # Create output directory
    os.makedirs("quick_generated", exist_ok=True)
    
    # Generate a few images
    for i in range(10):
        print(f"\nGenerating image {i+1}/10...")
        
        try:
            with torch.no_grad():
                # For unconditional generation, we need to create proper conditioning tokens
                # The model expects c to be token indices, not z vectors
                
                # Create a dummy conditioning sequence (just SOS token for unconditional)
                c_indices = torch.tensor([[0]], dtype=torch.long, device=model.device)  # SOS token
                
                # Start with empty sequence for unconditional generation
                z_start_indices = torch.zeros((1, 0), dtype=torch.long, device=model.device)
                
                # Use the model's sample method for proper generation
                generated_tokens = model.sample(
                    z_start_indices, 
                    c_indices,  # Use conditioning tokens, not z vectors
                    steps=256,  # Generate 256 tokens
                    temperature=1.0,
                    sample=True,
                    top_k=100,
                    callback=lambda k: print(f"  Generated {k} tokens") if k % 50 == 0 else None
                )
                
                print(f"Generated {generated_tokens.shape[1]} tokens total")
                
                # Decode to image using the correct zshape format
                zshape = (1, 256, 16, 16)  # (batch, channels, height, width)
                image = model.decode_to_img(generated_tokens, zshape)
                
                # Convert to PIL Image
                image = image.squeeze(0).cpu()
                # Clamp to [0, 1] range and convert to [0, 255]
                image = torch.clamp((image + 1.0) / 2.0, 0, 1)  # VQGAN outputs [-1, 1], convert to [0, 1]
                image = image.permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
                
                # Save image
                output_path = f"quick_generated/test_{i+1:03d}.png"
                Image.fromarray(image).save(output_path)
                print(f"Saved: {output_path}")
                
        except Exception as e:
            print(f"Error generating image {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nQuick generation test complete!")

if __name__ == "__main__":
    quick_generate()
