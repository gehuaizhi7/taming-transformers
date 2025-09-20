#!/usr/bin/env python3
"""
Test the model's built-in log_images method
"""

import torch
import numpy as np
from omegaconf import OmegaConf
from main import instantiate_from_config
from PIL import Image
import os

def test_log_images():
    """Test the model's built-in image generation"""
    
    print("Loading model...")
    config = OmegaConf.load("configs/unconditional_transformer.yaml")
    model = instantiate_from_config(config.model)
    
    # Load the trained checkpoint
    checkpoint_path = "checkpoints_variational/best_model_epoch_10.pt"
    print(f"Loading trained checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create output directory
    os.makedirs("log_images_test", exist_ok=True)
    
    # Create a dummy batch for log_images
    batch = {
        'image': torch.randn(4, 3, 256, 256)  # Random images for testing
    }
    
    print("Testing log_images method...")
    
    try:
        with torch.no_grad():
            # Use the model's built-in log_images method
            log = model.log_images(
                batch, 
                temperature=1.0, 
                top_k=100,
                lr_interface=False
            )
            
            print(f"Log keys: {log.keys()}")
            
            # Save the generated images
            for i, img in enumerate(log.get('samples', [])):
                if img is not None:
                    # Convert to PIL Image
                    img = img.squeeze(0).cpu()
                    img = torch.clamp((img + 1.0) / 2.0, 0, 1)
                    img = img.permute(1, 2, 0).numpy()
                    img = (img * 255).astype(np.uint8)
                    
                    output_path = f"log_images_test/log_sample_{i+1:03d}.png"
                    Image.fromarray(img).save(output_path)
                    print(f"Saved: {output_path}")
            
            # Also try the no-pixel version
            for i, img in enumerate(log.get('samples_nopix', [])):
                if img is not None:
                    # Convert to PIL Image
                    img = img.squeeze(0).cpu()
                    img = torch.clamp((img + 1.0) / 2.0, 0, 1)
                    img = img.permute(1, 2, 0).numpy()
                    img = (img * 255).astype(np.uint8)
                    
                    output_path = f"log_images_test/log_nopix_{i+1:03d}.png"
                    Image.fromarray(img).save(output_path)
                    print(f"Saved: {output_path}")
                    
    except Exception as e:
        print(f"Error in log_images: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nLog images test complete!")

if __name__ == "__main__":
    test_log_images()


