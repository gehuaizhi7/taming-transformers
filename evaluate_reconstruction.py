#!/usr/bin/env python3
"""
Evaluate reconstruction of the model:

Modes:
- reconstruct: encode an input image to z indices and decode back (x -> z_indices -> x_rec)
- decode_indices: load z indices from .npy and decode to image
- sample_with_latent: provide latent z (Gaussian or loaded .npy) and sample token indices conditioned on SOS, then decode
- random_reconstruct: randomly select images from training dataset and reconstruct them with side-by-side comparison

Outputs are written under eval_outputs/.
"""

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from main import instantiate_from_config
from taming.data.tiny_imagenet import TinyImageNetTrain


def load_model(config_path: str, checkpoint_path: str, device: str):
    print(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Support either full state dict or nested dict with key 'model_state_dict'
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def prepare_image(image_path: str, image_size: int, device: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (C,H,W)
    tensor = tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
    tensor = tensor.unsqueeze(0).to(device)  # (1,C,H,W)
    return tensor


def to_pil_image(x: torch.Tensor) -> Image.Image:
    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    x = (x * 255.0).round().to(torch.uint8)
    x = x.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


@torch.no_grad()
def run_reconstruct(model, image_path: str, outdir: str, image_size: int):
    os.makedirs(outdir, exist_ok=True)
    print("Running reconstruction from image -> z_indices -> image_rec")
    x = prepare_image(image_path, image_size, device=model.device)
    quant_z, z_indices = model.encode_to_z(x)
    # Decode using quant_z.shape which matches expected zshape for decode_to_img
    x_rec = model.decode_to_img(z_indices, quant_z.shape)
    rec_img = to_pil_image(x_rec.squeeze(0))
    outpath = os.path.join(outdir, "reconstruction.png")
    rec_img.save(outpath)
    print(f"Saved reconstruction to {outpath}")

    # Optionally save z_indices for inspection
    np.save(os.path.join(outdir, "z_indices.npy"), z_indices.detach().cpu().numpy())
    print(f"Saved z_indices to {os.path.join(outdir, 'z_indices.npy')}")


@torch.no_grad()
def run_decode_indices(model, indices_path: str, outdir: str, zshape_hint: Optional[str] = None):
    os.makedirs(outdir, exist_ok=True)
    print(f"Decoding z indices from {indices_path}")
    z_indices = np.load(indices_path)
    # Expect shape (B,S) or (S,) -> make (1,S)
    if z_indices.ndim == 1:
        z_indices = z_indices[None, :]
    z_indices = torch.from_numpy(z_indices).long().to(model.device)

    if zshape_hint is not None:
        # Format: B,C,H,W
        parts = [int(p) for p in zshape_hint.split('x')]
        assert len(parts) == 4, "zshape_hint must be 'BxCxHxW'"
        zshape = tuple(parts)
    else:
        # Fallback: infer zshape from model defaults (common for 256x256 VQGAN -> (B,256,16,16))
        batch_size = z_indices.shape[0]
        zshape = (batch_size, 256, 16, 16)
        print(f"zshape_hint not provided, using default {zshape}")

    x_rec = model.decode_to_img(z_indices, zshape)
    img = to_pil_image(x_rec.squeeze(0))
    outpath = os.path.join(outdir, "decoded_from_indices.png")
    img.save(outpath)
    print(f"Saved decoded image to {outpath}")


@torch.no_grad()
def run_random_reconstruct(model, outdir: str, data_path: str, num_images: int, image_size: int, seed: Optional[int] = None, compact_grid: bool = False):
    """Randomly select images from training dataset and reconstruct them"""
    os.makedirs(outdir, exist_ok=True)
    
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Using random seed: {seed}")
    
    print(f"Loading training dataset from {data_path}")
    dataset = TinyImageNetTrain(data_path=data_path, size=image_size, crop_size=image_size)
    
    # Randomly select indices
    total_images = len(dataset)
    selected_indices = random.sample(range(total_images), min(num_images, total_images))
    print(f"Selected {len(selected_indices)} random images for reconstruction")
    
    # Common variables
    tile_size = image_size
    
    if compact_grid:
        # Compact grid layout: square grid with alternating original/reconstructed
        grid_size = int(np.ceil(np.sqrt(num_images * 2)))  # *2 for original + reconstructed
        padding = 5
        label_height = 0  # No labels in compact mode
        grid_width = grid_size * tile_size + (grid_size + 1) * padding
        grid_height = grid_size * tile_size + (grid_size + 1) * padding
        comparison_grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # White background
    else:
        # Side-by-side layout: 2 columns (original | reconstructed) x num_images rows
        padding = 10  # Padding between images
        label_height = 30  # Height for labels
        
        # Calculate grid dimensions
        grid_width = 2 * tile_size + 3 * padding  # 2 images + 3 paddings
        grid_height = num_images * (tile_size + label_height + padding) + padding
        
        comparison_grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # White background
    
    for i, idx in enumerate(selected_indices):
        print(f"Processing image {i+1}/{len(selected_indices)} (dataset index {idx})")
        
        # Get original image
        sample = dataset[idx]
        original_image = sample["image"]  # Already in [-1, 1] range
        
        # Convert to tensor format expected by model
        x = original_image.unsqueeze(0).to(model.device)  # (1, C, H, W)
        
        # Encode and decode
        quant_z, z_indices = model.encode_to_z(x)
        x_rec = model.decode_to_img(z_indices, quant_z.shape)
        
        # Convert both to PIL images
        orig_img = to_pil_image(original_image)
        rec_img = to_pil_image(x_rec.squeeze(0))
        
        # Save individual images
        orig_path = os.path.join(outdir, f"original_{i+1:03d}.png")
        rec_path = os.path.join(outdir, f"reconstructed_{i+1:03d}.png")
        orig_img.save(orig_path)
        rec_img.save(rec_path)
        
        # Calculate positions for this row
        row_start_y = padding + i * (tile_size + label_height + padding)
        row_end_y = row_start_y + tile_size
        
        # Original image position (left column)
        orig_start_x = padding
        orig_end_x = orig_start_x + tile_size
        
        # Reconstructed image position (right column)
        rec_start_x = padding + tile_size + padding
        rec_end_x = rec_start_x + tile_size
        
        # Convert PIL images to numpy arrays
        orig_arr = np.array(orig_img)
        rec_arr = np.array(rec_img)
        
        if compact_grid:
            # Compact grid: place original and reconstructed in alternating positions
            # Original image position
            orig_pos = i * 2
            orig_row = orig_pos // grid_size
            orig_col = orig_pos % grid_size
            orig_start_y = padding + orig_row * (tile_size + padding)
            orig_end_y = orig_start_y + tile_size
            orig_start_x = padding + orig_col * (tile_size + padding)
            orig_end_x = orig_start_x + tile_size
            
            # Reconstructed image position
            rec_pos = i * 2 + 1
            rec_row = rec_pos // grid_size
            rec_col = rec_pos % grid_size
            rec_start_y = padding + rec_row * (tile_size + padding)
            rec_end_y = rec_start_y + tile_size
            rec_start_x = padding + rec_col * (tile_size + padding)
            rec_end_x = rec_start_x + tile_size
            
            # Place images
            comparison_grid[orig_start_y:orig_end_y, orig_start_x:orig_end_x] = orig_arr
            comparison_grid[rec_start_y:rec_end_y, rec_start_x:rec_end_x] = rec_arr
        else:
            # Side-by-side layout
            # Calculate positions for this row
            row_start_y = padding + i * (tile_size + label_height + padding)
            row_end_y = row_start_y + tile_size
            
            # Original image position (left column)
            orig_start_x = padding
            orig_end_x = orig_start_x + tile_size
            
            # Reconstructed image position (right column)
            rec_start_x = padding + tile_size + padding
            rec_end_x = rec_start_x + tile_size
            
            # Place images in grid
            comparison_grid[row_start_y:row_end_y, orig_start_x:orig_end_x] = orig_arr
            comparison_grid[row_start_y:row_end_y, rec_start_x:rec_end_x] = rec_arr
            
            # Add labels (simple text using PIL)
            from PIL import ImageDraw, ImageFont
            try:
                # Try to use a default font
                font = ImageFont.load_default()
            except:
                font = None
            
            # Create a temporary image for labels
            label_img = Image.new('RGB', (grid_width, label_height), (255, 255, 255))
            draw = ImageDraw.Draw(label_img)
            
            # Draw labels
            orig_label = f"Original {i+1}"
            rec_label = f"Reconstructed {i+1}"
            
            if font:
                draw.text((orig_start_x, 5), orig_label, fill=(0, 0, 0), font=font)
                draw.text((rec_start_x, 5), rec_label, fill=(0, 0, 0), font=font)
            else:
                draw.text((orig_start_x, 5), orig_label, fill=(0, 0, 0))
                draw.text((rec_start_x, 5), rec_label, fill=(0, 0, 0))
            
            # Place labels in the grid
            label_start_y = row_end_y
            label_end_y = label_start_y + label_height
            label_arr = np.array(label_img)
            comparison_grid[label_start_y:label_end_y, :] = label_arr
    
    # Save comparison grid
    comparison_path = os.path.join(outdir, f"comparison_grid_{num_images}.png")
    Image.fromarray(comparison_grid).save(comparison_path)
    print(f"Saved comparison grid to {comparison_path}")
    
    # Save z_indices for all images
    all_z_indices = []
    for i, idx in enumerate(selected_indices):
        sample = dataset[idx]
        original_image = sample["image"]
        x = original_image.unsqueeze(0).to(model.device)
        _, z_indices = model.encode_to_z(x)
        all_z_indices.append(z_indices.cpu().numpy())
    
    # Save all z_indices
    all_z_indices = np.concatenate(all_z_indices, axis=0)
    z_indices_path = os.path.join(outdir, f"all_z_indices_{num_images}.npy")
    np.save(z_indices_path, all_z_indices)
    print(f"Saved all z_indices to {z_indices_path}")


@torch.no_grad()
def run_sample_with_latent(model, outdir: str, steps: int, temperature: float, top_k: Optional[int],
                           latent_path: Optional[str] = None, grid: bool = False, grid_size: int = 1):
    os.makedirs(outdir, exist_ok=True)
    print("Sampling with latent z using model.sample_with_latent")

    batch_size = grid_size * grid_size if grid else 1
    # Conditioning: SOS token indices for unconditional
    c_indices = torch.tensor([[0]] * batch_size, dtype=torch.long, device=model.device)
    z_start_indices = torch.zeros((batch_size, 0), dtype=torch.long, device=model.device)

    # Determine latent z
    if latent_path is not None:
        z = np.load(latent_path)
        z = torch.from_numpy(z).float().to(model.device)
        print(f"Loaded latent z from {latent_path} with shape {tuple(z.shape)}")
    else:
        z = None  # Let the model draw from prior internally
        print("No latent provided, using model's prior (gaussian)")

    # Steps for 256x256 VQGAN latent is 16x16 = 256 tokens
    index = model.sample_with_latent(
        z_start_indices,
        c_indices,
        steps=steps,
        z=z,
        temperature=temperature,
        sample=True,
        top_k=top_k,
        callback=lambda k: print(f"  Generated {k} tokens") if k % 50 == 0 else None,
    )

    # Decode
    zshape = (batch_size, 256, 16, 16)
    x = model.decode_to_img(index, zshape)
    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    x = (x * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    if grid and batch_size > 1:
        # Make a grid image
        tile = 256
        grid_img = np.zeros((grid_size * tile, grid_size * tile, 3), dtype=np.uint8)
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < x.shape[0]:
                    grid_img[i*tile:(i+1)*tile, j*tile:(j+1)*tile] = x[idx]
        outpath = os.path.join(outdir, f"sample_with_latent_grid_{grid_size}x{grid_size}.png")
        Image.fromarray(grid_img).save(outpath)
        print(f"Saved grid to {outpath}")
    else:
        outpath = os.path.join(outdir, "sample_with_latent.png")
        Image.fromarray(x[0]).save(outpath)
        print(f"Saved sample to {outpath}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model reconstruction and decoding")
    p.add_argument("--mode", choices=["reconstruct", "decode_indices", "sample_with_latent", "random_reconstruct"], required=True)
    p.add_argument("--config", default="configs/tiny_imagenet_transformer.yaml")
    p.add_argument("--ckpt", default="checkpoints_variational/best_model_epoch_11.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default="eval_outputs")

    # reconstruct
    p.add_argument("--image", help="Path to input image for reconstruction")
    p.add_argument("--image_size", type=int, default=256)

    # decode_indices
    p.add_argument("--indices", help="Path to .npy z indices file")
    p.add_argument("--zshape", help="Hint for zshape as BxCxHxW, e.g., 1x256x16x16")

    # sample_with_latent
    p.add_argument("--latent", help="Path to .npy latent z tensor (optional)")
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--grid", action="store_true")
    p.add_argument("--grid_size", type=int, default=4)

    # random_reconstruct
    p.add_argument("--data_path", help="Path to Tiny ImageNet dataset (e.g., data/tiny-imagenet-200)")
    p.add_argument("--num_images", type=int, default=10, help="Number of random images to reconstruct")
    p.add_argument("--seed", type=int, help="Random seed for reproducible selection")
    p.add_argument("--compact_grid", action="store_true", help="Use compact grid layout (square grid) instead of side-by-side")

    return p.parse_args()


def main():
    args = parse_args()
    model = load_model(args.config, args.ckpt, args.device)
    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == "reconstruct":
        if not args.image:
            raise ValueError("--image is required for reconstruct mode")
        run_reconstruct(model, args.image, args.outdir, args.image_size)

    elif args.mode == "decode_indices":
        if not args.indices:
            raise ValueError("--indices is required for decode_indices mode")
        run_decode_indices(model, args.indices, args.outdir, args.zshape)

    elif args.mode == "sample_with_latent":
        run_sample_with_latent(model, args.outdir, args.steps, args.temperature, args.top_k,
                               latent_path=args.latent, grid=args.grid, grid_size=args.grid_size)

    elif args.mode == "random_reconstruct":
        if not args.data_path:
            raise ValueError("--data_path is required for random_reconstruct mode")
        run_random_reconstruct(model, args.outdir, args.data_path, args.num_images, args.image_size, args.seed, args.compact_grid)


if __name__ == "__main__":
    main()


