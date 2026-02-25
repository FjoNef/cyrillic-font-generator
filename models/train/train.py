"""
Training script for Cyrillic font generation cGAN.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from model import FontGeneratorGAN, PatchDiscriminator
from dataset import FontDataset, SyntheticFontDataset, collect_font_files


def train_gan(
    generator: FontGeneratorGAN,
    discriminator: PatchDiscriminator,
    dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
    checkpoint_dir: str,
    sample_dir: str,
    log_dir: str,
    lr_g: float = 0.0002,
    lr_d: float = 0.0002,
    lambda_l1: float = 100.0,
):
    """
    Train the cGAN model with adversarial + L1 loss.
    """
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # Loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    
    # TensorBoard logger
    writer = SummaryWriter(log_dir)
    
    # Training loop
    global_step = 0
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (style_glyphs, target_glyphs, char_indices) in enumerate(pbar):
            batch_size = style_glyphs.size(0)
            style_glyphs = style_glyphs.to(device)
            target_glyphs = target_glyphs.to(device)
            char_indices = char_indices.to(device)
            
            # Labels for real/fake
            real_labels = torch.ones(batch_size, 1, 14, 14, device=device)
            fake_labels = torch.zeros(batch_size, 1, 14, 14, device=device)
            
            # Generate fake glyphs
            fake_glyphs = generator(style_glyphs, char_indices)
            
            # ============ Train Discriminator ============
            optimizer_d.zero_grad()
            
            # Real glyphs
            # Use first style glyph as conditioning for discriminator
            style_cond = style_glyphs[:, 0, :, :, :]  # [B, 1, 128, 128]
            pred_real = discriminator(target_glyphs, style_cond)
            loss_d_real = criterion_gan(pred_real, real_labels)
            
            # Fake glyphs
            pred_fake = discriminator(fake_glyphs.detach(), style_cond)
            loss_d_fake = criterion_gan(pred_fake, fake_labels)
            
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()
            
            # ============ Train Generator ============
            optimizer_g.zero_grad()
            
            # Adversarial loss (fool discriminator)
            pred_fake_g = discriminator(fake_glyphs, style_cond)
            loss_g_gan = criterion_gan(pred_fake_g, real_labels)
            
            # L1 loss (pixel-wise reconstruction)
            loss_g_l1 = criterion_l1(fake_glyphs, target_glyphs)
            
            # Combined generator loss
            loss_g = loss_g_gan + lambda_l1 * loss_g_l1
            loss_g.backward()
            optimizer_g.step()
            
            # Accumulate losses
            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G': f'{loss_g.item():.4f}',
                'D': f'{loss_d.item():.4f}',
                'L1': f'{loss_g_l1.item():.4f}',
            })
            
            # Log to TensorBoard
            if batch_idx % 50 == 0:
                writer.add_scalar('Loss/Generator', loss_g.item(), global_step)
                writer.add_scalar('Loss/Discriminator', loss_d.item(), global_step)
                writer.add_scalar('Loss/L1', loss_g_l1.item(), global_step)
            
            global_step += 1
        
        # Epoch statistics
        avg_loss_g = epoch_loss_g / len(dataloader)
        avg_loss_d = epoch_loss_d / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - G: {avg_loss_g:.4f}, D: {avg_loss_d:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1:04d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'loss_g': avg_loss_g,
                'loss_d': avg_loss_d,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Generate sample outputs
        if (epoch + 1) % 5 == 0:
            generator.eval()
            with torch.no_grad():
                # Use first batch for samples
                sample_style, sample_target, sample_idx = next(iter(dataloader))
                sample_style = sample_style[:4].to(device)  # First 4 samples
                sample_idx = sample_idx[:4].to(device)
                sample_target = sample_target[:4].to(device)
                
                sample_fake = generator(sample_style, sample_idx)
                
                # Save comparison grid
                comparison = torch.cat([
                    sample_target,
                    sample_fake,
                    sample_style[:, 0, :, :, :],  # First style glyph for reference
                ], dim=0)
                
                sample_path = os.path.join(sample_dir, f'epoch_{epoch+1:04d}.png')
                save_image(comparison, sample_path, nrow=4, normalize=True, value_range=(-1, 1))
            
            generator.train()
    
    writer.close()
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train Cyrillic font generator cGAN')
    parser.add_argument('--data_dir', type=str, default='../../data/fonts',
                        help='Directory containing TTF/OTF font files')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic dataset (for testing without real fonts)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='Discriminator learning rate')
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='L1 loss weight')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--sample_dir', type=str, default='../samples',
                        help='Directory to save sample outputs')
    parser.add_argument('--log_dir', type=str, default='../logs',
                        help='TensorBoard log directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    if args.synthetic:
        print("Using synthetic dataset (random glyphs for testing)")
        dataset = SyntheticFontDataset(num_samples=1000)
    else:
        font_paths = collect_font_files(args.data_dir)
        if not font_paths:
            print(f"No fonts found in {args.data_dir}. Use --synthetic flag for testing.")
            return
        dataset = FontDataset(font_paths, augment=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Models
    generator = FontGeneratorGAN(num_chars=66, style_dim=256).to(device)
    discriminator = PatchDiscriminator().to(device)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        start_epoch = checkpoint['epoch']
    
    # Train
    train_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        sample_dir=args.sample_dir,
        log_dir=args.log_dir,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_l1=args.lambda_l1,
    )


if __name__ == '__main__':
    main()
