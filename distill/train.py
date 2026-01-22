#!/usr/bin/env python3
"""
Distillation training for Alpamayo-Nano student model.

Trains lightweight student to mimic Alpamayo-R1 teacher predictions.

Usage:
    python distill/train.py --data data/distill --output checkpoints/student
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.student import AlpamayoStudent, StudentConfig, create_student


class DistillDataset(Dataset):
    """Dataset for distillation training."""

    def __init__(
        self,
        data_dir: str,
        transform=None,
        max_samples: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform or self._default_transform()

        # Load samples
        self.samples = []
        samples_file = self.data_dir / "samples.jsonl"

        if samples_file.exists():
            with open(samples_file, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def _default_transform(self):
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)

        # Trajectory
        trajectory = torch.tensor(sample['trajectory'], dtype=torch.float32)

        return {
            'image': image,
            'trajectory': trajectory,
            'reasoning': sample.get('reasoning', ''),
        }


class SyntheticDistillDataset(Dataset):
    """Synthetic dataset for testing (no teacher required)."""

    def __init__(self, num_samples: int = 1000, num_waypoints: int = 33):
        self.num_samples = num_samples
        self.num_waypoints = num_waypoints
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Pre-generate trajectories
        self.trajectories = []
        for _ in range(num_samples):
            traj_type = np.random.choice(['straight', 'left', 'right'])
            self.trajectories.append(self._generate_trajectory(traj_type))

    def _generate_trajectory(self, traj_type: str) -> np.ndarray:
        trajectory = np.zeros((self.num_waypoints, 3))
        v = 5.0
        dt = 5.0 / self.num_waypoints

        if traj_type == 'straight':
            for i in range(self.num_waypoints):
                t = (i + 1) * dt
                trajectory[i] = [v * t, 0.0, 0.0]

        elif traj_type == 'left':
            radius = 20.0
            for i in range(self.num_waypoints):
                t = (i + 1) * dt
                angle = v * t / radius
                trajectory[i] = [
                    radius * np.sin(angle),
                    radius * (1 - np.cos(angle)),
                    angle
                ]

        elif traj_type == 'right':
            radius = 20.0
            for i in range(self.num_waypoints):
                t = (i + 1) * dt
                angle = -v * t / radius
                trajectory[i] = [
                    radius * np.sin(-angle),
                    -radius * (1 - np.cos(-angle)),
                    angle
                ]

        return trajectory

    def _generate_image(self) -> Image.Image:
        """Generate synthetic road image."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[112:, :] = 80  # Road
        img[112:, 70:75] = 200  # Left lane
        img[112:, 149:154] = 200  # Right lane
        noise = np.random.randint(0, 20, (224, 224, 3), dtype=np.uint8)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self._generate_image()
        image = self.transform(image)
        trajectory = torch.tensor(self.trajectories[idx], dtype=torch.float32)

        return {
            'image': image,
            'trajectory': trajectory,
            'reasoning': '',
        }


class DistillationLoss(nn.Module):
    """Combined loss for distillation training."""

    def __init__(
        self,
        trajectory_weight: float = 1.0,
        endpoint_weight: float = 0.5,
        smoothness_weight: float = 0.1
    ):
        super().__init__()
        self.trajectory_weight = trajectory_weight
        self.endpoint_weight = endpoint_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pred_traj: torch.Tensor,
        target_traj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_traj: (B, T, 3) predicted trajectory
            target_traj: (B, T, 3) teacher trajectory
        """
        # Main trajectory loss (L1 + L2)
        l1_loss = F.l1_loss(pred_traj, target_traj)
        l2_loss = F.mse_loss(pred_traj, target_traj)
        traj_loss = l1_loss + 0.5 * l2_loss

        # Endpoint loss (final position matters more)
        endpoint_loss = F.mse_loss(pred_traj[:, -1], target_traj[:, -1])

        # Smoothness loss (penalize jerky predictions)
        pred_diff = pred_traj[:, 1:] - pred_traj[:, :-1]
        target_diff = target_traj[:, 1:] - target_traj[:, :-1]
        smooth_loss = F.mse_loss(pred_diff, target_diff)

        # Total loss
        total = (
            self.trajectory_weight * traj_loss +
            self.endpoint_weight * endpoint_loss +
            self.smoothness_weight * smooth_loss
        )

        return {
            'total': total,
            'trajectory': traj_loss,
            'endpoint': endpoint_loss,
            'smoothness': smooth_loss,
        }


def train_epoch(
    model: AlpamayoStudent,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: DistillationLoss,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {'trajectory': 0., 'endpoint': 0., 'smoothness': 0.}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(device)
        target_traj = batch['trajectory'].to(device)

        optimizer.zero_grad()

        # Forward
        pred_traj, _ = model(images)

        # Loss
        losses = criterion(pred_traj, target_traj)
        loss = losses['total']

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += losses[k].item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    n = len(dataloader)
    return {
        'total': total_loss / n,
        **{k: v / n for k, v in loss_components.items()}
    }


def validate(
    model: AlpamayoStudent,
    dataloader: DataLoader,
    criterion: DistillationLoss,
    device: str,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    loss_components = {'trajectory': 0., 'endpoint': 0., 'smoothness': 0.}

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            target_traj = batch['trajectory'].to(device)

            pred_traj, _ = model(images)
            losses = criterion(pred_traj, target_traj)

            total_loss += losses['total'].item()
            for k in loss_components:
                loss_components[k] += losses[k].item()

    n = len(dataloader)
    return {
        'total': total_loss / n,
        **{k: v / n for k, v in loss_components.items()}
    }


def train(args):
    """Main training function."""
    print("=" * 60)
    print("  Alpamayo-Nano Distillation Training")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Dataset
    print(f"\nLoading dataset from {args.data}...")
    if args.synthetic:
        train_dataset = SyntheticDistillDataset(num_samples=args.synthetic)
        val_dataset = SyntheticDistillDataset(num_samples=args.synthetic // 10)
    else:
        train_dataset = DistillDataset(args.data)
        val_dataset = train_dataset  # TODO: proper split

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Model
    print(f"\nCreating student model: {args.encoder}")
    config = StudentConfig(
        vision_encoder=args.encoder,
        vision_pretrained=True,
        vision_freeze=args.freeze_vision,
        num_waypoints=33,
    )
    model = AlpamayoStudent(config).to(device)

    params = model.count_parameters()
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    criterion = DistillationLoss()

    # Training loop
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')
    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Validate
        val_losses = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Log
        print(f"Epoch {epoch}: train={train_losses['total']:.4f}, val={val_losses['total']:.4f}")

        # Save best
        if val_losses['total'] < best_loss:
            best_loss = val_losses['total']
            model.save_pretrained(output_dir / "best.pt")
            print(f"  -> Saved best model (loss={best_loss:.4f})")

        # Save checkpoint
        if epoch % args.save_every == 0:
            model.save_pretrained(output_dir / f"epoch_{epoch}.pt")

    # Save final
    model.save_pretrained(output_dir / "final.pt")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Models saved to {output_dir}")

    # Quick benchmark
    print("\n--- Inference Benchmark ---")
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)

    # Measure
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()

    n_iter = 100
    for _ in range(n_iter):
        with torch.no_grad():
            _ = model(dummy)

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - t0

    fps = n_iter / elapsed
    print(f"Speed: {fps:.1f} FPS ({1000/fps:.1f}ms)")

    if device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"VRAM: {mem:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Train student model via distillation")
    parser.add_argument("--data", type=str, default="data/distill", help="Data directory")
    parser.add_argument("--output", type=str, default="checkpoints/student", help="Output dir")
    parser.add_argument("--encoder", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "efficientnet_b4", "mobilenet_v3", "vit_tiny"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-vision", action="store_true", help="Freeze vision encoder")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--synthetic", type=int, default=None,
                        help="Use synthetic data with N samples (for testing)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
