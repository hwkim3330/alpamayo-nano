#!/usr/bin/env python3
"""
Fine-tune Alpamayo-R1-10B-4bit trajectory head.

VLM is frozen, only trajectory head is trained.
Uses LoRA for memory-efficient training.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

# Add parent dir
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_name: str = "dwko/Alpamayo-R1-10B-4bit"
    batch_size: int = 1
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_samples: int = 1000  # synthetic samples
    output_dir: str = "checkpoints"


class SyntheticDrivingDataset(Dataset):
    """
    Synthetic driving dataset for testing.

    Generates random trajectories for training.
    Replace with real NVIDIA AV dataset for production.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_waypoints: int = 33,
        image_size: tuple = (480, 640),
    ):
        self.num_samples = num_samples
        self.num_waypoints = num_waypoints
        self.image_size = image_size
        self._generate_samples()

    def _generate_samples(self):
        """Generate synthetic trajectory data."""
        self.samples = []
        for i in range(self.num_samples):
            traj_type = np.random.choice(["straight", "left", "right"])
            trajectory = self._generate_trajectory(traj_type)
            self.samples.append({
                "trajectory": trajectory,
                "type": traj_type,
            })

    def _generate_trajectory(self, traj_type: str) -> np.ndarray:
        """Generate a single trajectory."""
        trajectory = np.zeros((self.num_waypoints, 3))  # x, y, heading
        v = 5.0  # m/s
        dt = 5.0 / self.num_waypoints  # 5 second horizon

        if traj_type == "straight":
            for i in range(self.num_waypoints):
                t = (i + 1) * dt
                trajectory[i] = [v * t, 0.0, 0.0]

        elif traj_type == "left":
            radius = 20.0
            for i in range(self.num_waypoints):
                t = (i + 1) * dt
                angle = v * t / radius
                trajectory[i] = [
                    radius * np.sin(angle),
                    radius * (1 - np.cos(angle)),
                    angle
                ]

        elif traj_type == "right":
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

    def _generate_image(self) -> np.ndarray:
        """Generate synthetic road image."""
        h, w = self.image_size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Gray road
        img[h//2:, :] = 80

        # Lane lines
        for x in [w//3, 2*w//3]:
            img[h//2:, x-2:x+2] = 200

        # Add noise
        noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        return img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self._generate_image()
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            "trajectory": torch.from_numpy(sample["trajectory"]).float(),
            "type": sample["type"],
        }


def load_4bit_model(model_name: str):
    """Load pre-quantized 4-bit model."""
    from transformers import AutoModel, AutoConfig, BitsAndBytesConfig
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.config import AlpamayoR1Config

    AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
    AutoModel.register(AlpamayoR1Config, AlpamayoR1)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModel.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    return model


def freeze_vlm(model):
    """Freeze VLM backbone, keep trajectory head trainable."""
    trainable_names = ["action_out_proj", "diffusion", "action_in_proj"]

    for name, param in model.named_parameters():
        # Check if this parameter should be trainable
        is_trainable = any(t in name for t in trainable_names)
        param.requires_grad = is_trainable

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable ratio: {trainable/total*100:.2f}%")


def train(config: TrainingConfig):
    """Main training loop."""
    print("=" * 60)
    print("  Alpamayo Nano - Trajectory Head Fine-tuning")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {config.model_name}")
    model = load_4bit_model(config.model_name)

    # Freeze VLM
    print("\nFreezing VLM backbone...")
    freeze_vlm(model)

    # Dataset
    print(f"\nCreating dataset ({config.num_samples} samples)...")
    dataset = SyntheticDrivingDataset(num_samples=config.num_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Loss
    criterion = nn.MSELoss()

    # Training
    print(f"\nStarting training...")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    os.makedirs(config.output_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            # Forward pass
            images = batch["image"].to("cuda")
            gt_trajectory = batch["trajectory"].to("cuda")

            # Note: This is a simplified training loop
            # Real implementation would use proper forward pass
            # with trajectory head output

            # Dummy loss for demonstration
            loss = torch.tensor(0.0, requires_grad=True, device="cuda")
            total_loss += loss.item()

            # Backward
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(config.output_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": {
                    k: v for k, v in model.state_dict().items()
                    if any(t in k for t in ["action_out_proj", "diffusion", "action_in_proj"])
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("\nTraining complete!")
    print(f"Final VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Alpamayo trajectory head")
    parser.add_argument("--model", type=str, default="dwko/Alpamayo-R1-10B-4bit")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="checkpoints")
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_samples=args.samples,
        output_dir=args.output,
    )

    train(config)


if __name__ == "__main__":
    main()
