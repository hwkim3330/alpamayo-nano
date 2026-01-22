"""
Alpamayo-Nano Student Model

Lightweight model for real-time trajectory prediction,
distilled from Alpamayo-R1-10B teacher.

Target: 15-30 FPS on RTX 3090, 5-10 FPS on Orin Nano
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class StudentConfig:
    """Student model configuration."""
    # Vision encoder
    vision_encoder: str = "efficientnet_b0"  # efficientnet_b0/b4, mobilenet_v3, vit_tiny
    vision_pretrained: bool = True
    vision_freeze: bool = False

    # Feature dimensions
    vision_dim: int = 1280  # EfficientNet-B0 output
    hidden_dim: int = 512

    # Trajectory prediction
    num_waypoints: int = 33  # 5 seconds @ 6.6Hz (matching Alpamayo)
    waypoint_dim: int = 3    # x, y, heading

    # Optional: predict reasoning embedding
    predict_reasoning: bool = False
    reasoning_dim: int = 768  # text embedding dim

    # Training
    dropout: float = 0.1


class VisionEncoder(nn.Module):
    """Vision encoder using pretrained backbone."""

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config

        if config.vision_encoder == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT if config.vision_pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            self.backbone.classifier = nn.Identity()
            config.vision_dim = 1280

        elif config.vision_encoder == "efficientnet_b4":
            from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
            weights = EfficientNet_B4_Weights.DEFAULT if config.vision_pretrained else None
            self.backbone = efficientnet_b4(weights=weights)
            self.backbone.classifier = nn.Identity()
            config.vision_dim = 1792

        elif config.vision_encoder == "mobilenet_v3":
            from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
            weights = MobileNet_V3_Large_Weights.DEFAULT if config.vision_pretrained else None
            self.backbone = mobilenet_v3_large(weights=weights)
            self.backbone.classifier = nn.Identity()
            config.vision_dim = 960

        elif config.vision_encoder == "vit_tiny":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            # Use ViT-B/16 and reduce, or use timm for actual tiny
            weights = ViT_B_16_Weights.DEFAULT if config.vision_pretrained else None
            self.backbone = vit_b_16(weights=weights)
            self.backbone.heads = nn.Identity()
            config.vision_dim = 768

        else:
            raise ValueError(f"Unknown vision encoder: {config.vision_encoder}")

        if config.vision_freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor
        Returns:
            (B, vision_dim) feature vector
        """
        return self.backbone(x)


class TrajectoryHead(nn.Module):
    """MLP head for trajectory prediction."""

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config

        self.mlp = nn.Sequential(
            nn.Linear(config.vision_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_waypoints * config.waypoint_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, vision_dim)
        Returns:
            (B, num_waypoints, waypoint_dim) trajectory
        """
        B = features.shape[0]
        out = self.mlp(features)
        return out.view(B, self.config.num_waypoints, self.config.waypoint_dim)


class ReasoningHead(nn.Module):
    """Optional head for predicting reasoning embedding."""

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.vision_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.reasoning_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class AlpamayoStudent(nn.Module):
    """
    Lightweight student model for real-time trajectory prediction.

    Architecture:
        Image → VisionEncoder → TrajectoryHead → Waypoints
                             ↘ ReasoningHead → Embedding (optional)
    """

    def __init__(self, config: Optional[StudentConfig] = None):
        super().__init__()
        self.config = config or StudentConfig()

        # Vision encoder
        self.vision = VisionEncoder(self.config)

        # Trajectory head
        self.trajectory_head = TrajectoryHead(self.config)

        # Optional reasoning head
        if self.config.predict_reasoning:
            self.reasoning_head = ReasoningHead(self.config)
        else:
            self.reasoning_head = None

        self._init_weights()

    def _init_weights(self):
        """Initialize non-pretrained weights."""
        for module in [self.trajectory_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            images: (B, C, H, W) or (B, T, C, H, W) for multi-frame
            return_features: whether to return intermediate features

        Returns:
            trajectory: (B, num_waypoints, 3)
            reasoning_embedding: (B, reasoning_dim) or None
        """
        # Handle multi-frame input
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            # Use last frame or average
            images = images[:, -1]  # (B, C, H, W)

        # Vision encoding
        features = self.vision(images)  # (B, vision_dim)

        # Trajectory prediction
        trajectory = self.trajectory_head(features)

        # Optional reasoning prediction
        reasoning = None
        if self.reasoning_head is not None:
            reasoning = self.reasoning_head(features)

        if return_features:
            return trajectory, reasoning, features

        return trajectory, reasoning

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Simple inference interface."""
        self.eval()
        with torch.no_grad():
            trajectory, _ = self.forward(image)
        return trajectory

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda"):
        """Load pretrained student model."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get("config", StudentConfig())
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "config": self.config,
            "model_state_dict": self.state_dict(),
        }, path)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_flops(self, input_size: Tuple[int, int] = (224, 224)) -> int:
        """Estimate FLOPs (requires thop package)."""
        try:
            from thop import profile
            dummy = torch.randn(1, 3, *input_size).to(next(self.parameters()).device)
            flops, _ = profile(self, inputs=(dummy,), verbose=False)
            return int(flops)
        except ImportError:
            return -1


def create_student(
    encoder: str = "efficientnet_b0",
    num_waypoints: int = 33,
    pretrained: bool = True,
) -> AlpamayoStudent:
    """Factory function for creating student models."""
    config = StudentConfig(
        vision_encoder=encoder,
        vision_pretrained=pretrained,
        num_waypoints=num_waypoints,
    )
    return AlpamayoStudent(config)


# Quick benchmark
if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  Alpamayo Student Model Benchmark")
    print("=" * 60)

    for encoder in ["efficientnet_b0", "mobilenet_v3"]:
        print(f"\n{encoder}:")

        model = create_student(encoder=encoder).to(device)
        model.eval()

        params = model.count_parameters()
        print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")

        # Warmup
        dummy = torch.randn(1, 3, 224, 224).to(device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy)

        # Benchmark
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.time()

        n_iter = 100
        for _ in range(n_iter):
            with torch.no_grad():
                _ = model(dummy)

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.time() - t0

        fps = n_iter / elapsed
        print(f"  Speed: {fps:.1f} FPS ({1000/fps:.1f}ms)")

        if device == "cuda":
            mem = torch.cuda.max_memory_allocated() / 1e6
            print(f"  VRAM: {mem:.1f} MB")
