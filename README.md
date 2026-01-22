# Alpamayo Nano

Pre-quantized 4-bit Alpamayo-R1 for edge deployment.

## Overview

4-bit quantized NVIDIA Alpamayo-R1-10B VLA model for autonomous driving, optimized for edge devices.

- **Model**: Alpamayo-R1-10B-4bit (pre-quantized NF4)
- **VRAM**: ~7.6GB (fits RTX 3060 12GB, RTX 3080 10GB+)
- **Inference**: ~1.0 FPS on RTX 3090

## Quick Start

```bash
pip install torch transformers bitsandbytes accelerate einops
pip install git+https://github.com/NVlabs/alpamayo-r1.git

# Run inference
python inference.py --image road.jpg
```

## Usage

```python
from alpamayo_nano import AlpamayoNano

# Load pre-quantized 4-bit model
model = AlpamayoNano.from_pretrained("dwko/Alpamayo-R1-10B-4bit")

# Inference
trajectory, reasoning = model.infer(
    images=[front_camera],
    prompt="Please drive carefully on the road."
)

# trajectory: (T, 3) waypoints [x, y, heading]
# reasoning: Chain-of-Causation text
```

## Memory Usage

| Device | VRAM | Status |
|--------|------|--------|
| RTX 4090 | 24 GB | OK (fast) |
| RTX 3090 | 24 GB | OK |
| RTX 3080 | 10 GB | OK (tight) |
| RTX 3060 | 12 GB | OK |
| Orin NX | 16 GB | OK |

## Features

### 1. Inference API
Simple wrapper for trajectory prediction with Chain-of-Causation reasoning.

### 2. jetson-server Integration
Web-based robot control with Tesla FSD-style visualization.

### 3. Trajectory Head Fine-tuning
Train custom trajectory head while keeping VLM frozen (memory efficient).

## Project Structure

```
alpamayo_nano/
├── inference.py          # Main inference script
├── alpamayo_nano.py      # Model wrapper
├── finetune/             # Head-only fine-tuning
│   └── train_head.py
└── jetson-server/        # Web control interface (submodule)
```

## Pre-quantized Models

| Model | Bits | VRAM | HuggingFace |
|-------|------|------|-------------|
| 4-bit NF4 | 4 | 7.6 GB | [dwko/Alpamayo-R1-10B-4bit](https://huggingface.co/dwko/Alpamayo-R1-10B-4bit) |
| 4-bit NF4 | 4 | 7.6 GB | [kimhyunwoo/alpamayo-r1-10b-int4-bnb](https://huggingface.co/kimhyunwoo/alpamayo-r1-10b-int4-bnb) |

## Links

- Original: [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- GitHub: [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo)
- Control Server: [hwkim3330/jetson-server](https://github.com/hwkim3330/jetson-server)

## License

Apache 2.0
