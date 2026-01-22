---
license: apache-2.0
language:
- en
tags:
- image-text-to-text
- transformers
- quantized
- 2-bit
- quanto
- vision-language-model
- vlm
- vla
- autonomous-driving
- nvidia
- alpamayo
- jetson
- orin-nano
base_model: nvidia/Alpamayo-R1-10B
pipeline_tag: image-text-to-text
library_name: transformers
---

# Alpamayo-R1-10B-2bit

NVIDIA Alpamayo-R1-10B의 **2-bit 양자화** 버전입니다.

**Jetson Orin Nano 8GB**에서 실행 가능한 극한 압축 모델.

## Benchmark (RTX 3090)

| Model | Bits | VRAM | Compression | Target |
|-------|------|------|-------------|--------|
| BF16 (Original) | 16 | 20.89 GB | 1x | A100/H100 |
| INT4 (BnB) | 4 | 7.63 GB | 2.7x | RTX 3090 |
| **2-bit (this)** | 2 | **4.14 GB** | **5.0x** | **Orin Nano** |
| 1.58-bit (BitNet) | 1.58 | ~2.5 GB | 8x | Orin Nano |

## Key Features

- **4.14 GB VRAM** - Fits Orin Nano 8GB with margin
- **5x compression** vs BF16 original
- **Quanto qint2** quantization
- Full model architecture preserved

## Model Architecture

```
AlpamayoR1 (11B parameters)
├── vlm: Qwen3VLForConditionalGeneration
├── expert: Qwen3VLTextModel
├── action_space: UnicycleAccelCurvatureActionSpace
├── diffusion: FlowMatching
├── action_in_proj: PerWaypointActionInProjV2
└── action_out_proj: Linear
```

## Requirements

```bash
pip install torch transformers accelerate
pip install quanto
pip install einops hydra-core

# alpamayo_r1 package
git clone https://github.com/NVlabs/alpamayo
cd alpamayo && pip install -e .
```

## Usage

### Method 1: Load and Quantize on-the-fly

```python
import torch
from transformers import AutoModel, AutoConfig
from quanto import freeze, qint2, quantize

# Register Alpamayo model
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.config import AlpamayoR1Config

AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)

# Load to CPU first (saves GPU memory)
model = AutoModel.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)

# 2-bit quantization
quantize(model, weights=qint2)
freeze(model)

# Move to GPU
model = model.to("cuda")
print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
# Output: VRAM: 4.14 GB
```

### Method 2: Load Pre-quantized (Coming Soon)

```python
# Direct loading will be available after model upload
model = AutoModel.from_pretrained(
    "kimhyunwoo/alpamayo-r1-10b-2bit",
    device_map="auto",
    trust_remote_code=True,
)
```

## Hardware Compatibility

| Device | VRAM | 2-bit Status |
|--------|------|--------------|
| A100/H100 | 40-80 GB | ✅ Overkill |
| RTX 4090/3090 | 24 GB | ✅ OK |
| RTX 3080 | 10 GB | ✅ OK |
| Orin NX | 16 GB | ✅ OK |
| **Orin Nano** | 8 GB | ✅ **Target** |

## Quantization Details

- **Method**: Quanto qint2 (2-bit integer)
- **Precision**: 2 bits per weight
- **Scale**: Per-tensor or per-channel
- **Activations**: BF16 (not quantized)

## Links

- Original: [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- INT4 version: [kimhyunwoo/alpamayo-r1-10b-int4-bnb](https://huggingface.co/kimhyunwoo/alpamayo-r1-10b-int4-bnb)
- GitHub: [hwkim3330/alpamayo-nano](https://github.com/hwkim3330/alpamayo-nano)

## Citation

```bibtex
@article{alpamayo2025,
  title={Alpamayo: Multimodal In-Context Learning for Autonomous Driving},
  author={NVIDIA},
  year={2025}
}
```
