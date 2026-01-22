---
license: apache-2.0
language:
- en
tags:
- image-text-to-text
- transformers
- safetensors
- quantized
- int4
- nf4
- bitsandbytes
- vision-language-model
- vlm
- vla
- autonomous-driving
- nvidia
- alpamayo
base_model: nvidia/Alpamayo-R1-10B
pipeline_tag: image-text-to-text
library_name: transformers
---

# Alpamayo-R1-10B-INT4-BnB

NVIDIA Alpamayo-R1-10B의 INT4 (NF4) 양자화 버전입니다.

자율주행용 Vision-Language-Action (VLA) 모델로, BitsAndBytes를 사용하여 4-bit 양자화되었습니다.

## Benchmark Results (RTX 3090 24GB)

| Model | Bits | VRAM Load | VRAM Peak | Status |
|-------|------|-----------|-----------|--------|
| BF16 (Original) | 16 | 20.89 GB | 20.89 GB | Works on 3090 |
| **INT4-BnB (this)** | 4 | 7.63 GB | 10.44 GB | Works on 3090 |
| 2-bit (Quanto) | 2 | 4.14 GB | ~5 GB | Orin Nano OK |
| 1.58-bit (BitNet) | 1.58 | ~2.5 GB | ~4 GB | Orin Nano OK |

## Model Architecture

```
AlpamayoR1
├── vlm: Qwen3VLForConditionalGeneration
├── expert: Qwen3VLTextModel
├── action_space: UnicycleAccelCurvatureActionSpace
├── diffusion: FlowMatching
├── action_in_proj: PerWaypointActionInProjV2
└── action_out_proj: Linear4bit
```

- Parameters: 11,078,526,194 (11B)
- Vision: Qwen3-VL backbone
- Action: Flow Matching diffusion for trajectory prediction

## Requirements

```bash
pip install torch transformers bitsandbytes accelerate
pip install einops hydra-core

# alpamayo_r1 package
git clone https://github.com/NVlabs/alpamayo
cd alpamayo
pip install -e .
```

## Usage

```python
import torch
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig

# Register Alpamayo model
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.config import AlpamayoR1Config

AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)

# INT4 NF4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load model
model = AutoModel.from_pretrained(
    "kimhyunwoo/alpamayo-r1-10b-int4-bnb",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
# Output: VRAM: 7.63 GB
```

## 2-bit Quantization (For Orin Nano)

```python
from quanto import freeze, qint2, quantize

# Load BF16 to CPU first
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
# Output: VRAM: 4.14 GB  <- Fits Orin Nano 8GB!
```

## Target Hardware

| Device | VRAM | BF16 | INT4 | 2-bit |
|--------|------|------|------|-------|
| A100/H100 | 40-80 GB | OK | OK | OK |
| RTX 4090 | 24 GB | OK | OK | OK |
| **RTX 3090** | 24 GB | OK (tight) | **OK** | OK |
| RTX 3080 | 10 GB | No | Tight | OK |
| Orin NX | 16 GB | No | OK | OK |
| **Orin Nano** | 8 GB | No | No | **OK** |

## Links

- Original: [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- GitHub: [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo)

## Citation

```bibtex
@article{alpamayo2025,
  title={Alpamayo: Multimodal In-Context Learning for Autonomous Driving},
  author={NVIDIA},
  year={2025}
}
```
