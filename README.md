# Alpamayo Nano

NVIDIA Alpamayo-R1-10B VLA 모델의 양자화 및 Edge 배포 파이프라인.

## Overview

Alpamayo-R1-10B를 다양한 하드웨어에서 실행하기 위한 양자화 도구 모음:

| 방법 | Bits | VRAM | 타겟 하드웨어 |
|------|------|------|--------------|
| BF16 (원본) | 16 | ~21GB | RTX 3090/4090 |
| **INT4 NF4** | 4 | ~7.6GB | RTX 3060+, Orin NX |
| 2-bit Quanto | 2 | ~4GB | Orin Nano 8GB |
| 1.58-bit BitNet | 1.58 | ~3GB | Orin Nano (실험적) |

## Quick Start

### Option 1: Pre-quantized 모델 사용 (권장)

```python
from alpamayo_nano import AlpamayoNano

# HuggingFace에서 4-bit 모델 로드
model = AlpamayoNano.from_pretrained("dwko/Alpamayo-R1-10B-4bit")

# 추론
trajectory, reasoning = model.infer([camera_image])
```

### Option 2: 직접 양자화

```bash
# 4-bit 양자화
python quantize/quantize_4bit.py --output ./my-alpamayo-4bit

# 2-bit 양자화 (Orin Nano용)
python quantize/quantize_2bit.py --output ./my-alpamayo-2bit

# 1.58-bit BitNet (실험적)
python quantize/quantize_1bit.py --output ./my-alpamayo-1bit
```

## Installation

```bash
pip install torch transformers bitsandbytes accelerate einops
pip install git+https://github.com/NVlabs/alpamayo-r1.git

# 2-bit 양자화용
pip install quanto

# Fine-tuning용
pip install peft datasets
```

## Project Structure

```
alpamayo-nano/
├── alpamayo_nano.py      # 4-bit 모델 래퍼 (inference API)
├── inference.py          # CLI 추론
│
├── quantize/             # 양자화 파이프라인
│   ├── quantize_4bit.py  # INT4 NF4 (BitsAndBytes)
│   ├── quantize_2bit.py  # 2-bit (Quanto)
│   ├── quantize_1bit.py  # 1.58-bit BitNet
│   └── test_quantized.py # 양자화 모델 테스트
│
├── finetune/             # Fine-tuning
│   └── train_head.py     # Trajectory head 학습 (VLM frozen)
│
└── models/               # 커스텀 모델
    └── nano_vlm.py       # Qwen3-VL-2B 기반 경량 VLM (개발중)
```

## Benchmark Results (RTX 3090)

### 메모리

| Model | VRAM Load | VRAM Peak |
|-------|-----------|-----------|
| BF16 원본 | 20.89 GB | 22+ GB |
| **INT4 NF4** | 7.52 GB | 10.44 GB |
| 2-bit | 4.14 GB | ~5 GB |
| 1.58-bit | ~2.5 GB | ~4 GB |

### 추론 속도 (1 camera, 4 frames)

| Model | Time | FPS |
|-------|------|-----|
| BF16 | 0.75s | 1.3 |
| INT4 | 0.75s | 1.3 |
| 2-bit | 0.8s | 1.25 |

**참고**: RTX 3090에서는 양자화해도 속도 향상 없음 (dequantize 오버헤드)

## Hardware Compatibility

| Device | VRAM | BF16 | INT4 | 2-bit | 1.58-bit |
|--------|------|------|------|-------|----------|
| A100/H100 | 40-80GB | OK | OK | OK | OK |
| RTX 4090 | 24GB | OK | OK | OK | OK |
| RTX 3090 | 24GB | OK | OK | OK | OK |
| RTX 3080 | 10GB | No | Tight | OK | OK |
| RTX 3060 | 12GB | No | OK | OK | OK |
| Orin NX | 16GB | No | OK | OK | OK |
| **Orin Nano** | 8GB | No | No | **OK** | **OK** |

## Fine-tuning

VLM을 freeze하고 trajectory head만 학습:

```bash
python finetune/train_head.py \
    --model dwko/Alpamayo-R1-10B-4bit \
    --epochs 10 \
    --lr 1e-4
```

VRAM 사용량: ~10GB (4-bit 모델 + gradients)

## Pre-quantized Models

| Model | HuggingFace |
|-------|-------------|
| INT4 NF4 | [dwko/Alpamayo-R1-10B-4bit](https://huggingface.co/dwko/Alpamayo-R1-10B-4bit) |
| INT4 NF4 | [kimhyunwoo/alpamayo-r1-10b-int4-bnb](https://huggingface.co/kimhyunwoo/alpamayo-r1-10b-int4-bnb) |

## Model Architecture

```
AlpamayoR1 (11B parameters)
├── vlm: Qwen3VLForConditionalGeneration (Vision-Language)
├── expert: Qwen3VLTextModel
├── action_space: UnicycleAccelCurvatureActionSpace
├── diffusion: FlowMatching
├── action_in_proj: PerWaypointActionInProjV2
└── action_out_proj: Linear
```

## Roadmap

- [x] INT4 NF4 양자화 (BitsAndBytes)
- [x] 2-bit 양자화 (Quanto)
- [x] 1.58-bit BitNet 실험
- [x] Pre-quantized 모델 HuggingFace 업로드
- [ ] TensorRT 변환 (Orin Nano 최적화)
- [ ] 커스텀 경량 VLM (Qwen3-VL-2B 기반)
- [ ] ONNX export

## Links

- Original: [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- GitHub: [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo)
- Control Server: [hwkim3330/jetson-server](https://github.com/hwkim3330/jetson-server)

## License

Apache 2.0
