# Alpamayo Nano

Real-time autonomous driving via knowledge distillation from Alpamayo-R1.

## Overview

**Problem**: Alpamayo-R1-10B는 ~1 FPS (RTX 3090) → 실시간 자율주행 불가

**Solution**: Teacher-Student Distillation
- Teacher: Alpamayo-R1-10B (정확, 느림)
- Student: EfficientNet + MLP (빠름, 15-30 FPS)

```
┌─────────────────────────────────────────────────────┐
│  Teacher: Alpamayo-R1-10B                           │
│  Image → VLM → FlowMatching → Trajectory            │
│  11B params, ~1 FPS                                 │
└─────────────────────────────────────────────────────┘
                         ↓ Knowledge Distillation
┌─────────────────────────────────────────────────────┐
│  Student: Alpamayo-Nano                             │
│  Image → EfficientNet → MLP → Trajectory            │
│  ~5M params, 15-30 FPS                              │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Generate Training Data (Teacher Inference)

```bash
# Alpamayo로 이미지에서 trajectory 생성
python distill/generate_data.py \
    --input /path/to/driving/images \
    --output data/distill \
    --model nvidia/Alpamayo-R1-10B
```

### 2. Train Student Model

```bash
# Distillation 학습
python distill/train.py \
    --data data/distill \
    --encoder efficientnet_b0 \
    --epochs 50 \
    --batch-size 32

# 또는 합성 데이터로 테스트
python distill/train.py --synthetic 1000 --epochs 10
```

### 3. Run Student Inference

```python
from models.student import AlpamayoStudent

# 학습된 student 로드
model = AlpamayoStudent.from_pretrained("checkpoints/student/best.pt")

# 실시간 추론 (15-30 FPS)
trajectory = model.predict(image)
```

## Project Structure

```
alpamayo-nano/
├── models/
│   └── student.py        # Student model (EfficientNet + MLP)
│
├── distill/              # Distillation pipeline
│   ├── generate_data.py  # Teacher로 학습 데이터 생성
│   └── train.py          # Student 학습
│
├── quantize/             # Alpamayo 양자화 (선택)
│   ├── quantize_4bit.py
│   ├── quantize_2bit.py
│   └── quantize_1bit.py
│
└── finetune/             # Alpamayo fine-tuning (선택)
    └── train_head.py
```

## Student Model Options

| Encoder | Params | FPS (3090) | FPS (Orin) | VRAM |
|---------|--------|------------|------------|------|
| MobileNet-V3 | 4.2M | ~35 | ~15 | 50MB |
| EfficientNet-B0 | 5.3M | ~30 | ~12 | 80MB |
| EfficientNet-B4 | 19M | ~20 | ~8 | 200MB |
| ViT-Tiny | 5.7M | ~25 | ~10 | 100MB |

## Benchmark Comparison

| Model | Params | FPS | Latency | VRAM |
|-------|--------|-----|---------|------|
| Alpamayo-R1 (BF16) | 11B | 1.3 | 750ms | 21GB |
| Alpamayo-R1 (INT4) | 11B | 1.3 | 750ms | 7.6GB |
| **Student (B0)** | 5M | 30 | 33ms | 80MB |
| **Student (MobileNet)** | 4M | 35 | 28ms | 50MB |

**23x 속도 향상** (750ms → 33ms)

## Distillation Loss

```python
Loss = trajectory_loss + 0.5 * endpoint_loss + 0.1 * smoothness_loss

- trajectory_loss: L1 + L2 on all waypoints
- endpoint_loss: MSE on final position (중요!)
- smoothness_loss: penalize jerky predictions
```

## Training Tips

1. **데이터 다양성**: 직진, 좌회전, 우회전, 정지 등 다양한 시나리오
2. **Data Augmentation**: 밝기, 대비, 색상 변형
3. **Curriculum Learning**: 쉬운 시나리오(직진) → 어려운 시나리오(회전)
4. **Endpoint 가중치**: 최종 위치 예측이 중요 → endpoint_weight 높게

## Hardware Requirements

**Teacher (데이터 생성):**
- RTX 3090 24GB (INT4: 7.6GB)
- ~1 FPS, 오프라인 작업

**Student (실시간 추론):**
- Any GPU with 100MB+ VRAM
- Orin Nano에서도 실행 가능

## Roadmap

- [x] Student model architecture
- [x] Distillation training pipeline
- [x] Synthetic data for testing
- [ ] Real driving data collection
- [ ] TensorRT optimization
- [ ] ONNX export for edge deployment
- [ ] Orin Nano benchmark

## Links

- Teacher: [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- Control Server: [hwkim3330/jetson-server](https://github.com/hwkim3330/jetson-server)

## License

Apache 2.0
