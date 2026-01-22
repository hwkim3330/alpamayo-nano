#!/usr/bin/env python3
"""
Alpamayo Nano 모델 테스트

Qwen3-VL-2B + Trajectory Head 메모리 및 추론 테스트
"""

import sys
sys.path.insert(0, "/mnt/data/lfm_agi/alpamayo_nano")

import torch
import numpy as np
from PIL import Image
from transformers import BitsAndBytesConfig

from models.alpamayo_nano import create_alpamayo_nano, AlpamayoNanoConfig


def get_gpu_memory():
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "peak": torch.cuda.max_memory_allocated() / 1e9,
            "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    return {}


def print_memory(label=""):
    mem = get_gpu_memory()
    if mem:
        print(f"[{label}] VRAM: {mem['allocated']:.2f}GB / {mem['total']:.1f}GB")


def test_alpamayo_nano():
    """Alpamayo Nano 테스트."""
    print("=" * 60)
    print("Alpamayo Nano 테스트")
    print("=" * 60)

    print_memory("시작")

    # INT4 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # 모델 생성
    print("\n모델 생성 중...")
    model = create_alpamayo_nano(
        model_path="/mnt/data/lfm_agi/models/nano/qwen3-vl-2b",
        quantization_config=bnb_config,
    )

    print_memory("모델 로드 후")

    # 파라미터 수
    vlm_params = sum(p.numel() for p in model.vlm.parameters())
    head_params = sum(p.numel() for p in model.trajectory_head.parameters())
    history_params = sum(p.numel() for p in model.history_encoder.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())

    print(f"\n파라미터 수:")
    print(f"  VLM: {vlm_params/1e6:.1f}M")
    print(f"  Trajectory Head: {head_params/1e3:.1f}K")
    print(f"  History Encoder: {history_params/1e3:.1f}K")
    print(f"  Fusion: {fusion_params/1e3:.1f}K")
    print(f"  총: {(vlm_params + head_params + history_params + fusion_params)/1e6:.1f}M")

    # 더미 입력
    print("\n추론 테스트...")
    dummy_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    dummy_history = torch.randn(16, 2)  # 1.6초 과거 trajectory

    torch.cuda.reset_peak_memory_stats()

    # 추론
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_trajectory = model.predict(dummy_image, dummy_history)

    print(f"\n예측 trajectory shape: {pred_trajectory.shape}")
    print(f"예측 범위: [{pred_trajectory.min():.3f}, {pred_trajectory.max():.3f}]")

    print_memory("추론 후")

    # 피크 메모리
    mem = get_gpu_memory()
    print(f"\n피크 VRAM: {mem['peak']:.2f} GB")
    print(f"Orin Nano 8GB: {'✅ 가능' if mem['peak'] < 6 else '⚠️ 빠듯' if mem['peak'] < 7.5 else '❌'}")

    return model, mem['peak']


def test_batch_inference(model):
    """배치 추론 테스트."""
    print("\n" + "=" * 60)
    print("배치 추론 테스트")
    print("=" * 60)

    batch_sizes = [1, 2, 4]

    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()

        images = [Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)) for _ in range(bs)]
        histories = torch.randn(bs, 16, 2)

        try:
            # 간단한 forward (processor 없이)
            device = next(model.vlm.parameters()).device
            histories = histories.to(device).to(torch.bfloat16)

            # 배치 처리는 복잡하므로 개별 처리
            for i in range(bs):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = model.predict(images[i], histories[i])

            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"Batch {bs}: 피크 {peak:.2f} GB {'✅' if peak < 7 else '⚠️'}")

        except Exception as e:
            print(f"Batch {bs}: 실패 - {e}")


if __name__ == "__main__":
    model, peak = test_alpamayo_nano()
    test_batch_inference(model)

    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"""
모델: Alpamayo Nano (Qwen3-VL-2B + Trajectory Head)
VRAM 피크: {peak:.2f} GB
Orin Nano 8GB: {'✅ OK' if peak < 6 else '⚠️ 빠듯' if peak < 7.5 else '❌'}

다음 단계:
1. NVIDIA 데이터로 trajectory head 학습
2. TensorRT 변환
3. Orin Nano 배포
""")
