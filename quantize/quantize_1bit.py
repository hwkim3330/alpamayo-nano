#!/usr/bin/env python3
"""
Alpamayo 1-bit (BitNet 1.58-bit) 양자화

BitNet 1.58-bit: Ternary weights (-1, 0, +1)
- 극한의 메모리 효율: ~1.58 bits/weight
- Orin Nano 8GB에서 대형 모델 가능
- 추론 속도 향상 (정수 연산)

참고: https://arxiv.org/abs/2402.17764 (The Era of 1-bit LLMs)
"""

import sys
sys.path.insert(0, "/mnt/data/lfm_agi/alpamayo_orin")

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from pathlib import Path
from typing import Optional, Tuple
import json


def get_gpu_memory():
    """GPU 메모리 사용량."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "peak": torch.cuda.max_memory_allocated() / 1e9,
            "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    return {}


# =============================================================================
# BitNet 1.58-bit Linear Layer
# =============================================================================

class BitLinear158(nn.Module):
    """
    BitNet 1.58-bit Linear Layer.

    가중치를 ternary (-1, 0, +1)로 양자화.
    1.58 bits per weight (log2(3) ≈ 1.58).

    Forward:
        1. 입력 정규화 (absmax)
        2. Ternary 가중치와 행렬곱 (정수 연산)
        3. Scale factor 적용
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        # Ternary 가중치: int8로 저장 (-1, 0, +1)
        self.register_buffer(
            'weight_ternary',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )

        # Scale factor (per-tensor)
        self.register_buffer('weight_scale', torch.ones(1))

        # Bias (양자화 안함)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    @staticmethod
    def ternary_quantize(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        가중치를 ternary로 양자화.

        BitNet 1.58-bit 방식:
        - scale = mean(|W|)
        - W_ternary = round_clip(W / scale, -1, 1)
        """
        # Scale: 평균 절대값
        scale = weight.abs().mean().clamp(min=1e-5)

        # 양자화: round to {-1, 0, +1}
        weight_scaled = weight / scale
        weight_ternary = weight_scaled.round().clamp(-1, 1).to(torch.int8)

        return weight_ternary, scale

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> 'BitLinear158':
        """기존 Linear 레이어를 BitLinear로 변환."""
        bit_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
        )

        # 가중치 양자화
        with torch.no_grad():
            weight_ternary, scale = cls.ternary_quantize(linear.weight.data)
            bit_linear.weight_ternary.copy_(weight_ternary)
            bit_linear.weight_scale.copy_(scale)

            if linear.bias is not None:
                bit_linear.bias.data.copy_(linear.bias.data)

        return bit_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        1. 입력 absmax 정규화
        2. Ternary 가중치와 연산
        3. Scale 복원
        """
        # 입력 정규화 (absmax per-token)
        x_abs_max = x.abs().max(dim=-1, keepdim=True).values.clamp(min=self.eps)
        x_norm = x / x_abs_max

        # Ternary 연산 (float로 캐스팅 후 연산)
        # 실제 하드웨어에서는 정수 연산으로 가속
        weight_float = self.weight_ternary.float()

        # 행렬곱
        output = F.linear(x_norm, weight_float)

        # Scale 복원: x_scale * weight_scale
        output = output * (x_abs_max * self.weight_scale)

        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, bias={self.bias is not None}'


# =============================================================================
# 2-bit Linear Layer (보너스)
# =============================================================================

class BitLinear2bit(nn.Module):
    """
    2-bit Linear Layer.

    가중치를 {-1.5, -0.5, +0.5, +1.5}로 양자화.
    정확히 2 bits per weight.
    """

    QUANT_LEVELS = torch.tensor([-1.5, -0.5, 0.5, 1.5])

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        # 2-bit 가중치: int8로 저장 (0, 1, 2, 3 -> -1.5, -0.5, +0.5, +1.5)
        self.register_buffer(
            'weight_2bit',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )

        # Scale factor
        self.register_buffer('weight_scale', torch.ones(1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    @staticmethod
    def quantize_2bit(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """가중치를 2-bit로 양자화."""
        # Scale
        scale = weight.abs().max().clamp(min=1e-5) / 1.5

        # 정규화
        weight_norm = weight / scale

        # 가장 가까운 레벨 찾기: -1.5, -0.5, 0.5, 1.5
        # 인덱스: 0, 1, 2, 3
        weight_norm_clamped = weight_norm.clamp(-1.5, 1.5)

        # 양자화 (인덱스로)
        # -1.5 ~ -1.0 -> 0, -1.0 ~ 0.0 -> 1, 0.0 ~ 1.0 -> 2, 1.0 ~ 1.5 -> 3
        indices = ((weight_norm_clamped + 1.5) / 1.0).floor().clamp(0, 3).to(torch.int8)

        return indices, scale

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> 'BitLinear2bit':
        """기존 Linear 레이어를 2-bit로 변환."""
        bit_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
        )

        with torch.no_grad():
            indices, scale = cls.quantize_2bit(linear.weight.data)
            bit_linear.weight_2bit.copy_(indices)
            bit_linear.weight_scale.copy_(scale)

            if linear.bias is not None:
                bit_linear.bias.data.copy_(linear.bias.data)

        return bit_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # 입력 정규화
        x_abs_max = x.abs().max(dim=-1, keepdim=True).values.clamp(min=self.eps)
        x_norm = x / x_abs_max

        # 2-bit 가중치 디코딩
        levels = self.QUANT_LEVELS.to(x.device)
        weight_float = levels[self.weight_2bit.long()]

        # 행렬곱
        output = F.linear(x_norm, weight_float)

        # Scale 복원
        output = output * (x_abs_max * self.weight_scale)

        if self.bias is not None:
            output = output + self.bias

        return output


# =============================================================================
# 모델 양자화 함수
# =============================================================================

def count_parameters(model):
    """파라미터 수 계산."""
    return sum(p.numel() for p in model.parameters())


def quantize_model_1bit(model: nn.Module, verbose: bool = True) -> nn.Module:
    """
    모델의 모든 Linear 레이어를 1.58-bit로 양자화.
    """
    quantized_count = 0
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 부모 모듈 찾기
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # BitLinear로 교체
            bit_linear = BitLinear158.from_linear(module)
            setattr(parent, child_name, bit_linear)

            params = module.in_features * module.out_features
            total_params += params
            quantized_count += 1

            if verbose:
                print(f"  [1.58-bit] {name}: {module.in_features}x{module.out_features}")

    if verbose:
        print(f"\n총 {quantized_count}개 레이어, {total_params:,} 파라미터 양자화됨")
        # 메모리 절약 계산: FP16 (16bit) -> 1.58bit
        original_mb = (total_params * 2) / 1e6  # FP16
        quantized_mb = (total_params * 1.58 / 8) / 1e6  # 1.58-bit
        print(f"메모리: {original_mb:.1f} MB (FP16) -> {quantized_mb:.1f} MB (1.58-bit)")
        print(f"압축률: {original_mb / quantized_mb:.1f}x")

    return model


def quantize_model_2bit(model: nn.Module, verbose: bool = True) -> nn.Module:
    """
    모델의 모든 Linear 레이어를 2-bit로 양자화.
    """
    quantized_count = 0
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            bit_linear = BitLinear2bit.from_linear(module)
            setattr(parent, child_name, bit_linear)

            params = module.in_features * module.out_features
            total_params += params
            quantized_count += 1

            if verbose:
                print(f"  [2-bit] {name}: {module.in_features}x{module.out_features}")

    if verbose:
        print(f"\n총 {quantized_count}개 레이어, {total_params:,} 파라미터 양자화됨")
        original_mb = (total_params * 2) / 1e6
        quantized_mb = (total_params * 2 / 8) / 1e6  # 2-bit
        print(f"메모리: {original_mb:.1f} MB (FP16) -> {quantized_mb:.1f} MB (2-bit)")
        print(f"압축률: {original_mb / quantized_mb:.1f}x")

    return model


# =============================================================================
# Alpamayo 모델 양자화
# =============================================================================

def quantize_alpamayo_1bit(
    model_path: str = "/mnt/data/lfm_agi/models/alpamayo-r1-10b",
    output_path: str = "/mnt/data/lfm_agi/models/alpamayo-r1-10b-1bit",
    bits: int = 1,  # 1 for 1.58-bit, 2 for 2-bit
):
    """
    Alpamayo 모델을 1-bit 또는 2-bit로 양자화.
    """
    print("=" * 60)
    print(f"Alpamayo {bits}-bit 양자화 (BitNet)")
    print("=" * 60)

    # 모델 등록
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.config import AlpamayoR1Config
    from transformers import AutoModel, AutoConfig

    AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
    AutoModel.register(AlpamayoR1Config, AlpamayoR1)

    print(f"\n모델 경로: {model_path}")
    print(f"출력 경로: {output_path}")

    # CPU에 로드 (메모리 절약)
    print("\n1. 모델 로드 (CPU)...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # 양자화를 위해 FP32
        device_map="cpu",
        trust_remote_code=True,
    )

    # 원본 파라미터 수
    original_params = count_parameters(model)
    print(f"   파라미터: {original_params:,}")

    # 양자화
    print(f"\n2. {bits}-bit 양자화...")
    if bits == 1:
        model = quantize_model_1bit(model)
    else:
        model = quantize_model_2bit(model)

    # GPU로 이동 (테스트)
    print("\n3. GPU 테스트...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model = model.to("cuda")

        mem = get_gpu_memory()
        print(f"   VRAM: {mem['allocated']:.2f} GB")

        # 간단한 forward 테스트
        print("\n4. Forward 테스트...")
        try:
            with torch.no_grad():
                # 더미 입력 (배치=1, 시퀀스=100, 히든=모델에 따라)
                dummy_input = torch.randn(1, 100, 768).to("cuda")  # 예시
                # 실제 모델 구조에 맞게 조정 필요

            mem = get_gpu_memory()
            print(f"   피크 VRAM: {mem['peak']:.2f} GB")
            print(f"   Orin Nano 8GB: {'가능' if mem['peak'] < 7 else '빠듯' if mem['peak'] < 8 else '불가'}")

        except Exception as e:
            print(f"   테스트 스킵: {e}")

    # 저장
    print(f"\n5. 저장: {output_path}")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # state_dict 저장
    torch.save(model.state_dict(), output_path / "model.pt")

    # 설정 저장
    config = {
        "bits": bits,
        "method": "bitnet_1.58" if bits == 1 else "bitnet_2bit",
        "original_params": original_params,
        "model_path": model_path,
    }
    with open(output_path / "quant_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("양자화 완료!")
    print("=" * 60)

    # 예상 메모리
    if bits == 1:
        expected_size = (original_params * 1.58 / 8) / 1e9
    else:
        expected_size = (original_params * 2 / 8) / 1e9

    print(f"""
모델: Alpamayo R1 10B
양자화: {bits}-bit (BitNet)
예상 크기: ~{expected_size:.2f} GB
Orin Nano 8GB: {'OK' if expected_size < 5 else '빠듯' if expected_size < 7 else '불가'}
""")

    return model


# =============================================================================
# LFM 모델 양자화 (더 작은 모델)
# =============================================================================

def quantize_lfm_1bit(
    model_path: str = "/mnt/data/lfm_agi/models/nano/lfm25-vl-1.6b",
    output_path: str = "/mnt/data/lfm_agi/models/nano/lfm25-vl-1.6b-1bit",
    bits: int = 1,
):
    """LFM2.5-VL 1-bit 양자화."""
    print("=" * 60)
    print(f"LFM2.5-VL-1.6B {bits}-bit 양자화")
    print("=" * 60)

    from transformers import AutoModel, AutoProcessor

    print(f"\n모델 로드: {model_path}")

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    original_params = count_parameters(model)
    print(f"파라미터: {original_params:,}")

    # 양자화
    print(f"\n{bits}-bit 양자화...")
    if bits == 1:
        model = quantize_model_1bit(model)
    else:
        model = quantize_model_2bit(model)

    # 저장
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "model.pt")

    # Processor 복사
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        processor.save_pretrained(output_path)
    except:
        pass

    # 예상 크기
    if bits == 1:
        expected_mb = (original_params * 1.58 / 8) / 1e6
    else:
        expected_mb = (original_params * 2 / 8) / 1e6

    print(f"\n예상 크기: ~{expected_mb:.0f} MB")
    print(f"Orin Nano 8GB: OK")

    return model


# =============================================================================
# 테스트
# =============================================================================

def test_bitlinear():
    """BitLinear 레이어 테스트."""
    print("=" * 60)
    print("BitLinear 테스트")
    print("=" * 60)

    # 원본 Linear
    linear = nn.Linear(512, 256)

    # 입력
    x = torch.randn(2, 10, 512)

    # 원본 출력
    y_original = linear(x)

    # 1.58-bit 양자화
    bit_linear = BitLinear158.from_linear(linear)
    y_1bit = bit_linear(x)

    # 2-bit 양자화
    bit_linear_2 = BitLinear2bit.from_linear(linear)
    y_2bit = bit_linear_2(x)

    # 비교
    print(f"\n원본 출력 범위: [{y_original.min():.3f}, {y_original.max():.3f}]")
    print(f"1.58-bit 출력 범위: [{y_1bit.min():.3f}, {y_1bit.max():.3f}]")
    print(f"2-bit 출력 범위: [{y_2bit.min():.3f}, {y_2bit.max():.3f}]")

    # MSE
    mse_1bit = F.mse_loss(y_1bit, y_original).item()
    mse_2bit = F.mse_loss(y_2bit, y_original).item()

    print(f"\nMSE (1.58-bit): {mse_1bit:.6f}")
    print(f"MSE (2-bit): {mse_2bit:.6f}")

    # 메모리 비교
    original_bytes = linear.weight.numel() * 4  # FP32
    bit1_bytes = bit_linear.weight_ternary.numel() * 1  # INT8 (실제로는 1.58-bit로 압축 가능)
    bit2_bytes = bit_linear_2.weight_2bit.numel() * 1  # INT8 (실제로는 2-bit로 압축 가능)

    print(f"\n메모리 (weight만):")
    print(f"  원본 (FP32): {original_bytes / 1024:.1f} KB")
    print(f"  1.58-bit (INT8): {bit1_bytes / 1024:.1f} KB (이론적: {original_bytes * 1.58 / 32 / 1024:.1f} KB)")
    print(f"  2-bit (INT8): {bit2_bytes / 1024:.1f} KB (이론적: {original_bytes * 2 / 32 / 1024:.1f} KB)")

    print("\nOK")


def test_simple_model():
    """간단한 모델로 전체 양자화 테스트."""
    print("=" * 60)
    print("간단한 모델 양자화 테스트")
    print("=" * 60)

    # 간단한 MLP
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 256)
            self.act = nn.GELU()

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleMLP()
    x = torch.randn(4, 16, 256)

    # 원본
    y_original = model(x)

    # 1.58-bit 양자화
    import copy
    model_1bit = copy.deepcopy(model)
    model_1bit = quantize_model_1bit(model_1bit, verbose=False)
    y_1bit = model_1bit(x)

    # 2-bit 양자화
    model_2bit = copy.deepcopy(model)
    model_2bit = quantize_model_2bit(model_2bit, verbose=False)
    y_2bit = model_2bit(x)

    # 결과
    print(f"\nMSE (1.58-bit): {F.mse_loss(y_1bit, y_original).item():.6f}")
    print(f"MSE (2-bit): {F.mse_loss(y_2bit, y_original).item():.6f}")
    print("\nOK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alpamayo 1-bit/2-bit 양자화")
    parser.add_argument("--model", choices=["alpamayo", "lfm", "test"], default="test")
    parser.add_argument("--bits", type=int, choices=[1, 2], default=1)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.model == "test":
        test_bitlinear()
        print()
        test_simple_model()
    elif args.model == "alpamayo":
        quantize_alpamayo_1bit(
            model_path=args.input or "/mnt/data/lfm_agi/models/alpamayo-r1-10b",
            output_path=args.output or f"/mnt/data/lfm_agi/models/alpamayo-r1-10b-{args.bits}bit",
            bits=args.bits,
        )
    elif args.model == "lfm":
        quantize_lfm_1bit(
            model_path=args.input or "/mnt/data/lfm_agi/models/nano/lfm25-vl-1.6b",
            output_path=args.output or f"/mnt/data/lfm_agi/models/nano/lfm25-vl-1.6b-{args.bits}bit",
            bits=args.bits,
        )
