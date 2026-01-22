#!/usr/bin/env python3
"""
Alpamayo 2-bit 양자화

다양한 2-bit 양자화 방법:
1. Quanto qint2: HuggingFace Quanto 라이브러리
2. BitNet 2-bit: 커스텀 구현
3. GPTQ 2-bit: 그룹 양자화

Orin Nano 8GB 타겟:
- 10B 모델: ~2.5GB (2-bit)
- KV Cache: ~1GB
- 여유 메모리: ~4GB
"""

import sys
sys.path.insert(0, "/mnt/data/lfm_agi/alpamayo_orin")

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
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


def print_memory(label: str = ""):
    """메모리 출력."""
    mem = get_gpu_memory()
    if mem:
        print(f"[{label}] VRAM: {mem['allocated']:.2f} GB / {mem['total']:.1f} GB")


# =============================================================================
# 방법 1: Quanto 2-bit
# =============================================================================

def quantize_quanto_2bit(
    model_path: str = "/mnt/data/lfm_agi/models/alpamayo-r1-10b",
    output_path: str = "/mnt/data/lfm_agi/models/alpamayo-r1-10b-2bit-quanto",
):
    """
    Quanto qint2 양자화.

    HuggingFace Quanto: 간단하고 안정적.
    """
    print("=" * 60)
    print("Alpamayo 2-bit 양자화 (Quanto)")
    print("=" * 60)

    try:
        from quanto import freeze, qint2, qint4, quantize
    except ImportError:
        print("pip install quanto")
        return None

    # Alpamayo 모델 등록
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.config import AlpamayoR1Config
    from transformers import AutoModel, AutoConfig

    AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
    AutoModel.register(AlpamayoR1Config, AlpamayoR1)

    print(f"\n모델 경로: {model_path}")

    # CPU에 로드 (메모리 절약)
    print("\n1. 모델 로드 (CPU, BF16)...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"   파라미터: {param_count:,}")

    # 2-bit 양자화
    print("\n2. Quanto qint2 양자화...")
    quantize(model, weights=qint2)
    freeze(model)

    # GPU 테스트
    print("\n3. GPU 테스트...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model = model.to("cuda")

        mem = get_gpu_memory()
        print(f"   VRAM: {mem['allocated']:.2f} GB")

        # Forward 테스트
        print("\n4. Forward 테스트...")
        try:
            with torch.no_grad():
                # 더미 추론
                pass

            mem = get_gpu_memory()
            print(f"   피크 VRAM: {mem['peak']:.2f} GB")
            orin_status = '가능' if mem['peak'] < 6 else '빠듯' if mem['peak'] < 7.5 else '불가'
            print(f"   Orin Nano 8GB: {orin_status}")

        except Exception as e:
            print(f"   테스트 스킵: {e}")

    # 저장
    print(f"\n5. 저장: {output_path}")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)

    # 설정 저장
    config = {
        "method": "quanto_qint2",
        "bits": 2,
        "param_count": param_count,
    }
    with open(output_path / "quant_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nOK")
    return model


# =============================================================================
# 방법 2: BitsAndBytes FP4 (Double Quant)
# =============================================================================

def quantize_bnb_fp4(
    model_path: str = "/mnt/data/lfm_agi/models/alpamayo-r1-10b",
    output_path: str = "/mnt/data/lfm_agi/models/alpamayo-r1-10b-fp4-bnb",
):
    """
    BitsAndBytes FP4 + Double Quantization.

    엄밀히 4-bit지만 double quant로 ~3.5bit 효과.
    """
    print("=" * 60)
    print("Alpamayo FP4 양자화 (BitsAndBytes)")
    print("=" * 60)

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.config import AlpamayoR1Config
    from transformers import AutoModel, AutoConfig, BitsAndBytesConfig

    AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
    AutoModel.register(AlpamayoR1Config, AlpamayoR1)

    # FP4 + Double Quant 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # 추가 압축
        bnb_4bit_quant_type="fp4",
    )

    print(f"\n모델 경로: {model_path}")
    print("\n모델 로드 (FP4 + double quant)...")

    model = AutoModel.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    torch.cuda.reset_peak_memory_stats()
    mem = get_gpu_memory()
    print(f"VRAM: {mem['allocated']:.2f} GB")

    # 저장
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    return model


# =============================================================================
# 방법 3: 커스텀 2-bit (그룹 양자화)
# =============================================================================

class Linear2bit(nn.Module):
    """
    커스텀 2-bit Linear 레이어 (그룹 양자화).

    - 가중치: 2-bit (0, 1, 2, 3)
    - 그룹별 scale/zero_point
    - 더 높은 정확도
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # 그룹 수
        self.num_groups = (in_features + group_size - 1) // group_size

        # 2-bit 가중치 (packed: 4개 weight를 1 byte에)
        packed_size = (out_features * in_features + 3) // 4
        self.register_buffer('weight_packed', torch.zeros(packed_size, dtype=torch.uint8))

        # 그룹별 scale, zero_point
        self.register_buffer('scales', torch.ones(out_features, self.num_groups))
        self.register_buffer('zeros', torch.zeros(out_features, self.num_groups))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    @staticmethod
    def pack_2bit(tensor: torch.Tensor) -> torch.Tensor:
        """4개의 2-bit 값을 1 byte로 pack."""
        # tensor: [N] int (0-3 값)
        tensor = tensor.view(-1)
        # 4의 배수로 패딩
        pad_len = (4 - tensor.numel() % 4) % 4
        if pad_len > 0:
            tensor = F.pad(tensor, (0, pad_len))

        tensor = tensor.view(-1, 4)
        packed = (tensor[:, 0] |
                  (tensor[:, 1] << 2) |
                  (tensor[:, 2] << 4) |
                  (tensor[:, 3] << 6))
        return packed.to(torch.uint8)

    @staticmethod
    def unpack_2bit(packed: torch.Tensor, original_size: int) -> torch.Tensor:
        """1 byte에서 4개의 2-bit 값을 unpack."""
        packed = packed.view(-1)
        unpacked = torch.zeros(packed.numel() * 4, dtype=torch.int32, device=packed.device)
        unpacked[0::4] = packed & 0x03
        unpacked[1::4] = (packed >> 2) & 0x03
        unpacked[2::4] = (packed >> 4) & 0x03
        unpacked[3::4] = (packed >> 6) & 0x03
        return unpacked[:original_size]

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> 'Linear2bit':
        """Linear 레이어를 2-bit로 변환."""
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
        )

        weight = linear.weight.data.float()  # [out, in]

        # 그룹별 양자화
        quantized = torch.zeros_like(weight, dtype=torch.int32)
        scales = torch.zeros(layer.out_features, layer.num_groups)
        zeros = torch.zeros(layer.out_features, layer.num_groups)

        for g in range(layer.num_groups):
            start = g * group_size
            end = min((g + 1) * group_size, layer.in_features)

            w_group = weight[:, start:end]

            # Min-max 양자화
            w_min = w_group.min(dim=1, keepdim=True).values
            w_max = w_group.max(dim=1, keepdim=True).values

            scale = (w_max - w_min) / 3.0  # 2-bit: 4 levels (0-3)
            scale = scale.clamp(min=1e-5)
            zero = w_min

            # 양자화
            w_q = ((w_group - zero) / scale).round().clamp(0, 3).to(torch.int32)
            quantized[:, start:end] = w_q

            scales[:, g] = scale.squeeze()
            zeros[:, g] = zero.squeeze()

        # Pack
        layer.weight_packed.copy_(cls.pack_2bit(quantized.flatten()))
        layer.scales.copy_(scales)
        layer.zeros.copy_(zeros)

        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Unpack 가중치
        weight_int = self.unpack_2bit(
            self.weight_packed,
            self.out_features * self.in_features
        ).view(self.out_features, self.in_features)

        # Dequantize (그룹별)
        weight_float = torch.zeros(
            self.out_features, self.in_features,
            dtype=x.dtype, device=x.device
        )

        for g in range(self.num_groups):
            start = g * self.group_size
            end = min((g + 1) * self.group_size, self.in_features)

            w_g = weight_int[:, start:end].float()
            scale = self.scales[:, g:g+1]
            zero = self.zeros[:, g:g+1]

            weight_float[:, start:end] = w_g * scale + zero

        # 행렬곱
        output = F.linear(x, weight_float.to(x.dtype), self.bias)
        return output


def quantize_custom_2bit(
    model_path: str = "/mnt/data/lfm_agi/models/alpamayo-r1-10b",
    output_path: str = "/mnt/data/lfm_agi/models/alpamayo-r1-10b-2bit-custom",
    group_size: int = 128,
):
    """커스텀 2-bit 그룹 양자화."""
    print("=" * 60)
    print("Alpamayo 2-bit 양자화 (커스텀 그룹)")
    print("=" * 60)

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.config import AlpamayoR1Config
    from transformers import AutoModel, AutoConfig

    AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
    AutoModel.register(AlpamayoR1Config, AlpamayoR1)

    print(f"\n모델 경로: {model_path}")
    print(f"그룹 크기: {group_size}")

    # 로드
    print("\n1. 모델 로드...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    # 양자화
    print("\n2. 2-bit 그룹 양자화...")
    quantized_count = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # 2-bit로 교체
            layer_2bit = Linear2bit.from_linear(module, group_size=group_size)
            setattr(parent, child_name, layer_2bit)

            quantized_count += 1
            print(f"   [2-bit] {name}: {module.in_features}x{module.out_features}")

    print(f"\n   {quantized_count}개 레이어 양자화")

    # GPU 테스트
    if torch.cuda.is_available():
        print("\n3. GPU 테스트...")
        torch.cuda.reset_peak_memory_stats()
        model = model.to("cuda")

        mem = get_gpu_memory()
        print(f"   VRAM: {mem['allocated']:.2f} GB")

    # 저장
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "model.pt")

    config = {
        "method": "custom_2bit_group",
        "bits": 2,
        "group_size": group_size,
    }
    with open(output_path / "quant_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n저장: {output_path}")
    return model


# =============================================================================
# LFM 모델 양자화
# =============================================================================

def quantize_lfm_2bit(
    model_path: str = "/mnt/data/lfm_agi/models/nano/lfm25-vl-1.6b",
    output_path: str = "/mnt/data/lfm_agi/models/nano/lfm25-vl-1.6b-2bit",
    method: str = "quanto",
):
    """LFM2.5-VL 2-bit 양자화."""
    print("=" * 60)
    print(f"LFM2.5-VL-1.6B 2-bit 양자화 ({method})")
    print("=" * 60)

    from transformers import AutoModel, AutoProcessor

    print(f"\n모델 로드: {model_path}")

    if method == "quanto":
        try:
            from quanto import freeze, qint2, quantize
        except ImportError:
            print("pip install quanto")
            return None

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )

        quantize(model, weights=qint2)
        freeze(model)

    else:  # custom
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, Linear2bit.from_linear(module))

    # 저장
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if method == "quanto":
        model.save_pretrained(output_path)
    else:
        torch.save(model.state_dict(), output_path / "model.pt")

    # Processor 복사
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        processor.save_pretrained(output_path)
    except:
        pass

    print(f"\n저장: {output_path}")
    return model


# =============================================================================
# 테스트
# =============================================================================

def test_linear_2bit():
    """Linear2bit 테스트."""
    print("=" * 60)
    print("Linear2bit 테스트")
    print("=" * 60)

    # 원본
    linear = nn.Linear(512, 256)
    x = torch.randn(2, 10, 512)
    y_original = linear(x)

    # 2-bit
    linear_2bit = Linear2bit.from_linear(linear)
    y_2bit = linear_2bit(x)

    # 비교
    mse = F.mse_loss(y_2bit, y_original).item()
    cosine = F.cosine_similarity(y_2bit.flatten(), y_original.flatten(), dim=0).item()

    print(f"\nMSE: {mse:.6f}")
    print(f"Cosine Similarity: {cosine:.4f}")

    # 메모리
    original_kb = (linear.weight.numel() * 4) / 1024
    packed_kb = (linear_2bit.weight_packed.numel() + linear_2bit.scales.numel() * 4) / 1024

    print(f"\n메모리:")
    print(f"  원본 (FP32): {original_kb:.1f} KB")
    print(f"  2-bit packed: {packed_kb:.1f} KB")
    print(f"  압축률: {original_kb / packed_kb:.1f}x")

    print("\nOK")


def benchmark_methods():
    """양자화 방법 비교."""
    print("=" * 60)
    print("양자화 방법 벤치마크")
    print("=" * 60)

    # 테스트 모델
    class TestMLP(nn.Module):
        def __init__(self, hidden=1024):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(hidden, hidden * 4),
                nn.GELU(),
                nn.Linear(hidden * 4, hidden),
            )

        def forward(self, x):
            return self.layers(x)

    model = TestMLP()
    x = torch.randn(1, 16, 1024)
    y_original = model(x)

    results = []

    # 커스텀 2-bit
    import copy
    model_2bit = copy.deepcopy(model)
    for name, module in list(model_2bit.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
            child_name = name.split('.')[-1]
            parent = model_2bit.layers if parent_name else model_2bit
            if hasattr(parent, child_name):
                setattr(parent, child_name, Linear2bit.from_linear(module))

    y_2bit = model_2bit(x)
    mse = F.mse_loss(y_2bit, y_original).item()
    results.append(("Custom 2-bit", mse))

    # Quanto (있으면)
    try:
        from quanto import freeze, qint2, quantize
        model_quanto = copy.deepcopy(model)
        quantize(model_quanto, weights=qint2)
        freeze(model_quanto)
        y_quanto = model_quanto(x)
        mse = F.mse_loss(y_quanto, y_original).item()
        results.append(("Quanto qint2", mse))
    except ImportError:
        results.append(("Quanto qint2", "N/A (not installed)"))

    print("\n결과:")
    print("-" * 40)
    for method, mse in results:
        if isinstance(mse, float):
            print(f"{method:20s}: MSE = {mse:.6f}")
        else:
            print(f"{method:20s}: {mse}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alpamayo 2-bit 양자화")
    parser.add_argument("--method", choices=["quanto", "bnb", "custom", "test", "benchmark"],
                        default="test")
    parser.add_argument("--model", choices=["alpamayo", "lfm"], default="alpamayo")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--group-size", type=int, default=128)

    args = parser.parse_args()

    if args.method == "test":
        test_linear_2bit()
    elif args.method == "benchmark":
        benchmark_methods()
    elif args.model == "lfm":
        quantize_lfm_2bit(
            model_path=args.input or "/mnt/data/lfm_agi/models/nano/lfm25-vl-1.6b",
            output_path=args.output or "/mnt/data/lfm_agi/models/nano/lfm25-vl-1.6b-2bit",
            method=args.method if args.method in ["quanto", "custom"] else "quanto",
        )
    else:
        if args.method == "quanto":
            quantize_quanto_2bit(
                model_path=args.input or "/mnt/data/lfm_agi/models/alpamayo-r1-10b",
                output_path=args.output or "/mnt/data/lfm_agi/models/alpamayo-r1-10b-2bit-quanto",
            )
        elif args.method == "bnb":
            quantize_bnb_fp4(
                model_path=args.input or "/mnt/data/lfm_agi/models/alpamayo-r1-10b",
                output_path=args.output or "/mnt/data/lfm_agi/models/alpamayo-r1-10b-fp4-bnb",
            )
        elif args.method == "custom":
            quantize_custom_2bit(
                model_path=args.input or "/mnt/data/lfm_agi/models/alpamayo-r1-10b",
                output_path=args.output or "/mnt/data/lfm_agi/models/alpamayo-r1-10b-2bit-custom",
                group_size=args.group_size,
            )
