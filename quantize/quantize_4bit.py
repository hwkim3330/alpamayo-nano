#!/usr/bin/env python3
"""
LFM2.5-VL INT4 양자화 및 메모리 테스트

Orin Nano 8GB 타겟
"""

import os
import gc
import torch
from pathlib import Path
from transformers import AutoModel, AutoModelForVision2Seq, AutoTokenizer, AutoProcessor, BitsAndBytesConfig


def get_gpu_memory():
    """GPU 메모리 사용량."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
            "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    return {}


def print_memory(label=""):
    mem = get_gpu_memory()
    if mem:
        print(f"[{label}] VRAM: {mem['allocated']:.2f}GB / {mem['total']:.1f}GB")


def quantize_lfm_int4(
    model_path: str = "/mnt/data/lfm_agi/models/nano/lfm25-vl-1.6b",
    output_path: str = "/mnt/data/lfm_agi/models/nano/lfm25-vl-1.6b-int4",
):
    """LFM2.5-VL INT4 양자화."""
    print("=" * 60)
    print("LFM2.5-VL-1.6B INT4 양자화")
    print("=" * 60)

    print_memory("시작")

    # INT4 NF4 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"\n모델 로드: {model_path}")

    # 모델 로드 (VLM)
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Vision2Seq 실패: {e}, AutoModel 시도...")
        model = AutoModel.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    model.eval()

    print_memory("모델 로드 후")

    # 프로세서/토크나이저
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except:
        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 저장
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n저장: {output_path}")
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    # 결과
    orig_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors")) / 1e9
    quant_size = sum(f.stat().st_size for f in output_path.glob("*.safetensors")) / 1e9

    print("\n" + "=" * 60)
    print("양자화 완료!")
    print("=" * 60)
    print(f"원본: {orig_size:.2f} GB")
    print(f"INT4: {quant_size:.2f} GB")
    print(f"압축률: {orig_size/quant_size:.1f}x" if quant_size > 0 else "")

    return model, processor


def test_inference(model, processor):
    """간단한 추론 테스트."""
    print("\n" + "=" * 60)
    print("추론 테스트")
    print("=" * 60)

    import numpy as np
    from PIL import Image

    # 더미 이미지
    dummy_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    # 입력 준비
    prompt = "Describe this driving scene and predict the vehicle trajectory."

    try:
        # VLM 입력
        inputs = processor(
            text=prompt,
            images=dummy_image,
            return_tensors="pt",
        ).to("cuda")

        print_memory("입력 준비 후")

        # 추론
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )

        # 디코딩
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"\n응답: {response[:200]}...")

    except Exception as e:
        print(f"추론 오류: {e}")
        # 간단한 텍스트만 테스트
        try:
            inputs = processor(text=prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=20)
            print(f"텍스트 응답: {processor.decode(outputs[0])}")
        except Exception as e2:
            print(f"텍스트도 실패: {e2}")

    print_memory("추론 후")

    # 피크 메모리
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n피크 VRAM: {peak:.2f} GB")
    print(f"Orin Nano 8GB: {'✅ 가능' if peak < 6 else '⚠️ 빠듯' if peak < 7.5 else '❌ 불가'}")

    return peak


if __name__ == "__main__":
    model, processor = quantize_lfm_int4()
    peak = test_inference(model, processor)

    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    mem = get_gpu_memory()
    print(f"""
모델: LFM2.5-VL-1.6B INT4
VRAM 사용: {mem['allocated']:.2f} GB
피크 VRAM: {peak:.2f} GB
Orin Nano 8GB: {'✅ OK' if peak < 6 else '⚠️'}
""")
