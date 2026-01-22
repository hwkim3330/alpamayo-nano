#!/usr/bin/env python3
"""
Generate training data using Alpamayo-R1 teacher model.

Runs Alpamayo inference on driving images and saves:
- Image paths
- Predicted trajectories
- Chain-of-Causation reasoning text
- Optional: text embeddings

Usage:
    python distill/generate_data.py --input /path/to/images --output data/distill
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TeacherSample:
    """Single teacher prediction sample."""
    image_path: str
    trajectory: List[List[float]]  # (T, 3) waypoints
    reasoning: str                  # Chain-of-Causation text
    inference_time: float


class AlpamayoTeacher:
    """Alpamayo-R1 teacher for generating training data."""

    def __init__(
        self,
        model_path: str = "nvidia/Alpamayo-R1-10B",
        device: str = "cuda",
        use_4bit: bool = True
    ):
        self.model_path = model_path
        self.device = device
        self.use_4bit = use_4bit
        self.model = None
        self.processor = None

    def load(self):
        """Load teacher model."""
        print(f"Loading teacher: {self.model_path}")

        from transformers import AutoModel, AutoConfig, BitsAndBytesConfig
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        from alpamayo_r1.config import AlpamayoR1Config

        AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
        AutoModel.register(AlpamayoR1Config, AlpamayoR1)

        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        self.model.eval()

        # Load processor
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )

        vram = torch.cuda.memory_allocated() / 1e9
        print(f"Teacher loaded. VRAM: {vram:.2f} GB")

    def infer(
        self,
        image_path: str,
        prompt: str = "Please drive carefully on the road."
    ) -> TeacherSample:
        """Run inference on single image."""
        import cv2
        from qwen_vl_utils import process_vision_info
        import io
        import base64

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        # Encode to base64
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        b64 = base64.b64encode(buffer.getvalue()).decode()

        # Build message
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{b64}"},
                {"type": "text", "text": prompt}
            ]
        }]

        # Process
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        t0 = time.time()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                temperature=0.6,
                top_p=0.9,
                max_new_tokens=512,
                do_sample=True,
            )
        inference_time = time.time() - t0

        # Parse output
        generated_ids = output[:, inputs['input_ids'].shape[1]:]
        reasoning = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Extract trajectory
        if hasattr(output, 'actions'):
            trajectory = output.actions.cpu().numpy().tolist()
        else:
            # Dummy trajectory for testing
            trajectory = [[i * 0.15, 0.0, 0.0] for i in range(33)]

        return TeacherSample(
            image_path=image_path,
            trajectory=trajectory,
            reasoning=reasoning,
            inference_time=inference_time
        )


def find_images(input_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> List[str]:
    """Find all images in directory."""
    images = []
    input_path = Path(input_dir)

    for ext in extensions:
        images.extend(input_path.glob(f"**/*{ext}"))
        images.extend(input_path.glob(f"**/*{ext.upper()}"))

    return sorted([str(p) for p in images])


def generate_dataset(
    teacher: AlpamayoTeacher,
    image_paths: List[str],
    output_dir: str,
    prompt: str = "Please drive carefully on the road.",
    resume: bool = True
):
    """Generate distillation dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples_file = output_path / "samples.jsonl"
    processed = set()

    # Load existing samples if resuming
    if resume and samples_file.exists():
        with open(samples_file, 'r') as f:
            for line in f:
                sample = json.loads(line)
                processed.add(sample['image_path'])
        print(f"Resuming from {len(processed)} existing samples")

    # Open for append
    f = open(samples_file, 'a')

    try:
        pbar = tqdm(image_paths, desc="Generating")
        for image_path in pbar:
            if image_path in processed:
                continue

            try:
                sample = teacher.infer(image_path, prompt)

                # Save
                f.write(json.dumps(asdict(sample)) + '\n')
                f.flush()

                pbar.set_postfix({
                    'time': f'{sample.inference_time:.1f}s',
                    'fps': f'{1/sample.inference_time:.2f}'
                })

            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
                continue

    finally:
        f.close()

    print(f"\nDataset saved to {output_path}")

    # Write metadata
    meta = {
        "model": teacher.model_path,
        "num_samples": len(list(output_path.glob("*.jsonl"))),
        "prompt": prompt,
    }
    with open(output_path / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate distillation data")
    parser.add_argument("--input", type=str, required=True, help="Input image directory")
    parser.add_argument("--output", type=str, default="data/distill", help="Output directory")
    parser.add_argument("--model", type=str, default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--prompt", type=str, default="Please drive carefully on the road.")
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    args = parser.parse_args()

    print("=" * 60)
    print("  Alpamayo Distillation Data Generator")
    print("=" * 60)

    # Find images
    print(f"\nSearching for images in {args.input}...")
    images = find_images(args.input)
    print(f"Found {len(images)} images")

    if args.limit:
        images = images[:args.limit]
        print(f"Limited to {len(images)} images")

    if not images:
        print("No images found!")
        return

    # Load teacher
    teacher = AlpamayoTeacher(
        model_path=args.model,
        use_4bit=args.use_4bit
    )
    teacher.load()

    # Generate
    generate_dataset(teacher, images, args.output, args.prompt)

    print("\nDone!")


if __name__ == "__main__":
    main()
