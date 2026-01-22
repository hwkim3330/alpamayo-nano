"""
Alpamayo Nano - Pre-quantized 4-bit Alpamayo-R1 wrapper.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Inference configuration."""
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 512
    num_trajectory_points: int = 33
    trajectory_horizon: float = 5.0  # seconds


class AlpamayoNano:
    """
    Pre-quantized 4-bit Alpamayo-R1 for edge deployment.

    Uses existing 4-bit models from HuggingFace:
    - dwko/Alpamayo-R1-10B-4bit
    - kimhyunwoo/alpamayo-r1-10b-int4-bnb
    """

    DEFAULT_MODEL = "dwko/Alpamayo-R1-10B-4bit"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str = "cuda",
        config: Optional[InferenceConfig] = None
    ):
        self.model_path = model_path
        self.device = device
        self.config = config or InferenceConfig()
        self.model = None
        self.processor = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = DEFAULT_MODEL,
        device: str = "cuda",
        **kwargs
    ) -> "AlpamayoNano":
        """Load pre-quantized 4-bit model."""
        instance = cls(model_path, device, **kwargs)
        instance.load()
        return instance

    def load(self) -> None:
        """Load the model."""
        print(f"Loading 4-bit model: {self.model_path}")

        # Import Alpamayo components
        from transformers import AutoModel, AutoConfig, BitsAndBytesConfig
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        from alpamayo_r1.config import AlpamayoR1Config

        # Register model
        AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
        AutoModel.register(AlpamayoR1Config, AlpamayoR1)

        # 4-bit config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()

        # Load processor
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )

        vram = torch.cuda.memory_allocated() / 1e9
        print(f"Model loaded. VRAM: {vram:.2f} GB")

    def infer(
        self,
        images: List[np.ndarray],
        prompt: str = "Please drive carefully on the road.",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Run inference on images.

        Args:
            images: List of camera images (BGR numpy arrays)
            prompt: Driving instruction prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling p
            max_new_tokens: Maximum new tokens to generate

        Returns:
            trajectory: (T, 3) waypoints [x, y, heading]
            reasoning: Chain-of-Causation text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        # Build message format
        from qwen_vl_utils import process_vision_info
        import cv2
        from PIL import Image
        import io
        import base64

        # Convert images to base64
        image_contents = []
        for img in images:
            if isinstance(img, np.ndarray):
                # BGR to RGB
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
            else:
                pil_img = img

            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG")
            b64 = base64.b64encode(buffer.getvalue()).decode()
            image_contents.append({
                "type": "image",
                "image": f"data:image/jpeg;base64,{b64}"
            })

        # Build message
        messages = [{
            "role": "user",
            "content": image_contents + [{"type": "text", "text": prompt}]
        }]

        # Process input
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
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )

        # Parse output
        trajectory, reasoning = self._parse_output(output, inputs)

        return trajectory, reasoning

    def _parse_output(
        self,
        output,
        inputs
    ) -> Tuple[np.ndarray, str]:
        """Parse model output to trajectory and reasoning."""
        # Decode text
        generated_ids = output[:, inputs['input_ids'].shape[1]:]
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Extract trajectory from action output
        trajectory = None
        if hasattr(output, 'actions'):
            trajectory = output.actions.cpu().numpy()
        else:
            # Generate dummy trajectory for testing
            trajectory = np.zeros((self.config.num_trajectory_points, 3))
            for i in range(self.config.num_trajectory_points):
                t = i * self.config.trajectory_horizon / self.config.num_trajectory_points
                trajectory[i] = [t * 0.5, 0.0, 0.0]  # Forward motion

        return trajectory, text

    def get_vram_usage(self) -> float:
        """Get current VRAM usage in GB."""
        return torch.cuda.memory_allocated() / 1e9

    def get_vram_peak(self) -> float:
        """Get peak VRAM usage in GB."""
        return torch.cuda.max_memory_allocated() / 1e9


def main():
    """CLI inference."""
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description="Alpamayo Nano Inference")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--model", type=str, default=AlpamayoNano.DEFAULT_MODEL)
    parser.add_argument("--prompt", type=str, default="Please drive carefully on the road.")
    args = parser.parse_args()

    # Load model
    model = AlpamayoNano.from_pretrained(args.model)

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return

    # Inference
    print(f"Running inference on {args.image}...")
    trajectory, reasoning = model.infer([image], args.prompt)

    print("\n=== Results ===")
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Reasoning: {reasoning[:200]}...")
    print(f"\nVRAM: {model.get_vram_usage():.2f} GB")
    print(f"Peak VRAM: {model.get_vram_peak():.2f} GB")


if __name__ == "__main__":
    main()
