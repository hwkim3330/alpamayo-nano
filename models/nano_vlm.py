"""
Alpamayo Nano - Qwen3-VL-2B 기반 자율주행 VLM

Orin Nano 8GB 타겟으로 경량화된 Alpamayo.

구조:
- Base VLM: Qwen3-VL-2B
- Trajectory Head: MLP로 64 waypoints (x, y) 예측
- 입력: 카메라 이미지 + 과거 trajectory
- 출력: 미래 trajectory (6.4초, 10Hz)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig, AutoProcessor


@dataclass
class AlpamayoNanoConfig:
    """Alpamayo Nano 설정."""
    vlm_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    hidden_size: int = 2048  # Qwen3-VL-2B text hidden size
    num_waypoints: int = 64  # 6.4초 @ 10Hz
    waypoint_dim: int = 2    # x, y
    num_history_steps: int = 16  # 1.6초 과거
    trajectory_head_layers: int = 2
    dropout: float = 0.1


class TrajectoryHead(nn.Module):
    """Trajectory 예측 head."""

    def __init__(self, config: AlpamayoNanoConfig):
        super().__init__()
        self.config = config

        hidden = config.hidden_size
        output_dim = config.num_waypoints * config.waypoint_dim

        layers = []
        for i in range(config.trajectory_head_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ])
            else:
                layers.extend([
                    nn.Linear(hidden, hidden // 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ])
                hidden = hidden // 2

        layers.append(nn.Linear(hidden, output_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, hidden_size) VLM의 마지막 hidden state

        Returns:
            trajectory: (B, num_waypoints, 2) 예측된 waypoints
        """
        output = self.head(hidden_states)
        return output.view(-1, self.config.num_waypoints, self.config.waypoint_dim)


class HistoryEncoder(nn.Module):
    """과거 trajectory 인코더."""

    def __init__(self, config: AlpamayoNanoConfig):
        super().__init__()
        self.config = config

        # 과거 trajectory를 hidden_size로 인코딩
        input_dim = config.num_history_steps * config.waypoint_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, num_history_steps, 2) 과거 trajectory

        Returns:
            encoded: (B, hidden_size)
        """
        B = history.shape[0]
        history_flat = history.view(B, -1)
        return self.encoder(history_flat)


class AlpamayoNano(nn.Module):
    """
    Alpamayo Nano - 경량 자율주행 VLM

    Qwen3-VL-2B 기반으로 카메라 이미지에서 trajectory 예측.
    """

    def __init__(self, config: AlpamayoNanoConfig):
        super().__init__()
        self.config = config

        # Base VLM (나중에 로드)
        self.vlm = None
        self.processor = None

        # Trajectory components
        self.history_encoder = HistoryEncoder(config)
        self.trajectory_head = TrajectoryHead(config)

        # Fusion layer (VLM hidden + history)
        self.fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming 초기화."""
        for module in [self.history_encoder, self.trajectory_head, self.fusion]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def load_vlm(self, model_path: str = None, quantization_config=None):
        """VLM 로드."""
        model_path = model_path or self.config.vlm_model_name

        print(f"VLM 로드: {model_path}")

        if quantization_config:
            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        self.processor = AutoProcessor.from_pretrained(model_path)

        # VLM freeze (파인튜닝 시 일부만 학습)
        for param in self.vlm.parameters():
            param.requires_grad = False

        print("VLM 로드 완료")

    def get_vlm_features(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """VLM에서 features 추출."""
        with torch.no_grad():
            outputs = self.vlm(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # 마지막 hidden state
        last_hidden = outputs.hidden_states[-1]
        # (B, seq_len, hidden) -> (B, hidden)

        # attention_mask가 있으면 그것을 사용하여 평균 계산
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            features = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        else:
            # 전체 시퀀스의 평균 사용 (마지막 토큰은 NaN일 수 있음)
            features = last_hidden.mean(dim=1)

        # NaN 방지
        features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)

        return features.to(torch.bfloat16)

    def forward(
        self,
        vlm_inputs: Dict[str, torch.Tensor],
        history_trajectory: torch.Tensor,
        future_trajectory: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            vlm_inputs: processor 출력 (input_ids, attention_mask, pixel_values, image_grid_thw 등)
            history_trajectory: (B, 16, 2) 과거 trajectory
            future_trajectory: (B, 64, 2) 미래 trajectory (학습 시)

        Returns:
            dict with 'pred_trajectory' and optionally 'loss'
        """
        # VLM features
        vlm_features = self.get_vlm_features(vlm_inputs)

        # History encoding
        history_features = self.history_encoder(history_trajectory)

        # Fusion
        fused = torch.cat([vlm_features, history_features], dim=-1)
        fused = self.fusion(fused)

        # Trajectory prediction
        pred_trajectory = self.trajectory_head(fused)

        outputs = {"pred_trajectory": pred_trajectory}

        # Loss 계산 (학습 시)
        if future_trajectory is not None:
            loss = nn.functional.mse_loss(pred_trajectory, future_trajectory)
            outputs["loss"] = loss

        return outputs

    def predict(
        self,
        image,
        history_trajectory: torch.Tensor,
        prompt: str = "Predict the vehicle trajectory for the next 6 seconds.",
    ) -> torch.Tensor:
        """
        추론용 간단한 인터페이스.

        Args:
            image: PIL Image 또는 numpy array
            history_trajectory: (16, 2) 과거 trajectory
            prompt: 텍스트 프롬프트

        Returns:
            trajectory: (64, 2) 예측된 waypoints
        """
        self.eval()

        # 이미지 처리
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # Device
        device = next(self.vlm.parameters()).device
        vlm_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        if history_trajectory.dim() == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        history_trajectory = history_trajectory.to(device).to(torch.bfloat16)

        # Forward
        with torch.no_grad():
            outputs = self.forward(
                vlm_inputs=vlm_inputs,
                history_trajectory=history_trajectory,
            )

        return outputs["pred_trajectory"].squeeze(0)


def create_alpamayo_nano(
    model_path: str = None,
    quantization_config=None,
) -> AlpamayoNano:
    """Alpamayo Nano 모델 생성."""
    config = AlpamayoNanoConfig()
    model = AlpamayoNano(config)
    model.load_vlm(model_path, quantization_config)

    # Trajectory head를 device로 이동
    device = next(model.vlm.parameters()).device
    model.history_encoder = model.history_encoder.to(device).to(torch.bfloat16)
    model.trajectory_head = model.trajectory_head.to(device).to(torch.bfloat16)
    model.fusion = model.fusion.to(device).to(torch.bfloat16)

    return model
