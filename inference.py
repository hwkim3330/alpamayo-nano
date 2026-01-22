#!/usr/bin/env python3
"""
Alpamayo Nano - Quick inference script.

Usage:
    python inference.py --image road.jpg
    python inference.py --image road.jpg --model dwko/Alpamayo-R1-10B-4bit
"""

import argparse
import time
import cv2
import numpy as np

from alpamayo_nano import AlpamayoNano, InferenceConfig


def visualize_trajectory(
    image: np.ndarray,
    trajectory: np.ndarray,
    output_path: str = "output.jpg"
) -> np.ndarray:
    """Draw trajectory on image."""
    h, w = image.shape[:2]
    result = image.copy()

    # Camera params
    fx = 500
    cx, cy = w // 2, h + 50
    camera_height = 0.35

    # Project and draw
    points = []
    for p in trajectory:
        x_forward, y_left = p[0], p[1]
        if x_forward < 0.01:
            continue
        u = int(cx - (y_left / x_forward) * fx)
        v = int(cy - (camera_height / x_forward) * fx)
        if 0 <= u < w and 0 <= v < h:
            points.append((u, v))

    # Draw path
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(result, points[i], points[i + 1], (50, 255, 50), 3)
            cv2.circle(result, points[i], 5, (50, 200, 255), -1)

    cv2.imwrite(output_path, result)
    print(f"Saved visualization to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Alpamayo Nano Inference")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--model", type=str, default="dwko/Alpamayo-R1-10B-4bit")
    parser.add_argument("--prompt", type=str, default="Please drive carefully on the road.")
    parser.add_argument("--output", type=str, default="output.jpg")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    print("=" * 60)
    print("  Alpamayo Nano - 4-bit Inference")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {args.model}")
    t0 = time.time()
    model = AlpamayoNano.from_pretrained(args.model)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print(f"VRAM: {model.get_vram_usage():.2f} GB")

    # Load image
    print(f"\nLoading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return

    # Inference
    print(f"\nRunning inference...")
    t0 = time.time()
    trajectory, reasoning = model.infer([image], args.prompt)
    inference_time = time.time() - t0

    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"\nTrajectory: {trajectory.shape[0]} waypoints")
    print(f"First 5 points (x, y, heading):")
    for i, pt in enumerate(trajectory[:5]):
        print(f"  [{i}] x={pt[0]:.3f} y={pt[1]:.3f} h={pt[2]:.3f}")

    print(f"\nReasoning (Chain-of-Causation):")
    print(f"  {reasoning[:300]}...")

    print(f"\n--- Performance ---")
    print(f"Inference time: {inference_time:.2f}s ({1/inference_time:.2f} FPS)")
    print(f"VRAM: {model.get_vram_usage():.2f} GB")
    print(f"Peak VRAM: {model.get_vram_peak():.2f} GB")

    # Visualization
    if not args.no_viz:
        visualize_trajectory(image, trajectory, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
