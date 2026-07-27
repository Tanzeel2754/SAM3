"""Render segmentation overlays without matplotlib."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.visualization_utils import COLORS, render_masklet_frame


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bool:
            return tensor.numpy()
        return tensor.float().numpy()
    return np.asarray(value)


def render_image_overlay(
    image: Image.Image,
    masks: List[Any],
    boxes: List[Any],
    scores: List[Any],
    prompted_boxes: Optional[List[Dict]] = None,
    alpha: float = 0.5,
) -> Image.Image:
    """Render SAM3 image segmentation results onto a PIL image."""
    base = np.array(image.convert("RGB"))
    overlay = base.copy().astype(np.float32)

    objects: List[Dict] = []
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        mask_np = _tensor_to_numpy(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        box_np = _tensor_to_numpy(box)
        score_val = float(_tensor_to_numpy(score).item() if hasattr(score, "item") else score)
        color = COLORS[i % len(COLORS)]
        color255 = (np.array(color) * 255).astype(np.uint8)
        mask_bool = mask_np > 0.5

        for c in range(3):
            overlay[..., c][mask_bool] = (
                alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]
            )

        x0, y0, x1, y1 = box_np.tolist()
        objects.append(
            {
                "index": i,
                "score": score_val,
                "box": [x0, y0, x1, y1],
                "color": [int(c * 255) for c in color],
            }
        )

    result = overlay.clip(0, 255).astype(np.uint8)

    for obj in objects:
        x0, y0, x1, y1 = [int(v) for v in obj["box"]]
        color = tuple(obj["color"])
        cv2.rectangle(result, (x0, y0), (x1, y1), color, 2)
        label = f"{obj['score']:.2f}"
        cv2.putText(
            result,
            label,
            (x0, max(y0 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            label,
            (x0, max(y0 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )

    if prompted_boxes:
        for pb in prompted_boxes:
            x0, y0, x1, y1 = [int(v) for v in pb["box"]]
            color = (0, 220, 80) if pb["label"] else (255, 70, 70)
            cv2.rectangle(result, (x0, y0), (x1, y1), color, 2)

    return Image.fromarray(result)


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def ndarray_to_base64_jpeg(arr: np.ndarray, quality: int = 90) -> str:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def render_video_frame_overlay(frame: np.ndarray, outputs: Dict) -> np.ndarray:
    return render_masklet_frame(frame, outputs, alpha=0.5)
