"""In-memory image segmentation sessions."""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from webapp.backend.config import UPLOAD_DIR
from webapp.backend.services.models import get_image_processor, inference_mode
from webapp.backend.services.render import pil_to_base64, render_image_overlay, _tensor_to_numpy


@dataclass
class ImageSession:
    session_id: str
    image_path: Path
    image: Image.Image
    state: dict
    prompted_boxes: List[dict] = field(default_factory=list)


_sessions: Dict[str, ImageSession] = {}


def create_session(image: Image.Image, filename: str = "upload.jpg") -> ImageSession:
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / "image" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    image_path = session_dir / filename
    image.save(image_path, format="JPEG", quality=95)

    processor = get_image_processor()
    with inference_mode():
        state = processor.set_image(image)
    session = ImageSession(
        session_id=session_id,
        image_path=image_path,
        image=image,
        state=state,
    )
    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> ImageSession:
    if session_id not in _sessions:
        raise KeyError(f"Image session {session_id} not found")
    return _sessions[session_id]


def delete_session(session_id: str) -> int:
    session = _sessions.pop(session_id, None)
    if session is not None:
        shutil.rmtree(session.image_path.parent, ignore_errors=True)
    return len(_sessions)


def delete_all_sessions() -> None:
    for session_id in list(_sessions.keys()):
        delete_session(session_id)


def session_count() -> int:
    return len(_sessions)


def apply_text_prompt(session_id: str, prompt: str) -> ImageSession:
    session = get_session(session_id)
    processor = get_image_processor()
    session.prompted_boxes.clear()
    with inference_mode():
        session.state = processor.set_text_prompt(prompt.strip(), session.state)
    return session


def apply_box_prompt(session_id: str, box: List[float], label: bool) -> ImageSession:
    session = get_session(session_id)
    processor = get_image_processor()
    with inference_mode():
        session.state = processor.add_geometric_prompt(box, label, session.state)

    img_w = session.state["original_width"]
    img_h = session.state["original_height"]
    cx, cy, w, h = box
    x_min = (cx - w / 2) * img_w
    y_min = (cy - h / 2) * img_h
    x_max = (cx + w / 2) * img_w
    y_max = (cy + h / 2) * img_h
    session.prompted_boxes.append({"box": [x_min, y_min, x_max, y_max], "label": label})
    return session


def set_confidence(session_id: str, threshold: float) -> ImageSession:
    session = get_session(session_id)
    processor = get_image_processor()
    with inference_mode():
        session.state = processor.set_confidence_threshold(threshold, session.state)
    return session


def reset_prompts(session_id: str) -> ImageSession:
    session = get_session(session_id)
    processor = get_image_processor()
    processor.reset_all_prompts(session.state)
    session.prompted_boxes.clear()
    return session


def session_result(session_id: str) -> dict:
    session = get_session(session_id)
    masks = session.state.get("masks", [])
    boxes = session.state.get("boxes", [])
    scores = session.state.get("scores", [])

    has_masks = len(masks) > 0
    overlay = render_image_overlay(
        session.image,
        masks,
        boxes,
        scores,
        prompted_boxes=session.prompted_boxes,
    )

    objects = []
    if has_masks:
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            box_np = _tensor_to_numpy(box).tolist()
            score_arr = _tensor_to_numpy(score)
            objects.append(
                {
                    "index": i,
                    "score": float(score_arr.item() if score_arr.ndim == 0 else score_arr.squeeze()),
                    "box": box_np,
                }
            )

    return {
        "session_id": session_id,
        "width": session.state["original_width"],
        "height": session.state["original_height"],
        "object_count": len(objects),
        "objects": objects,
        "has_masks": has_masks,
        "peak_score": session.state.get("peak_score"),
        "overlay_image": pil_to_base64(overlay),
        "source_image": pil_to_base64(session.image),
    }
