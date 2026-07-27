"""Video tracking sessions backed by SAM 3.1 multiplex predictor."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional

import cv2
import numpy as np

from webapp.backend.config import UPLOAD_DIR
from webapp.backend.services.models import get_video_predictor, inference_mode
from webapp.backend.services.render import ndarray_to_base64_jpeg, render_video_frame_overlay
from sam3.visualization_utils import load_frame


@dataclass
class VideoSession:
    session_id: str
    predictor_session_id: str
    frames_dir: Path
    frame_count: int
    width: int
    height: int
    fps: float
    trim_start: float | None = None
    trim_end: float | None = None
    frame_outputs: Dict[int, dict] = field(default_factory=dict)


_sessions: Dict[str, VideoSession] = {}


def _extract_frames(
    video_path: Path,
    output_dir: Path,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[int, int, int, float, float | None, float | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    trim_start = 0.0 if start_time is None else max(0.0, start_time)
    if end_time is not None:
        trim_end = max(trim_start, end_time)
    elif total_frames > 0:
        trim_end = total_frames / fps
    else:
        trim_end = None

    if trim_end is not None and trim_end <= trim_start:
        cap.release()
        raise ValueError("Trim end must be after trim start")

    start_frame = int(trim_start * fps)
    if trim_end is not None:
        end_frame = int(trim_end * fps)
        if total_frames > 0:
            end_frame = min(end_frame, total_frames)
    else:
        end_frame = total_frames if total_frames > 0 else None

    if end_frame is not None and end_frame <= start_frame:
        cap.release()
        raise ValueError("Selected trim range contains no frames")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = 0
    abs_frame = start_frame
    while True:
        if end_frame is not None and abs_frame >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_path = output_dir / f"{frame_idx}.jpg"
        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        frame_idx += 1
        abs_frame += 1

    cap.release()
    if frame_idx == 0:
        raise ValueError("Video contains no frames in the selected range")
    applied_end = trim_end if trim_end is not None else (abs_frame / fps)
    return frame_idx, width, height, fps, trim_start, applied_end


def create_session_from_video(
    video_path: Path,
    start_time: float | None = None,
    end_time: float | None = None,
) -> VideoSession:
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / "video" / session_id
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_count, width, height, fps, trim_start, trim_end = _extract_frames(
        video_path, frames_dir, start_time=start_time, end_time=end_time
    )

    predictor = get_video_predictor()
    with inference_mode():
        response = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(frames_dir),
                "session_id": session_id,
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": False,
            }
        )

    session = VideoSession(
        session_id=session_id,
        predictor_session_id=response["session_id"],
        frames_dir=frames_dir,
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
        trim_start=trim_start,
        trim_end=trim_end,
    )
    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> VideoSession:
    if session_id not in _sessions:
        raise KeyError(f"Video session {session_id} not found")
    return _sessions[session_id]


def delete_session(session_id: str) -> int:
    session = _sessions.pop(session_id, None)
    if session is not None:
        try:
            predictor = get_video_predictor()
            predictor.handle_request({"type": "close_session", "session_id": session.predictor_session_id})
        except Exception:
            pass
        shutil.rmtree(session.frames_dir.parent, ignore_errors=True)
    return len(_sessions)


def delete_all_sessions() -> None:
    for session_id in list(_sessions.keys()):
        delete_session(session_id)


def session_count() -> int:
    return len(_sessions)


def add_text_prompt(session_id: str, frame_idx: int, text: str) -> dict:
    session = get_session(session_id)
    predictor = get_video_predictor()
    with inference_mode():
        response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session.predictor_session_id,
                "frame_index": frame_idx,
                "text": text.strip(),
            }
        )
    session.frame_outputs[response["frame_index"]] = response["outputs"]
    return response


def add_point_prompt(
    session_id: str,
    frame_idx: int,
    points: List[List[float]],
    labels: List[int],
    clear_old_points: bool = True,
    obj_id: Optional[int] = None,
) -> dict:
    session = get_session(session_id)
    predictor = get_video_predictor()
    request = {
        "type": "add_prompt",
        "session_id": session.predictor_session_id,
        "frame_index": frame_idx,
        "points": points,
        "point_labels": labels,
        "clear_old_points": clear_old_points,
        "rel_coordinates": True,
    }
    if obj_id is not None:
        request["obj_id"] = obj_id
    with inference_mode():
        response = predictor.handle_request(request)
    session.frame_outputs[response["frame_index"]] = response["outputs"]
    return response


def add_box_prompt(
    session_id: str,
    frame_idx: int,
    boxes: List[List[float]],
    labels: List[int],
    clear_old_boxes: bool = True,
) -> dict:
    session = get_session(session_id)
    predictor = get_video_predictor()
    with inference_mode():
        response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session.predictor_session_id,
                "frame_index": frame_idx,
                "bounding_boxes": boxes,
                "bounding_box_labels": labels,
                "clear_old_boxes": clear_old_boxes,
                "rel_coordinates": True,
            }
        )
    session.frame_outputs[response["frame_index"]] = response["outputs"]
    return response


def reset_session(session_id: str) -> None:
    session = get_session(session_id)
    predictor = get_video_predictor()
    predictor.handle_request({"type": "reset_session", "session_id": session.predictor_session_id})
    session.frame_outputs.clear()


def propagate(session_id: str) -> Generator[dict, None, None]:
    session = get_session(session_id)
    predictor = get_video_predictor()
    with inference_mode():
        for response in predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session.predictor_session_id}
        ):
            frame_idx = response["frame_index"]
            session.frame_outputs[frame_idx] = response["outputs"]
            yield {
                "frame_index": frame_idx,
                "object_count": len(response["outputs"].get("out_obj_ids", [])),
            }


def get_frame_image(session_id: str, frame_idx: int) -> np.ndarray:
    session = get_session(session_id)
    if frame_idx < 0 or frame_idx >= session.frame_count:
        raise IndexError("Frame index out of range")
    frame_path = session.frames_dir / f"{frame_idx}.jpg"
    return load_frame(str(frame_path))


def get_frame_result(session_id: str, frame_idx: int) -> dict:
    session = get_session(session_id)
    frame = get_frame_image(session_id, frame_idx)
    outputs = session.frame_outputs.get(frame_idx)

    result = {
        "frame_index": frame_idx,
        "frame_count": session.frame_count,
        "has_output": outputs is not None,
        "source_image": ndarray_to_base64_jpeg(frame),
    }

    if outputs is not None and len(outputs.get("out_obj_ids", [])) > 0:
        overlay = render_video_frame_overlay(frame, outputs)
        result["overlay_image"] = ndarray_to_base64_jpeg(overlay)
        result["object_count"] = len(outputs["out_obj_ids"])
        result["objects"] = [
            {
                "obj_id": int(outputs["out_obj_ids"][i]),
                "score": float(outputs["out_probs"][i]) if "out_probs" in outputs else None,
            }
            for i in range(len(outputs["out_obj_ids"]))
        ]
    else:
        result["object_count"] = 0
        result["objects"] = []

    return result
