"""FastAPI application for SAM3 web UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

from webapp.backend.config import MAX_IMAGE_SIZE_MB, MAX_VIDEO_SIZE_MB, UPLOAD_DIR
from webapp.backend.services import image_sessions, video_sessions
from webapp.backend.services.models import device_info
from webapp.backend.services import resources

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

app = FastAPI(title="SAM3 Studio", version="1.0.0", description="Interactive SAM3 image & video segmentation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextPromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=256)


class BoxPromptRequest(BaseModel):
    box: List[float] = Field(..., min_length=4, max_length=4)
    label: bool = True


class ConfidenceRequest(BaseModel):
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class VideoTextPromptRequest(BaseModel):
    frame_index: int = 0
    text: str = Field(..., min_length=1, max_length=256)


class VideoPointPromptRequest(BaseModel):
    frame_index: int = 0
    points: List[List[float]]
    labels: List[int]
    clear_old_points: bool = True
    obj_id: Optional[int] = None


class VideoBoxPromptRequest(BaseModel):
    frame_index: int = 0
    boxes: List[List[float]]
    labels: List[int]
    clear_old_boxes: bool = True


@app.get("/api/health")
def health():
    return {"status": "ok", **device_info()}


@app.post("/api/models/release/image")
def release_image_model():
    return {"status": "ok", **resources.release_image(), **device_info()}


@app.post("/api/models/release/video")
def release_video_model():
    return {"status": "ok", **resources.release_video(), **device_info()}


# ── Image routes ──────────────────────────────────────────────────────


@app.post("/api/image/sessions")
async def create_image_session(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")

    data = await file.read()
    if len(data) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"Image exceeds {MAX_IMAGE_SIZE_MB}MB limit")

    try:
        image = Image.open(__import__("io").BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc

    session = image_sessions.create_session(image, filename=file.filename or "upload.jpg")
    result = image_sessions.session_result(session.session_id)
    return {
        "session_id": session.session_id,
        "width": result["width"],
        "height": result["height"],
        "source_image": result["source_image"],
    }


@app.post("/api/image/sessions/{session_id}/text-prompt")
def image_text_prompt(session_id: str, body: TextPromptRequest):
    try:
        image_sessions.apply_text_prompt(session_id, body.prompt)
        return image_sessions.session_result(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found") from None
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/image/sessions/{session_id}/box-prompt")
def image_box_prompt(session_id: str, body: BoxPromptRequest):
    try:
        image_sessions.apply_box_prompt(session_id, body.box, body.label)
        return image_sessions.session_result(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found") from None
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.put("/api/image/sessions/{session_id}/confidence")
def image_confidence(session_id: str, body: ConfidenceRequest):
    try:
        image_sessions.set_confidence(session_id, body.threshold)
        return image_sessions.session_result(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found") from None


@app.post("/api/image/sessions/{session_id}/reset")
def image_reset(session_id: str):
    try:
        image_sessions.reset_prompts(session_id)
        return image_sessions.session_result(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found") from None


@app.get("/api/image/sessions/{session_id}/result")
def image_result(session_id: str):
    try:
        return image_sessions.session_result(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found") from None


@app.delete("/api/image/sessions/{session_id}")
def image_delete(session_id: str):
    image_sessions.delete_session(session_id)
    resources.maybe_release_image_if_idle()
    return {"ok": True}


# ── Video routes ──────────────────────────────────────────────────────


@app.post("/api/video/sessions")
async def create_video_session(
    file: UploadFile = File(...),
    start_time: float | None = Form(None),
    end_time: float | None = Form(None),
):
    if not file.content_type or "video" not in file.content_type:
        raise HTTPException(400, "Please upload a video file (MP4, WebM, etc.)")

    if start_time is not None and start_time < 0:
        raise HTTPException(400, "start_time must be >= 0")
    if end_time is not None and end_time <= 0:
        raise HTTPException(400, "end_time must be > 0")
    if start_time is not None and end_time is not None and end_time <= start_time:
        raise HTTPException(400, "end_time must be greater than start_time")

    data = await file.read()
    if len(data) > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"Video exceeds {MAX_VIDEO_SIZE_MB}MB limit")

    session_dir = UPLOAD_DIR / "video" / "uploads"
    session_dir.mkdir(parents=True, exist_ok=True)
    video_path = session_dir / f"{__import__('uuid').uuid4().hex}.mp4"
    video_path.write_bytes(data)

    try:
        session = video_sessions.create_session_from_video(
            video_path, start_time=start_time, end_time=end_time
        )
    except ValueError as exc:
        video_path.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        video_path.unlink(missing_ok=True)
        raise HTTPException(500, str(exc)) from exc
    finally:
        video_path.unlink(missing_ok=True)

    return {
        "session_id": session.session_id,
        "frame_count": session.frame_count,
        "width": session.width,
        "height": session.height,
        "fps": session.fps,
        "trim_start": session.trim_start,
        "trim_end": session.trim_end,
    }


@app.post("/api/video/sessions/{session_id}/text-prompt")
def video_text_prompt(session_id: str, body: VideoTextPromptRequest):
    try:
        video_sessions.add_text_prompt(session_id, body.frame_index, body.text)
        return video_sessions.get_frame_result(session_id, body.frame_index)
    except KeyError:
        raise HTTPException(404, "Session not found") from None
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/video/sessions/{session_id}/point-prompt")
def video_point_prompt(session_id: str, body: VideoPointPromptRequest):
    try:
        video_sessions.add_point_prompt(
            session_id,
            body.frame_index,
            body.points,
            body.labels,
            clear_old_points=body.clear_old_points,
            obj_id=body.obj_id,
        )
        return video_sessions.get_frame_result(session_id, body.frame_index)
    except KeyError:
        raise HTTPException(404, "Session not found") from None
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/video/sessions/{session_id}/box-prompt")
def video_box_prompt(session_id: str, body: VideoBoxPromptRequest):
    try:
        video_sessions.add_box_prompt(
            session_id,
            body.frame_index,
            body.boxes,
            body.labels,
            clear_old_boxes=body.clear_old_boxes,
        )
        return video_sessions.get_frame_result(session_id, body.frame_index)
    except KeyError:
        raise HTTPException(404, "Session not found") from None
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/video/sessions/{session_id}/reset")
def video_reset(session_id: str):
    try:
        video_sessions.reset_session(session_id)
        return {"ok": True}
    except KeyError:
        raise HTTPException(404, "Session not found") from None


@app.get("/api/video/sessions/{session_id}/frames/{frame_idx}")
def video_frame(session_id: str, frame_idx: int):
    try:
        return video_sessions.get_frame_result(session_id, frame_idx)
    except KeyError:
        raise HTTPException(404, "Session not found") from None
    except IndexError:
        raise HTTPException(400, "Frame index out of range") from None


@app.get("/api/video/sessions/{session_id}/propagate")
def video_propagate(session_id: str):
    def event_stream():
        try:
            for update in video_sessions.propagate(session_id):
                yield f"data: {json.dumps(update)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except KeyError:
            yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/api/video/sessions/{session_id}")
def video_delete(session_id: str):
    video_sessions.delete_session(session_id)
    resources.maybe_release_video_if_idle()
    return {"ok": True}


# ── Static frontend ───────────────────────────────────────────────────

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")
