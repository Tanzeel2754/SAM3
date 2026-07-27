"""Release GPU-backed model resources when a mode is no longer active."""

from __future__ import annotations

from webapp.backend.services import image_sessions, video_sessions
from webapp.backend.services.models import (
    image_model_loaded,
    unload_image_model,
    unload_video_model,
    video_model_loaded,
)


def release_image() -> dict:
    """Close all image sessions and unload the image model from GPU."""
    image_sessions.delete_all_sessions()
    unloaded = unload_image_model()
    return {
        "released": unloaded or not image_model_loaded(),
        "image_model_loaded": image_model_loaded(),
        "video_model_loaded": video_model_loaded(),
    }


def release_video() -> dict:
    """Close all video sessions and unload the video model from GPU."""
    video_sessions.delete_all_sessions()
    unloaded = unload_video_model()
    return {
        "released": unloaded or not video_model_loaded(),
        "image_model_loaded": image_model_loaded(),
        "video_model_loaded": video_model_loaded(),
    }


def maybe_release_image_if_idle() -> bool:
    if image_sessions.session_count() == 0:
        return unload_image_model()
    return False


def maybe_release_video_if_idle() -> bool:
    if video_sessions.session_count() == 0:
        return unload_video_model()
    return False
