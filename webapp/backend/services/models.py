"""Lazy-loaded SAM3 model singletons."""

from __future__ import annotations

import gc
import threading
from contextlib import contextmanager

import torch

from webapp.backend.config import USE_FA3

_lock = threading.Lock()
_image_model = None
_image_processor = None
_video_predictor = None
_infer_dtype: torch.dtype | None = None


def _setup_torch():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _autocast_dtype() -> torch.dtype | None:
    """Match notebook inference: bfloat16 on CUDA when supported, else float16."""
    global _infer_dtype
    if _infer_dtype is not None:
        return _infer_dtype
    if not torch.cuda.is_available():
        _infer_dtype = None
    elif torch.cuda.is_bf16_supported():
        _infer_dtype = torch.bfloat16
    else:
        _infer_dtype = torch.float16
    return _infer_dtype


@contextmanager
def inference_mode():
    """Run model forward passes with the same autocast settings as the example notebooks."""
    dtype = _autocast_dtype()
    if torch.cuda.is_available() and dtype is not None:
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            yield
    else:
        with torch.inference_mode():
            yield


def get_image_processor():
    global _image_model, _image_processor
    with _lock:
        if _image_processor is None:
            _setup_torch()
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _image_model = build_sam3_image_model(device=device)
            _image_processor = Sam3Processor(_image_model, device=device, confidence_threshold=0.35)
        return _image_processor


def get_video_predictor():
    global _video_predictor
    with _lock:
        if _video_predictor is None:
            _setup_torch()
            from sam3.model_builder import build_sam3_multiplex_video_predictor

            _video_predictor = build_sam3_multiplex_video_predictor(use_fa3=USE_FA3)
        return _video_predictor


def image_model_loaded() -> bool:
    with _lock:
        return _image_processor is not None


def video_model_loaded() -> bool:
    with _lock:
        return _video_predictor is not None


def _free_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def unload_image_model() -> bool:
    """Drop the image model from memory and free GPU cache."""
    global _image_model, _image_processor
    with _lock:
        if _image_processor is None and _image_model is None:
            return False
        _image_processor = None
        _image_model = None
    _free_cuda_memory()
    return True


def unload_video_model() -> bool:
    """Drop the video predictor from memory and free GPU cache."""
    global _video_predictor
    with _lock:
        if _video_predictor is None:
            return False
        _video_predictor = None
    _free_cuda_memory()
    return True


def device_info() -> dict:
    cuda = torch.cuda.is_available()
    info = {
        "cuda_available": cuda,
        "device": "cuda" if cuda else "cpu",
        "image_model_loaded": image_model_loaded(),
        "video_model_loaded": video_model_loaded(),
    }
    if cuda:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        info["gpu_memory_free_gb"] = round(free / 1e9, 2)
        info["gpu_memory_total_gb"] = round(total / 1e9, 2)
    return info
