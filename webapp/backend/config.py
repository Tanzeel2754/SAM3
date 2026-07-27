import os
import tempfile
from pathlib import Path

SAM3_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = Path(os.environ.get("SAM3_UPLOAD_DIR", tempfile.gettempdir())) / "sam3_webapp"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_SIZE_MB = int(os.environ.get("SAM3_MAX_IMAGE_MB", "20"))
MAX_VIDEO_SIZE_MB = int(os.environ.get("SAM3_MAX_VIDEO_MB", "500"))
USE_FA3 = os.environ.get("SAM3_USE_FA3", "false").lower() in ("1", "true", "yes")
