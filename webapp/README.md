# SAM3 Studio

A professional web interface for **SAM3 image segmentation** and **SAM 3.1 video tracking**, built on the same APIs used in the example notebooks.

## Features

- **Image mode**: upload images, text prompts, positive/negative box drawing, confidence threshold
- **Video mode**: upload MP4/WebM, **trim clip before processing**, text & point prompts on any frame, full-video propagation, timeline scrubber & playback
- GPU status in the header, loading states, drag-and-drop uploads

## Quick start

### 1. Install backend dependencies

From the repo root (with your `sam3` conda env active):

```bash
pip install -r webapp/requirements.txt
```

### 2. Build the frontend

```bash
cd webapp/frontend
npm install
npm run build
cd ../..
```

### 3. Start the server

```bash
python webapp/run.py --host 0.0.0.0 --port 7860
```

Open **http://127.0.0.1:7860** in your browser.

## Development mode

Run backend and frontend separately for hot reload:

```bash
# Terminal 1 — API server
python webapp/run.py --port 7860

# Terminal 2 — Vite dev server (proxies /api to :7860)
cd webapp/frontend && npm run dev
```

Open **http://127.0.0.1:5173**.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM3_UPLOAD_DIR` | system temp | Directory for uploaded files |
| `SAM3_MAX_IMAGE_MB` | 20 | Max image upload size |
| `SAM3_MAX_VIDEO_MB` | 500 | Max video upload size |
| `SAM3_USE_FA3` | false | Enable Flash Attention 3 (Linux only) |

## Notes

- Models load lazily on first use (image upload or video upload).
- **Switching tabs** (Image ↔ Video) unloads the other mode’s model and clears its sessions to free GPU memory.
- Closing the last session for a mode also unloads that model.
- Video uses **SAM 3.1 Object Multiplex** (`build_sam3_multiplex_video_predictor`).
- Image uses `build_sam3_image_model` + `Sam3Processor`, matching `sam3_image_interactive.ipynb`.
