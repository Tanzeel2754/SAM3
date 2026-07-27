import { useCallback, useEffect, useRef, useState } from "react";
import {
  BoxSelect,
  Eraser,
  ImageIcon,
  MousePointer2,
  Sparkles,
  Upload,
} from "lucide-react";
import { api, SegmentationResult, toDataUrl } from "../api/client";

interface Props {
  onToast: (msg: string, error?: boolean) => void;
}

type BoxMode = "positive" | "negative";
type Tool = "view" | "box";

export default function ImageWorkspace({ onToast }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [confidence, setConfidence] = useState(0.35);
  const [boxMode, setBoxMode] = useState<BoxMode>("positive");
  const [tool, setTool] = useState<Tool>("view");
  const [drawing, setDrawing] = useState(false);
  const [boxStart, setBoxStart] = useState<{ x: number; y: number } | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    return () => {
      const id = sessionIdRef.current;
      if (id) api.imageDelete(id).catch(() => undefined);
    };
  }, []);

  const displaySrc = result?.overlay_image
    ? toDataUrl(result.overlay_image)
    : result?.source_image
      ? toDataUrl(result.source_image)
      : null;

  const uploadFile = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      onToast("Please upload an image file", true);
      return;
    }
    setLoading(true);
    try {
      const session = await api.createImageSession(file);
      setSessionId(session.session_id);
      setResult({
        session_id: session.session_id,
        width: session.width,
        height: session.height,
        object_count: 0,
        objects: [],
        has_masks: false,
        overlay_image: "",
        source_image: session.source_image,
      });
      setConfidence(0.35);
      setTool("view");
      onToast(`Image loaded (${session.width}×${session.height})`);
    } catch (e) {
      onToast(e instanceof Error ? e.message : "Upload failed", true);
    } finally {
      setLoading(false);
    }
  };

  const runTextPrompt = async () => {
    if (!sessionId || !prompt.trim()) return;
    setLoading(true);
    try {
      await api.imageSetConfidence(sessionId, confidence);
      const res = await api.imageTextPrompt(sessionId, prompt.trim());
      setResult(res);
      if (res.object_count > 0) {
        onToast(`Found ${res.object_count} object(s)`);
      } else if (res.peak_score != null && res.peak_score > 0) {
        onToast(
          `No objects above ${confidence.toFixed(2)} (best match: ${res.peak_score.toFixed(2)}). Lower confidence to include it.`,
          true,
        );
      } else {
        onToast('No objects detected. Try a simpler prompt (e.g. "person").', true);
      }
    } catch (e) {
      onToast(e instanceof Error ? e.message : "Segmentation failed", true);
    } finally {
      setLoading(false);
    }
  };

  const updateConfidence = async (value: number) => {
    setConfidence(value);
    if (!sessionId || !result?.has_masks) return;
    try {
      const res = await api.imageSetConfidence(sessionId, value);
      setResult(res);
    } catch (e) {
      onToast(e instanceof Error ? e.message : "Confidence update failed", true);
    }
  };

  const resetPrompts = async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const res = await api.imageReset(sessionId);
      setResult(res);
      setPrompt("");
      setConfidence(0.35);
      setTool("view");
      onToast("Prompts cleared");
    } catch (e) {
      onToast(e instanceof Error ? e.message : "Reset failed", true);
    } finally {
      setLoading(false);
    }
  };

  const getCanvasCoords = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const drawPreview = useCallback(
    (x: number, y: number) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !ctx || !boxStart) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const color = boxMode === "positive" ? "#22c55e" : "#ef4444";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      const w = x - boxStart.x;
      const h = y - boxStart.y;
      ctx.strokeRect(boxStart.x, boxStart.y, w, h);
    },
    [boxStart, boxMode],
  );

  const handleMouseDown = (e: React.MouseEvent) => {
    if (tool !== "box" || !sessionId) return;
    const coords = getCanvasCoords(e);
    if (!coords) return;
    setDrawing(true);
    setBoxStart(coords);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!drawing || !boxStart) return;
    const coords = getCanvasCoords(e);
    if (!coords) return;
    drawPreview(coords.x, coords.y);
  };

  const handleMouseUp = async (e: React.MouseEvent) => {
    if (!drawing || !boxStart || !sessionId || !result) return;
    setDrawing(false);
    const coords = getCanvasCoords(e);
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (coords) ctx?.clearRect(0, 0, canvas!.width, canvas!.height);

    const x0 = Math.min(boxStart.x, coords?.x ?? boxStart.x);
    const y0 = Math.min(boxStart.y, coords?.y ?? boxStart.y);
    const x1 = Math.max(boxStart.x, coords?.x ?? boxStart.x);
    const y1 = Math.max(boxStart.y, coords?.y ?? boxStart.y);
    setBoxStart(null);

    if (Math.abs(x1 - x0) < 5 || Math.abs(y1 - y0) < 5) return;

    const imgW = result.width;
    const imgH = result.height;
    const cx = ((x0 + x1) / 2) / imgW;
    const cy = ((y0 + y1) / 2) / imgH;
    const w = (x1 - x0) / imgW;
    const h = (y1 - y0) / imgH;

    setLoading(true);
    try {
      const res = await api.imageBoxPrompt(sessionId, [cx, cy, w, h], boxMode === "positive");
      setResult(res);
      if (res.object_count > 0) {
        onToast(`Added ${boxMode} box — ${res.object_count} object(s)`);
      } else if (boxMode === "negative") {
        onToast("Exclude box applied — detections in this region were removed", true);
      } else {
        onToast("Include box applied — no objects matched this region", true);
      }
    } catch (err) {
      onToast(err instanceof Error ? err.message : "Box prompt failed", true);
    } finally {
      setLoading(false);
    }
  };

  const syncCanvasSize = useCallback(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !result) return;
    canvas.width = result.width;
    canvas.height = result.height;
  }, [result]);

  useEffect(() => {
    const img = imgRef.current;
    if (!img || !result) return;
    if (img.complete) syncCanvasSize();
    else img.onload = syncCanvasSize;
  }, [result, displaySrc, syncCanvasSize]);

  useEffect(() => {
    if (tool === "box") syncCanvasSize();
  }, [tool, syncCanvasSize]);

  return (
    <div className="workspace">
      <aside className="sidebar">
        <div className="panel">
          <h3>Upload</h3>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden-input"
            onChange={(e) => e.target.files?.[0] && uploadFile(e.target.files[0])}
          />
          <button className="btn btn-secondary" style={{ width: "100%" }} onClick={() => fileRef.current?.click()}>
            <Upload size={16} />
            Choose Image
          </button>
        </div>

        <div className="panel">
          <h3>Text Prompt</h3>
          <div className="field">
            <input
              type="text"
              placeholder='e.g. "person", "red car"'
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runTextPrompt()}
              disabled={!sessionId}
            />
          </div>
          <button className="btn btn-primary" style={{ width: "100%" }} onClick={runTextPrompt} disabled={!sessionId || loading}>
            <Sparkles size={16} />
            Segment
          </button>
        </div>

        <div className="panel">
          <h3>Box Prompt</h3>
          <div className="toggle-group" style={{ marginBottom: "0.75rem" }}>
            <button
              className={`toggle-btn ${boxMode === "positive" ? "active positive" : ""}`}
              onClick={() => setBoxMode("positive")}
            >
              Include
            </button>
            <button
              className={`toggle-btn ${boxMode === "negative" ? "active negative" : ""}`}
              onClick={() => setBoxMode("negative")}
            >
              Exclude
            </button>
          </div>
          <div className="btn-group">
            <button
              className={`btn btn-secondary ${tool === "view" ? "active" : ""}`}
              onClick={() => setTool("view")}
              disabled={!sessionId}
            >
              <MousePointer2 size={15} />
              View
            </button>
            <button
              className={`btn btn-secondary ${tool === "box" ? "active" : ""}`}
              onClick={() => setTool("box")}
              disabled={!sessionId}
            >
              <BoxSelect size={15} />
              Draw Box
            </button>
          </div>
          <p className="panel-hint">Click Draw Box, then drag on the image to refine results.</p>
        </div>

        <div className="panel">
          <h3>Confidence</h3>
          <div className="slider-row">
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={confidence}
              onChange={(e) => updateConfidence(parseFloat(e.target.value))}
              disabled={!sessionId || !result?.has_masks}
            />
            <span className="slider-value">{confidence.toFixed(2)}</span>
          </div>
          <p className="panel-hint">
            {result?.has_masks
              ? "Higher values show fewer detections."
              : "Run Segment first to filter results."}
          </p>
        </div>

        <button className="btn btn-danger" onClick={resetPrompts} disabled={!sessionId || loading}>
          <Eraser size={15} />
          Clear Prompts
        </button>

        {result && result.object_count > 0 && (
          <div className="panel">
            <h3>Detected Objects</h3>
            <div className="object-list">
              {result.objects.map((obj) => (
                <div key={obj.index} className="object-item">
                  <span>#{obj.index + 1}</span>
                  <span>{obj.score.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </aside>

      <main className="canvas-area">
        <div className="canvas-toolbar">
          <span className="canvas-meta">
            {result
              ? `${result.width} × ${result.height}px · ${result.object_count} objects`
              : "No image loaded"}
          </span>
          {tool === "box" && sessionId && <span className="badge">Box draw mode active</span>}
        </div>

        <div className={`viewer ${!displaySrc ? "empty" : ""}`}>
          {!displaySrc ? (
            <label
              className={`dropzone ${dragOver ? "dragover" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                const file = e.dataTransfer.files[0];
                if (file) uploadFile(file);
              }}
            >
              <div className="dropzone-icon">
                <ImageIcon size={28} />
              </div>
              <h2>Drop an image here</h2>
              <p>Upload a photo to segment objects with text or box prompts using SAM3</p>
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => fileRef.current?.click()}
              >
                Browse Files
              </button>
            </label>
          ) : (
            <div className="image-canvas-wrap">
              <img ref={imgRef} src={displaySrc} alt="Segmentation result" draggable={false} />
              <canvas
                ref={canvasRef}
                className={tool === "box" ? "" : "canvas-inactive"}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={() => {
                  if (drawing) {
                    setDrawing(false);
                    setBoxStart(null);
                    canvasRef.current?.getContext("2d")?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                  }
                }}
              />
            </div>
          )}
          {loading && (
            <div className="loading-overlay">
              <div className="spinner" />
              <span>Running SAM3 inference…</span>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
