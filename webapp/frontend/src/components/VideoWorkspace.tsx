import { useCallback, useEffect, useRef, useState } from "react";
import {
  Eraser,
  FastForward,
  MousePointerClick,
  Pause,
  Play,
  Scissors,
  Sparkles,
  Upload,
  Video,
  X,
} from "lucide-react";
import { api, FrameResult, toDataUrl, VideoSessionResponse } from "../api/client";

interface Props {
  onToast: (msg: string, error?: boolean) => void;
}

type PointMode = "positive" | "negative";

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  return `${m}:${s.toString().padStart(2, "0")}.${ms}`;
}

export default function VideoWorkspace({ onToast }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const previewVideoRef = useRef<HTMLVideoElement>(null);

  const [session, setSession] = useState<VideoSessionResponse | null>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState(0);
  const [videoFps, setVideoFps] = useState(24);
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState(0);
  const [frameIndex, setFrameIndex] = useState(0);
  const [frameResult, setFrameResult] = useState<FrameResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [propagating, setPropagating] = useState(false);
  const [propProgress, setPropProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [pointMode, setPointMode] = useState<PointMode>("positive");
  const [clickMode, setClickMode] = useState(false);
  const [accumulatedPoints, setAccumulatedPoints] = useState<{ points: number[][]; labels: number[] }>({
    points: [],
    labels: [],
  });
  const sessionIdRef = useRef<string | null>(null);

  useEffect(() => {
    sessionIdRef.current = session?.session_id ?? null;
  }, [session]);

  useEffect(() => {
    return () => {
      const id = sessionIdRef.current;
      if (id) api.videoDelete(id).catch(() => undefined);
    };
  }, []);

  const loadFrame = useCallback(
    async (idx: number) => {
      if (!session) return;
      try {
        const res = await api.videoFrame(session.session_id, idx);
        setFrameResult(res);
      } catch (e) {
        onToast(e instanceof Error ? e.message : "Failed to load frame", true);
      }
    },
    [session, onToast],
  );

  useEffect(() => {
    if (session) loadFrame(frameIndex);
  }, [frameIndex, session, loadFrame]);

  useEffect(() => {
    return () => {
      if (playRef.current) clearInterval(playRef.current);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const clearPending = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPendingFile(null);
    setPreviewUrl(null);
    setVideoDuration(0);
    setTrimStart(0);
    setTrimEnd(0);
  };

  const selectFile = (file: File) => {
    if (!file.type.startsWith("video/")) {
      onToast("Please upload a video file", true);
      return;
    }
    clearPending();
    setSession(null);
    setFrameResult(null);
    setFrameIndex(0);
    setPendingFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const onPreviewLoaded = () => {
    const video = previewVideoRef.current;
    if (!video || !Number.isFinite(video.duration)) return;
    const duration = video.duration;
    setVideoDuration(duration);
    setTrimStart(0);
    setTrimEnd(duration);
    setVideoFps(30);
  };

  const seekPreview = (time: number) => {
    const video = previewVideoRef.current;
    if (video) video.currentTime = Math.min(Math.max(time, 0), videoDuration || time);
  };

  const estimatedFrames =
    videoDuration > 0 ? Math.max(1, Math.round((trimEnd - trimStart) * videoFps)) : 0;
  const isTrimmed = videoDuration > 0 && (trimStart > 0.05 || trimEnd < videoDuration - 0.05);

  const uploadPending = async () => {
    if (!pendingFile || videoDuration <= 0) return;
    if (trimEnd <= trimStart) {
      onToast("End time must be after start time", true);
      return;
    }

    setLoading(true);
    try {
      const trim = isTrimmed ? { startTime: trimStart, endTime: trimEnd } : undefined;
      const res = await api.createVideoSession(pendingFile, trim);
      setSession(res);
      setFrameIndex(0);
      setAccumulatedPoints({ points: [], labels: [] });
      clearPending();
      const trimNote =
        res.trim_start != null && res.trim_end != null
          ? ` (trimmed ${formatTime(res.trim_start)}–${formatTime(res.trim_end)})`
          : "";
      onToast(`Video loaded — ${res.frame_count} frames @ ${res.fps.toFixed(1)} fps${trimNote}`);
    } catch (e) {
      onToast(e instanceof Error ? e.message : "Upload failed", true);
    } finally {
      setLoading(false);
    }
  };

  const uploadFile = (file: File) => selectFile(file);

  const runTextPrompt = async () => {
    if (!session || !prompt.trim()) return;
    setLoading(true);
    try {
      const res = await api.videoTextPrompt(session.session_id, frameIndex, prompt.trim());
      setFrameResult(res);
      onToast(`Frame ${frameIndex}: ${res.object_count} object(s) detected`);
    } catch (e) {
      onToast(e instanceof Error ? e.message : "Prompt failed", true);
    } finally {
      setLoading(false);
    }
  };

  const handleFrameClick = async (e: React.MouseEvent<HTMLImageElement>) => {
    if (!clickMode || !session || !frameResult) return;
    const img = e.currentTarget;
    const rect = img.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    const label = pointMode === "positive" ? 1 : 0;

    const newPoints = [...accumulatedPoints.points, [x, y]];
    const newLabels = [...accumulatedPoints.labels, label];
    setAccumulatedPoints({ points: newPoints, labels: newLabels });

    setLoading(true);
    try {
      const res = await api.videoPointPrompt(session.session_id, frameIndex, newPoints, newLabels, false);
      setFrameResult(res);
      onToast(`Point added (${pointMode}) — ${res.object_count} object(s)`);
    } catch (err) {
      onToast(err instanceof Error ? err.message : "Point prompt failed", true);
    } finally {
      setLoading(false);
    }
  };

  const propagate = () => {
    if (!session) return;
    setPropagating(true);
    setPropProgress(0);
    const cancel = api.propagateVideo(
      session.session_id,
      (data) => {
        setPropProgress(((data.frame_index + 1) / session.frame_count) * 100);
      },
      async () => {
        setPropagating(false);
        setPropProgress(100);
        await loadFrame(frameIndex);
        onToast("Propagation complete");
      },
      (msg) => {
        setPropagating(false);
        onToast(msg, true);
      },
    );
    return cancel;
  };

  const resetSession = async () => {
    if (!session) return;
    setLoading(true);
    try {
      await api.videoReset(session.session_id);
      setAccumulatedPoints({ points: [], labels: [] });
      setPrompt("");
      await loadFrame(frameIndex);
      onToast("Session reset");
    } catch (e) {
      onToast(e instanceof Error ? e.message : "Reset failed", true);
    } finally {
      setLoading(false);
    }
  };

  const togglePlay = () => {
    if (!session) return;
    if (playing) {
      if (playRef.current) clearInterval(playRef.current);
      setPlaying(false);
      return;
    }
    setPlaying(true);
    playRef.current = setInterval(() => {
      setFrameIndex((prev) => {
        if (!session) return prev;
        if (prev >= session.frame_count - 1) {
          if (playRef.current) clearInterval(playRef.current);
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, Math.max(50, 1000 / (session.fps || 24)));
  };

  const displaySrc = frameResult
    ? showOverlay && frameResult.overlay_image
      ? toDataUrl(frameResult.overlay_image, "image/jpeg")
      : toDataUrl(frameResult.source_image, "image/jpeg")
    : null;

  return (
    <div className="workspace">
      <aside className="sidebar">
        <div className="panel">
          <h3>Upload Video</h3>
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            className="hidden-input"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) selectFile(file);
              e.target.value = "";
            }}
          />
          <button
            className="btn btn-secondary"
            style={{ width: "100%" }}
            onClick={() => fileRef.current?.click()}
            disabled={loading || propagating}
          >
            <Upload size={16} />
            {session || pendingFile ? "Choose Another Video" : "Choose Video"}
          </button>
          {pendingFile && (
            <p className="point-mode-hint" style={{ marginTop: "0.5rem" }}>
              Set trim range in the preview, then upload.
            </p>
          )}
        </div>

        <div className="panel">
          <h3>Text Prompt</h3>
          <div className="field">
            <input
              type="text"
              placeholder='e.g. "person", "shoe"'
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runTextPrompt()}
              disabled={!session}
            />
          </div>
          <button className="btn btn-primary" style={{ width: "100%" }} onClick={runTextPrompt} disabled={!session || loading}>
            <Sparkles size={16} />
            Segment Frame
          </button>
        </div>

        <div className="panel">
          <h3>Point Prompt</h3>
          <div className="toggle-group" style={{ marginBottom: "0.75rem" }}>
            <button
              className={`toggle-btn ${pointMode === "positive" ? "active positive" : ""}`}
              onClick={() => setPointMode("positive")}
            >
              Positive
            </button>
            <button
              className={`toggle-btn ${pointMode === "negative" ? "active negative" : ""}`}
              onClick={() => setPointMode("negative")}
            >
              Negative
            </button>
          </div>
          <button
            className={`btn btn-secondary ${clickMode ? "active" : ""}`}
            style={{ width: "100%", marginBottom: "0.5rem" }}
            onClick={() => setClickMode(!clickMode)}
            disabled={!session}
          >
            <MousePointerClick size={15} />
            {clickMode ? "Click Mode On" : "Enable Click Mode"}
          </button>
          <p className="point-mode-hint">
            Click on the current frame to add positive (include) or negative (exclude) points.
          </p>
        </div>

        <div className="panel">
          <h3>Tracking</h3>
          <button
            className="btn btn-primary"
            style={{ width: "100%", marginBottom: "0.5rem" }}
            onClick={propagate}
            disabled={!session || propagating}
          >
            <FastForward size={16} />
            Propagate in Video
          </button>
          {propagating && (
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${propProgress}%` }} />
            </div>
          )}
          <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.75rem", fontSize: "0.82rem", color: "var(--text-muted)" }}>
            <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} />
            Show mask overlay
          </label>
        </div>

        <button className="btn btn-danger" onClick={resetSession} disabled={!session || loading}>
          <Eraser size={15} />
          Reset Session
        </button>

        {frameResult && frameResult.object_count > 0 && (
          <div className="panel">
            <h3>Tracked Objects</h3>
            <div className="object-list">
              {frameResult.objects.map((obj) => (
                <div key={obj.obj_id} className="object-item">
                  <span>id={obj.obj_id}</span>
                  <span>{obj.score != null ? obj.score.toFixed(2) : "—"}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </aside>

      <main className="canvas-area">
        <div className="canvas-toolbar">
          <span className="canvas-meta">
            {session
              ? `Frame ${frameIndex + 1} / ${session.frame_count} · ${session.width}×${session.height}`
              : "No video loaded"}
          </span>
          {clickMode && <span className="badge">Click to add {pointMode} point</span>}
        </div>

        {session && (
          <div className="timeline">
            <div className="timeline-controls">
              <button className="icon-btn" onClick={togglePlay} disabled={propagating}>
                {playing ? <Pause size={16} /> : <Play size={16} />}
              </button>
            </div>
            <input
              type="range"
              min={0}
              max={session.frame_count - 1}
              value={frameIndex}
              onChange={(e) => {
                setPlaying(false);
                if (playRef.current) clearInterval(playRef.current);
                setFrameIndex(parseInt(e.target.value, 10));
              }}
            />
            <span className="slider-value">{frameIndex}</span>
          </div>
        )}

        <div className={`viewer ${!displaySrc && !pendingFile ? "empty" : ""}`}>
          {pendingFile && previewUrl ? (
            <div className="trim-panel">
              <div className="video-preview-wrap">
                <video
                  ref={previewVideoRef}
                  src={previewUrl}
                  controls
                  onLoadedMetadata={onPreviewLoaded}
                />
              </div>

              <div className="trim-range-row">
                <label>Start</label>
                <input
                  type="range"
                  min={0}
                  max={videoDuration || 1}
                  step={0.1}
                  value={trimStart}
                  disabled={!videoDuration}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setTrimStart(Math.min(v, trimEnd - 0.1));
                    seekPreview(v);
                  }}
                />
                <span className="slider-value">{formatTime(trimStart)}</span>
              </div>

              <div className="trim-range-row">
                <label>End</label>
                <input
                  type="range"
                  min={0}
                  max={videoDuration || 1}
                  step={0.1}
                  value={trimEnd}
                  disabled={!videoDuration}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setTrimEnd(Math.max(v, trimStart + 0.1));
                    seekPreview(v);
                  }}
                />
                <span className="slider-value">{formatTime(trimEnd)}</span>
              </div>

              <div className="trim-meta">
                <span>
                  Duration: <strong>{formatTime(Math.max(0, trimEnd - trimStart))}</strong>
                </span>
                <span>
                  Est. frames: <strong>{estimatedFrames}</strong>
                </span>
                <span>Full video: {formatTime(videoDuration)}</span>
              </div>

              <div className="trim-actions">
                <button className="btn btn-primary" onClick={uploadPending} disabled={loading || !videoDuration}>
                  <Scissors size={16} />
                  Upload {isTrimmed ? "Trimmed Clip" : "Full Video"}
                </button>
                <button className="btn btn-secondary" onClick={clearPending} disabled={loading}>
                  <X size={16} />
                  Cancel
                </button>
              </div>
            </div>
          ) : !displaySrc ? (
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
                <Video size={28} />
              </div>
              <h2>Drop a video here</h2>
              <p>Upload MP4/WebM, trim to the segment you need, then track with SAM 3.1</p>
              <button type="button" className="btn btn-primary" onClick={() => fileRef.current?.click()}>
                Browse Files
              </button>
            </label>
          ) : (
            <div className="image-canvas-wrap">
              <img
                src={displaySrc}
                alt={`Frame ${frameIndex}`}
                draggable={false}
                onClick={handleFrameClick}
                style={{ cursor: clickMode ? "crosshair" : "default" }}
              />
            </div>
          )}
          {(loading || propagating) && (
            <div className="loading-overlay">
              <div className="spinner" />
              <span>{propagating ? `Propagating… ${propProgress.toFixed(0)}%` : "Processing…"}</span>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
