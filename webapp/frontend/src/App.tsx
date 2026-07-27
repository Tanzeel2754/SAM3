import { useEffect, useState } from "react";
import { Cpu, Image, Video, Zap } from "lucide-react";
import { api, HealthInfo } from "./api/client";
import ImageWorkspace from "./components/ImageWorkspace";
import VideoWorkspace from "./components/VideoWorkspace";

type Tab = "image" | "video";

interface Toast {
  id: number;
  message: string;
  error?: boolean;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("image");
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = (message: string, error = false) => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, error }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4000);
  };

  useEffect(() => {
    const boot = async () => {
      try {
        // Clear any models left loaded by a previous browser session.
        await api.releaseImageModel();
        await api.releaseVideoModel();
        setHealth(await api.health());
      } catch {
        setHealth({ status: "error", cuda_available: false, device: "unknown" });
      }
    };
    boot();
  }, []);

  const switchTab = async (next: Tab) => {
    if (next === tab) return;
    try {
      const updated =
        tab === "image" ? await api.releaseImageModel() : await api.releaseVideoModel();
      setHealth(updated);
    } catch {
      /* release is best-effort; still switch tabs */
    }
    setTab(next);
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">
            <Zap size={20} color="white" />
          </div>
          <div>
            <h1>SAM3 Studio</h1>
            <p>Interactive image & video segmentation</p>
          </div>
        </div>
        <div className="status-pill">
          <span className={`status-dot ${health?.status === "ok" ? "" : "offline"}`} />
          <Cpu size={14} />
          {health?.cuda_available
            ? `${health.gpu_name ?? "GPU"} · ${health.gpu_memory_free_gb ?? "?"} GB free`
            : health?.status === "ok"
              ? "CPU mode"
              : "Connecting…"}
        </div>
      </header>

      <nav className="tab-bar">
        <button className={`tab-btn ${tab === "image" ? "active" : ""}`} onClick={() => switchTab("image")}>
          <Image size={16} />
          Image Segmentation
        </button>
        <button className={`tab-btn ${tab === "video" ? "active" : ""}`} onClick={() => switchTab("video")}>
          <Video size={16} />
          Video Tracking
        </button>
      </nav>

      {tab === "image" ? <ImageWorkspace onToast={showToast} /> : <VideoWorkspace onToast={showToast} />}

      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.error ? "error" : ""}`}>
          {t.message}
        </div>
      ))}
    </div>
  );
}
