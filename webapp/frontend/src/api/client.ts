const API_BASE = "";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    let message = text;
    try {
      const json = JSON.parse(text);
      message = json.detail || json.error || text;
    } catch {
      /* use raw text */
    }
    throw new Error(message);
  }
  return res.json();
}

export interface HealthInfo {
  status: string;
  cuda_available: boolean;
  device: string;
  gpu_name?: string;
  gpu_memory_free_gb?: number;
  gpu_memory_total_gb?: number;
  image_model_loaded?: boolean;
  video_model_loaded?: boolean;
}

export interface ImageSessionResponse {
  session_id: string;
  width: number;
  height: number;
  source_image: string;
}

export interface SegmentationResult {
  session_id: string;
  width: number;
  height: number;
  object_count: number;
  objects: { index: number; score: number; box: number[] }[];
  has_masks: boolean;
  peak_score?: number | null;
  overlay_image: string;
  source_image: string;
}

export interface VideoSessionResponse {
  session_id: string;
  frame_count: number;
  width: number;
  height: number;
  fps: number;
  trim_start?: number | null;
  trim_end?: number | null;
}

export interface FrameResult {
  frame_index: number;
  frame_count: number;
  has_output: boolean;
  source_image: string;
  overlay_image?: string;
  object_count: number;
  objects: { obj_id: number; score: number | null }[];
}

export const api = {
  health: () => fetch(`${API_BASE}/api/health`).then(handleResponse<HealthInfo>),

  releaseImageModel: () =>
    fetch(`${API_BASE}/api/models/release/image`, { method: "POST" }).then(handleResponse<HealthInfo>),

  releaseVideoModel: () =>
    fetch(`${API_BASE}/api/models/release/video`, { method: "POST" }).then(handleResponse<HealthInfo>),

  createImageSession: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${API_BASE}/api/image/sessions`, { method: "POST", body: form }).then(
      handleResponse<ImageSessionResponse>,
    );
  },

  imageTextPrompt: (sessionId: string, prompt: string) =>
    fetch(`${API_BASE}/api/image/sessions/${sessionId}/text-prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    }).then(handleResponse<SegmentationResult>),

  imageBoxPrompt: (sessionId: string, box: number[], label: boolean) =>
    fetch(`${API_BASE}/api/image/sessions/${sessionId}/box-prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ box, label }),
    }).then(handleResponse<SegmentationResult>),

  imageSetConfidence: (sessionId: string, threshold: number) =>
    fetch(`${API_BASE}/api/image/sessions/${sessionId}/confidence`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold }),
    }).then(handleResponse<SegmentationResult>),

  imageReset: (sessionId: string) =>
    fetch(`${API_BASE}/api/image/sessions/${sessionId}/reset`, { method: "POST" }).then(
      handleResponse<SegmentationResult>,
    ),

  imageDelete: (sessionId: string) =>
    fetch(`${API_BASE}/api/image/sessions/${sessionId}`, { method: "DELETE" }).then(
      handleResponse<{ ok: boolean }>,
    ),

  createVideoSession: (file: File, trim?: { startTime: number; endTime: number }) => {
    const form = new FormData();
    form.append("file", file);
    if (trim) {
      form.append("start_time", String(trim.startTime));
      form.append("end_time", String(trim.endTime));
    }
    return fetch(`${API_BASE}/api/video/sessions`, { method: "POST", body: form }).then(
      handleResponse<VideoSessionResponse>,
    );
  },

  videoTextPrompt: (sessionId: string, frameIndex: number, text: string) =>
    fetch(`${API_BASE}/api/video/sessions/${sessionId}/text-prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame_index: frameIndex, text }),
    }).then(handleResponse<FrameResult>),

  videoPointPrompt: (
    sessionId: string,
    frameIndex: number,
    points: number[][],
    labels: number[],
    clearOldPoints = true,
  ) =>
    fetch(`${API_BASE}/api/video/sessions/${sessionId}/point-prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frame_index: frameIndex,
        points,
        labels,
        clear_old_points: clearOldPoints,
      }),
    }).then(handleResponse<FrameResult>),

  videoFrame: (sessionId: string, frameIndex: number) =>
    fetch(`${API_BASE}/api/video/sessions/${sessionId}/frames/${frameIndex}`).then(
      handleResponse<FrameResult>,
    ),

  videoReset: (sessionId: string) =>
    fetch(`${API_BASE}/api/video/sessions/${sessionId}/reset`, { method: "POST" }).then(
      handleResponse<{ ok: boolean }>,
    ),

  videoDelete: (sessionId: string) =>
    fetch(`${API_BASE}/api/video/sessions/${sessionId}`, { method: "DELETE" }).then(
      handleResponse<{ ok: boolean }>,
    ),

  propagateVideo: (
    sessionId: string,
    onProgress: (data: { frame_index: number; object_count: number }) => void,
    onDone: () => void,
    onError: (msg: string) => void,
  ) => {
    const source = new EventSource(`${API_BASE}/api/video/sessions/${sessionId}/propagate`);
    source.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) {
        source.close();
        onError(data.error);
      } else if (data.done) {
        source.close();
        onDone();
      } else {
        onProgress(data);
      }
    };
    source.onerror = () => {
      source.close();
      onError("Propagation stream failed");
    };
    return () => source.close();
  },
};

export function toDataUrl(base64: string, mime = "image/png") {
  return `data:${mime};base64,${base64}`;
}
