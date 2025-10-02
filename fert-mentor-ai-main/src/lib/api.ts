export type PredictRequest = {
  N: number;
  P: number;
  K: number;
  temperature: number;
  humidity: number;
  pH: number;
  rainfall: number;
  moisture: number;
  crop: string;
};

export type PredictResponse = {
  fertilizer: string;
  predicted_class: number;
};

export type MetadataResponse = {
  feature_order: string[];
  crops: string[];
  label_mapping: Record<number, string>;
  artifacts_dir: string;
  uses_preprocessor: boolean;
};

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export async function fetchMetadata(signal?: AbortSignal): Promise<MetadataResponse> {
  const res = await fetch(`${API_BASE}/metadata`, { signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to load metadata (${res.status}): ${text}`);
  }
  return res.json();
}

export async function predictFertilizer(payload: PredictRequest, signal?: AbortSignal): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = (data && (data.detail as string)) || `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return data as PredictResponse;
}


