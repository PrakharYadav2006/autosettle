import type { ChatReply, ClaimState, DemoSession, Health } from "./types";

const URL_KEY = "autosettle.apiUrl";
const TIMEOUT_MS = 180_000; // video assessment can take a while on a laptop GPU

export function getApiUrl(): string {
  try {
    const saved = localStorage.getItem(URL_KEY);
    if (saved) return saved.replace(/\/$/, "");
  } catch { /* storage blocked */ }
  return (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
}

export function setApiUrl(url: string): void {
  try {
    if (url.trim()) localStorage.setItem(URL_KEY, url.trim());
    else localStorage.removeItem(URL_KEY);
  } catch { /* ignore */ }
}

/** Per-tab session storage (cleared when the tab closes). Never holds admin credentials. */
export const store = {
  get<T>(k: string): T | null {
    try { const v = sessionStorage.getItem("autosettle." + k); return v ? (JSON.parse(v) as T) : null; } catch { return null; }
  },
  set(k: string, v: unknown): void {
    try { sessionStorage.setItem("autosettle." + k, JSON.stringify(v)); } catch { /* ignore */ }
  },
  clear(prefix = ""): void {
    try {
      Object.keys(sessionStorage).filter((k) => k.startsWith("autosettle." + prefix)).forEach((k) => sessionStorage.removeItem(k));
    } catch { /* ignore */ }
  },
};

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

export async function request<T>(path: string, init: RequestInit = {}, headers: Record<string, string> = {}): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(getApiUrl() + path, { ...init, headers: { ...headers, ...(init.headers as Record<string, string>) }, signal: ctrl.signal });
  } catch (e) {
    throw new ApiError((e as Error).name === "AbortError" ? "The server took too long to respond. Please try again." : "We couldn't reach the claims service. Check your connection and try again.", 0);
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : detail; } catch { /* not JSON */ }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }
  return res.json() as Promise<T>;
}

const auth = () => ({ "X-Session-Token": store.get<string>("token") ?? "" });

export const api = {
  health: () => request<Health>("/api/health"),
  demoSession: () => request<DemoSession>("/api/demo_session", { method: "POST" }),
  analyzeClaim: (file: File, plate: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("plate", plate);
    fd.append("vehicle_type", "auto");
    return request<ClaimState>("/api/analyze_claim", { method: "POST", body: fd }, auth());
  },
  claim: (id: string) => request<ClaimState>(`/api/claim/${encodeURIComponent(id)}`, {}, auth()),
  chat: (claim_id: string, message: string) =>
    request<ChatReply>("/api/chat_negotiate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ claim_id, message }),
    }, auth()),
};
