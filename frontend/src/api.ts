import type { ChatReply, ClaimState, DemoTier, Health, Session } from "./types";

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

interface StatusReply { claim_id: string; status: "processing" | "done" | "failed"; claim?: ClaimState; error?: string; elapsed_s?: number }

export const api = {
  health: () => request<Health>("/api/health"),
  demoSession: (tier: DemoTier) => request<Session>("/api/demo_session", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tier }),
  }),
  policySession: (policy_no: string, plate: string) => request<Session>("/api/session", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ policy_no, plate }),
  }),
  // The vehicle / policy / demo tier come from the signed session on the server, never from this form.
  // Background mode: the server replies at once and processes the file; we poll for the result.
  // (A proxy such as a Cloudflare quick tunnel cuts any single request after ~100 s, and a video
  // with VLM checks can take longer than that.)
  analyzeClaim: async (file: File): Promise<ClaimState> => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("vehicle_type", "auto");
    const first = await request<ClaimState | { claim_id: string; status: string }>(
      "/api/analyze_claim?background=true", { method: "POST", body: fd }, auth());
    if ("quote" in first) return first;                 // an older server answered synchronously
    return pollClaim(first.claim_id);
  },
  /** Proof for reconsideration: a photo/video is re-assessed by the same pipeline; a document goes to a specialist. */
  submitEvidence: async (claimId: string, file: File, kind: "photo" | "document"): Promise<ClaimState> => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("kind", kind);
    const first = await request<ClaimState | { claim_id: string; status: string }>(
      `/api/claim/${encodeURIComponent(claimId)}/evidence?background=true`, { method: "POST", body: fd }, auth());
    if ("quote" in first) return first;                 // an older server answered at once
    return pollClaim(first.claim_id);
  },
  claim: (id: string) => request<ClaimState>(`/api/claim/${encodeURIComponent(id)}`, {}, auth()),
  chat: (claim_id: string, message: string) =>
    request<ChatReply>("/api/chat_negotiate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ claim_id, message }),
    }, auth()),
};

async function pollClaim(claimId: string): Promise<ClaimState> {
    const first = { claim_id: claimId };
    const deadline = Date.now() + 15 * 60_000;
    let misses = 0;
    for (;;) {
      await new Promise((r) => setTimeout(r, 3000));
      let st: StatusReply;
      try {
        st = await request<StatusReply>(`/api/claim_status/${encodeURIComponent(first.claim_id)}`, {}, auth());
        misses = 0;
      } catch (e) {
        // a brief tunnel / network hiccup while polling is not a failed assessment
        if (e instanceof ApiError && e.status === 0 && ++misses < 6) continue;
        throw e;
      }
      if (st.status === "done" && st.claim) return st.claim;
      if (st.status === "failed") throw new ApiError(st.error ?? "We could not assess this file.", 422);
      if (Date.now() > deadline) throw new ApiError("The assessment is taking unusually long. Please try again.", 0);
    }
}

