import { ApiError, request } from "../api";

// Admin bearer token lives in sessionStorage for this tab only; the server checks it on every call.
const KEY = "autosettle-admin.token";
export const adminToken = {
  get: () => { try { return sessionStorage.getItem(KEY); } catch { return null; } },
  set: (t: string) => { try { sessionStorage.setItem(KEY, t); } catch { /* ignore */ } },
  clear: () => { try { sessionStorage.removeItem(KEY); } catch { /* ignore */ } },
};

const auth = () => ({ Authorization: `Bearer ${adminToken.get() ?? ""}` });

export interface QueueItem {
  claim_id: string; created_at: string; last_activity: string; plate: string | null;
  vehicle_type: string; media_type: string; findings: number; quote_status: string; chat_state: string;
  current_offer_inr: number | null; settled_amount_inr: number | null; garage_estimate_inr: number | null;
  reasons: string[]; needs_review: boolean;
  blocker_codes?: string[]; estimate_scope?: "complete" | "partial" | "none" | null; items_pending?: number;
}
export interface Overview {
  claims_total: number; awaiting_customer: number; needs_review: number; settled: number; no_payout: number;
  failed_assessments_since_start: number; recent_failures: { time: string; claim_id: string; error: string }[];
  health: {
    models_loaded: boolean; severity: string; dino: string;
    vision_model?: string | null; vlm_approval_gate?: boolean;
    chat_llm: boolean; chat_llm_model?: string | null;
  };
  missing_capabilities: string[];
}
export interface AuditRec { time: string; actor: string; action: string; claim_id: string | null; [k: string]: unknown }

export const adminApi = {
  login: (username: string, password: string) =>
    request<{ token: string; user: string; expires_at: string }>("/api/admin/login", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, password }),
    }),
  logout: () => request("/api/admin/logout", { method: "POST" }, auth()).catch(() => null),
  overview: () => request<Overview>("/api/admin/overview", {}, auth()),
  claims: (status?: string) => request<QueueItem[]>(`/api/admin/claims${status ? `?status=${status}` : ""}`, {}, auth()),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  claim: (id: string) => request<any>(`/api/admin/claims/${encodeURIComponent(id)}`, {}, auth()),
  audit: () => request<AuditRec[]>("/api/admin/audit?limit=300", {}, auth()),
};

export { ApiError };
