import { useEffect, useState, type ReactNode } from "react";
import { ApiError } from "./api";
import { cx, prettyReason } from "../format";
import type { Tone } from "../components/ui";

export const when = (iso?: string | null) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString("en-IN", { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
};
export const ref = (id: string) => `AS-${id.slice(0, 6).toUpperCase()}`;

export function claimTone(q: string, chat: string, scope?: string | null): { label: string; tone: Tone } {
  if (chat === "SETTLED" && scope === "partial") return { label: "Partially settled — rest under review", tone: "amber" };
  if (chat === "SETTLED") return { label: "Settled", tone: "teal" };
  if (chat === "ESCALATED") return { label: "Escalated by customer", tone: "red" };
  if (q === "ESCALATED") return { label: "Escalated", tone: "red" };
  if (q === "REVIEW_REQUIRED") return { label: "Review required", tone: "amber" };
  if (q === "NO_PAYOUT") return { label: "No payout", tone: "neutral" };
  if (chat === "CONFIRMING") return { label: "Customer confirming", tone: "teal" };
  return { label: "Offer with customer", tone: "teal" };
}

export function reasonText(r: string) { return prettyReason(r); }

/** Router for /admin/... paths (history API, no extra dependency). */
export function usePath(): [string, string, (p: string) => void, (p: string) => void] {
  const [loc, setLoc] = useState({ path: window.location.pathname, search: window.location.search });
  useEffect(() => {
    const f = () => setLoc({ path: window.location.pathname, search: window.location.search });
    window.addEventListener("popstate", f); return () => window.removeEventListener("popstate", f);
  }, []);
  const nav = (p: string) => { window.history.pushState({}, "", p); setLoc({ path: window.location.pathname, search: window.location.search }); window.scrollTo({ top: 0 }); };
  const replace = (p: string) => { window.history.replaceState({}, "", p); setLoc({ path: window.location.pathname, search: window.location.search }); };
  return [loc.path, loc.search, nav, replace];
}

export function useLoad<T>(fn: () => Promise<T>, deps: unknown[], onAuth: () => void) {
  const [data, setData] = useState<T | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    setErr(null);
    fn().then((d) => live && setData(d)).catch((e) => {
      if (e instanceof ApiError && e.status === 401) onAuth(); else if (live) setErr((e as Error).message);
    });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return { data, err, reload: () => setTick((t) => t + 1) };
}

export function Card({ title, sub, right, children, className, pad = true }: {
  title?: ReactNode; sub?: ReactNode; right?: ReactNode; children: ReactNode; className?: string; pad?: boolean;
}) {
  return (
    <section className={cx("panel", pad && "p-6", className)}>
      {(title || right) && (
        <div className={cx("mb-5 flex flex-wrap items-start justify-between gap-3", !pad && "px-6 pt-6")}>
          <div>{title && <h2 className="text-base font-semibold text-slate-50">{title}</h2>}{sub && <p className="mt-1 text-sm text-slate-400">{sub}</p>}</div>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export const Th = ({ children, r, className }: { children?: ReactNode; r?: boolean; className?: string }) =>
  <th className={cx("whitespace-nowrap px-4 py-3 text-xs font-medium uppercase tracking-wide text-slate-500", r ? "text-right" : "text-left", className)}>{children}</th>;
export const Td = ({ children, r, className }: { children?: ReactNode; r?: boolean; className?: string }) =>
  <td className={cx("px-4 py-4 align-top text-sm", r && "num text-right", className)}>{children}</td>;

export function State({ err, empty, children }: { err?: string | null; empty?: string; children?: ReactNode }) {
  return (
    <div className={cx("rounded-2xl border border-dashed px-6 py-12 text-center", err ? "border-rose-400/30 text-rose-200" : "border-white/10 text-slate-400")}>
      {err ?? empty ?? "Loading…"}{children}
    </div>
  );
}

export function Kv({ k, v, strong }: { k: ReactNode; v: ReactNode; strong?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-6 py-2.5">
      <dt className="text-sm text-slate-400">{k}</dt>
      <dd className={cx("num text-right", strong ? "text-base font-semibold text-slate-50" : "text-sm text-slate-200")}>{v}</dd>
    </div>
  );
}

/** Audit record -> readable sentence. Raw payload stays available under "Details". */
export function describeAudit(a: Record<string, unknown>): { title: string; detail?: string; tone: Tone } {
  const inr = (v: unknown) => (typeof v === "number" ? "₹" + v.toLocaleString("en-IN") : "—");
  switch (a.action) {
    case "claim_assessed": return { title: `Claim assessed — ${String(a.status).replace("_", " ").toLowerCase()}`, detail: [Array.isArray(a.reasons) && a.status !== "APPROVED" ? (a.reasons as string[]).map(reasonText).join("; ") : a.offer_inr ? `Opening offer ${inr(a.offer_inr)}` : "",
      a.mode === "demo" ? `Demo tier ${String(a.demo_tier)} (${String(a.tier_source)})` : "",
      Array.isArray(a.scoped_offer) && a.scoped_offer.length ? `scoped offer on finding(s) ${(a.scoped_offer as number[]).map((i) => `#${i + 1}`).join(", ")}` : "",
      Array.isArray(a.demo_rules) && a.demo_rules.length ? `demo rule: ${(a.demo_rules as { finding: number; rule: string }[]).map((d) => `#${d.finding + 1} ${d.rule}`).join("; ")}` : "",
      Array.isArray(a.excluded_already_paid) && a.excluded_already_paid.length ? `already paid, excluded: ${(a.excluded_already_paid as number[]).map((i) => `#${i + 1}`).join(", ")}` : ""].filter(Boolean).join(" · ") || undefined, tone: a.status === "APPROVED" ? "teal" : a.status === "NO_PAYOUT" ? "neutral" : "amber" };
    case "assessment_failed": return { title: "Assessment failed", detail: String(a.error ?? ""), tone: "red" };
    case "chat_turn": return { title: `Customer message → ${String(a.outcome).replace(/_/g, " ").toLowerCase()}`, detail: `intent ${a.intent} · state ${a.state} · step ${Number(a.step) + 1}/3${a.offer_inr ? ` · offer ${inr(a.offer_inr)}` : ""}`, tone: "neutral" };
    case "offer_accepted": return a.scope === "partial"
      ? { title: `Partially settled — customer accepted ${inr(a.amount_inr)} for listed items`, detail: `Items ${(a.item_ids as string[] | undefined)?.join(", ") ?? "—"} · still under review: finding(s) ${((a.still_under_review as number[] | undefined) ?? []).map((i) => `#${i + 1}`).join(", ") || "none"} · demo, no payment`, tone: "amber" }
      : { title: `Customer accepted ${inr(a.amount_inr)}`, detail: `Items ${(a.item_ids as string[] | undefined)?.join(", ") ?? "—"} · demo, no payment`, tone: "teal" };
    case "escalated_to_human": return { title: "Escalated to a specialist", tone: "amber" };
    case "admin_login": return { title: "Staff signed in", tone: "neutral" };
    case "admin_login_failed": return { title: "Failed staff sign-in attempt", tone: "red" };
    case "admin_view_claim": return { title: "Staff opened the claim", tone: "neutral" };
    default: return { title: String(a.action), tone: "neutral" };
  }
}
