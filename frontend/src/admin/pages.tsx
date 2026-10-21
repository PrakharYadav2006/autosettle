import { useMemo, useState } from "react";
import { AlertTriangle, ArrowRight, CheckCircle2, ChevronRight, Clock3, Search, XCircle } from "lucide-react";
import { adminApi, type AuditRec, type Overview, type QueueItem } from "./api";
import { Card, State, Td, Th, claimTone, describeAudit, reasonText, ref, useLoad, when } from "./kit";
import { cx, inr, title } from "../format";
import { Pill, Segmented } from "../components/ui";

type Nav = (p: string) => void;

/* ------------------------------ Overview ------------------------------ */
export function OverviewPage({ onAuth, nav }: { onAuth: () => void; nav: Nav }) {
  const ov = useLoad<Overview>(adminApi.overview, [], onAuth);
  const q = useLoad<QueueItem[]>(() => adminApi.claims("review"), [], onAuth);
  const o = ov.data;
  if (!o) return <State err={ov.err} />;
  const tiles: { k: string; v: number; hint: string; to?: string; tone?: string }[] = [
    { k: "Claims processed", v: o.claims_total, hint: "stored on this server", to: "/admin/claims" },
    { k: "Needs review or escalated", v: o.needs_review, hint: "automatic settlement paused", to: "/admin/claims?status=review", tone: o.needs_review ? "text-amber-200" : undefined },
    { k: "Offer with customer", v: o.awaiting_customer, hint: "awaiting a decision", to: "/admin/claims?status=OPEN" },
    { k: "Settled", v: o.settled, hint: "accepted by customer (demo)", to: "/admin/claims?status=SETTLED" },
    { k: "Failed assessments", v: o.failed_assessments_since_start, hint: "since server start", tone: o.failed_assessments_since_start ? "text-rose-200" : undefined },
  ];
  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {tiles.map((t) => (
          <button key={t.k} disabled={!t.to} onClick={() => t.to && nav(t.to)}
            className="panel group p-5 text-left transition enabled:hover:border-teal-300/25 enabled:hover:bg-ink-850">
            <div className="text-sm text-slate-400">{t.k}</div>
            <div className={cx("num mt-3 text-4xl font-semibold tracking-tight text-slate-50", t.tone)}>{t.v}</div>
            <div className="mt-2 flex items-center justify-between text-xs text-slate-500">{t.hint}{t.to && <ArrowRight className="h-3.5 w-3.5 opacity-0 transition group-hover:opacity-100" />}</div>
          </button>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Card title="Needs attention" sub="Claims where automatic settlement is paused." pad={false}
          right={<button className="btn-quiet" onClick={() => nav("/admin/claims?status=review")}>Open queue <ChevronRight className="h-4 w-4" /></button>}>
          {!q.data ? <div className="px-6 pb-6"><State err={q.err} /></div> : q.data.length === 0 ? (
            <div className="flex items-center gap-3 px-6 pb-8 text-slate-400"><CheckCircle2 className="h-5 w-5 text-teal-300" /> Nothing needs review right now.</div>
          ) : (
            <ul className="divide-y divide-white/[0.05] border-t border-white/[0.05]">
              {q.data.slice(0, 6).map((c) => {
                const st = claimTone(c.quote_status, c.chat_state, c.settlement_scope);
                return (
                  <li key={c.claim_id}>
                    <button onClick={() => nav(`/admin/claims/${c.claim_id}/summary`)} className="flex w-full items-center gap-4 px-6 py-4 text-left transition hover:bg-white/[0.03]">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="num font-medium text-slate-100">{ref(c.claim_id)}</span>
                          <Pill tone={st.tone} dot>{st.label}</Pill>
                        </div>
                        <div className="mt-1 truncate text-sm text-slate-400">
                          {c.reasons.length ? reasonText(c.reasons[0]) : c.chat_state === "ESCALATED" ? "Customer asked for a person or declined the final offer" : "—"}
                        </div>
                      </div>
                      <div className="hidden text-right text-sm text-slate-500 sm:block">{when(c.last_activity)}</div>
                      <ChevronRight className="h-4 w-4 text-slate-600" />
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </Card>

        <div className="space-y-6">
          <Card title="Service health" sub="Reported by the running backend.">
            <dl className="divide-y divide-white/[0.05]">
              {[
                ["Models", o.health.models_loaded ? "Loaded" : "Not loaded", o.health.models_loaded],
                ["Part detection", `Grounding DINO (${o.health.dino})`, true],
                ["Vision model (damage witness)", o.health.severity === "vlm" ? `${(o.health.vision_model ?? "Qwen VLM").split("/").pop()} · facts + rules` : "Off · measured-area rules", true],
                ["Pricing requires VLM confirmation", o.health.vlm_approval_gate ? "Yes (fail-closed)" : "No · detector-only items can be priced", !!o.health.vlm_approval_gate || o.health.severity !== "vlm"],
                ["Demo Mode entry", o.health.demo_mode ? "On · judges can start a demo session" : "Off (hidden from customers)", true],
                ["Chat language model (text only)", o.health.chat_llm ? `${o.health.chat_llm_model ?? "Ollama"} · wording guarded` : "Off · rules + fixed templates", true],
              ].map(([k, v, ok]) => (
                <div key={k as string} className="flex items-center justify-between gap-4 py-3 text-sm">
                  <dt className="text-slate-400">{k}</dt>
                  <dd className="flex items-center gap-2 text-slate-200">{ok ? <span className="h-2 w-2 rounded-full bg-teal-300" /> : <span className="h-2 w-2 rounded-full bg-rose-300" />}{v}</dd>
                </div>
              ))}
            </dl>
          </Card>
          <Card title="Not yet available" sub="Backend capabilities this console does not pretend to have.">
            <ul className="space-y-2.5 text-sm text-slate-400">
              {o.missing_capabilities.map((m) => <li key={m} className="flex gap-2.5"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300/80" />{m}</li>)}
            </ul>
          </Card>
        </div>
      </div>

      {o.recent_failures.length > 0 && (
        <Card title="Recent failed assessments">
          <ul className="space-y-3 text-sm">{o.recent_failures.map((f, i) => (
            <li key={i} className="flex gap-3"><XCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-300" /><span className="text-slate-300">{f.error}</span><span className="ml-auto whitespace-nowrap text-slate-500">{when(f.time)}</span></li>
          ))}</ul>
        </Card>
      )}
    </div>
  );
}

/* ------------------------------ Claim queue ------------------------------ */
const FILTERS = [
  { value: "", label: "All" }, { value: "review", label: "Needs review" }, { value: "OPEN", label: "With customer" },
  { value: "SETTLED", label: "Settled" }, { value: "NO_PAYOUT", label: "No payout" },
];

export function QueuePage({ onAuth, nav, search, replace }: { onAuth: () => void; nav: Nav; search: string; replace: Nav }) {
  const params = new URLSearchParams(search);
  const status = params.get("status") ?? "";
  const [q, setQ] = useState(params.get("q") ?? "");
  const { data, err } = useLoad<QueueItem[]>(() => adminApi.claims(status || undefined), [status], onAuth);
  const rows = useMemo(() => {
    const s = q.trim().toLowerCase().replace(/^as-/, "");
    return (data ?? []).filter((c) => !s || c.claim_id.toLowerCase().startsWith(s) || (c.plate ?? "").toLowerCase().includes(s));
  }, [data, q]);
  const setUrl = (st: string, qq: string) => {
    const p = new URLSearchParams(); if (st) p.set("status", st); if (qq) p.set("q", qq);
    replace(`/admin/claims${p.toString() ? "?" + p : ""}`);
  };

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <label className="relative min-w-[240px] flex-1 sm:max-w-sm">
          <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
          <input className="input pl-10" placeholder="Search claim ref or plate" value={q} aria-label="Search claims"
            onChange={(e) => { setQ(e.target.value); setUrl(status, e.target.value); }} />
        </label>
        <div className="overflow-x-auto"><Segmented size="sm" value={status} onChange={(v) => setUrl(v, q)} options={FILTERS} /></div>
      </div>

      {!data ? <State err={err} /> : rows.length === 0 ? <State empty="No claims match this search or filter." /> : (
        <>
          {/* desktop table */}
          <div className="panel hidden overflow-hidden md:block">
            <table className="w-full">
              <thead className="border-b border-white/[0.06] bg-white/[0.02]"><tr>
                <Th>Claim</Th><Th>Vehicle</Th><Th>Status</Th><Th>Reason / outcome</Th><Th r>Estimate</Th><Th r>Offer</Th><Th>Last activity</Th><Th />
              </tr></thead>
              <tbody className="divide-y divide-white/[0.04]">
                {rows.map((c) => {
                  const st = claimTone(c.quote_status, c.chat_state, c.settlement_scope);
                  return (
                    <tr key={c.claim_id} onClick={() => nav(`/admin/claims/${c.claim_id}/summary`)} className="group cursor-pointer transition hover:bg-white/[0.03]">
                      <Td><div className="num font-medium text-slate-100">{ref(c.claim_id)}</div><div className="mt-0.5 text-xs text-slate-500">{c.media_type} · {c.findings} finding{c.findings === 1 ? "" : "s"}</div></Td>
                      <Td><div className="text-slate-200">{title(c.vehicle_type ?? "")}</div><div className="num mt-0.5 text-xs text-slate-500">{c.plate ?? "no plate"}</div></Td>
                      <Td><Pill tone={st.tone} dot>{st.label}</Pill></Td>
                      <Td className="max-w-[340px] text-slate-400">
                        {c.reasons.length ? <span>{reasonText(c.reasons[0])}{c.reasons.length > 1 && <span className="text-slate-500"> +{c.reasons.length - 1} more</span>}</span>
                          : c.chat_state === "ESCALATED" ? "Customer asked for a person / declined final offer"
                          : c.chat_state === "SETTLED" ? "Offer accepted" : "All checks passed"}
                      </Td>
                      <Td r className="text-slate-300">{inr(c.garage_estimate_inr)}</Td>
                      <Td r className="text-slate-100">{inr(c.settled_amount_inr ?? c.current_offer_inr)}</Td>
                      <Td className="whitespace-nowrap text-slate-400">{when(c.last_activity)}</Td>
                      <Td><ChevronRight className="h-4 w-4 text-slate-600 transition group-hover:text-slate-300" /></Td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {/* mobile cards */}
          <ul className="space-y-3 md:hidden">
            {rows.map((c) => {
              const st = claimTone(c.quote_status, c.chat_state, c.settlement_scope);
              return (
                <li key={c.claim_id}>
                  <button onClick={() => nav(`/admin/claims/${c.claim_id}/summary`)} className="panel w-full p-4 text-left">
                    <div className="flex items-center justify-between gap-3"><span className="num font-medium text-slate-100">{ref(c.claim_id)}</span><Pill tone={st.tone} dot>{st.label}</Pill></div>
                    <div className="mt-2 text-sm text-slate-400">{title(c.vehicle_type ?? "")} · {c.plate ?? "no plate"} · {when(c.last_activity)}</div>
                    {c.reasons.length > 0 && <div className="mt-2 text-sm text-slate-300">{reasonText(c.reasons[0])}</div>}
                  </button>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}

/* ------------------------------ Audit log ------------------------------ */
const TONE_DOT = { neutral: "bg-slate-500", teal: "bg-teal-300", amber: "bg-amber-300", red: "bg-rose-300" };

export function AuditList({ rows, nav, hideClaim }: { rows: AuditRec[]; nav?: Nav; hideClaim?: boolean }) {
  if (!rows.length) return <State empty="No audit records yet." />;
  // Collapse runs of identical view/login events ("Staff opened the claim ×5") - the raw log is unchanged.
  const groups: { r: AuditRec; n: number }[] = [];
  for (const r of rows) {
    const last = groups[groups.length - 1];
    const quiet = r.action === "admin_view_claim" || r.action === "admin_login";
    if (last && quiet && last.r.action === r.action && last.r.actor === r.actor && last.r.claim_id === r.claim_id) last.n += 1;
    else groups.push({ r, n: 1 });
  }
  return (
    <ol className="relative space-y-1">
      {groups.map(({ r, n }, i) => {
        const d = describeAudit(r);
        const { time, actor, action, claim_id, ...payload } = r;
        void action;
        return (
          <li key={i} className="relative flex gap-4 rounded-xl px-3 py-3 hover:bg-white/[0.02]">
            <span className={cx("mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full", TONE_DOT[d.tone])} />
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="font-medium text-slate-100">{d.title}{n > 1 && <span className="ml-1.5 text-sm font-normal text-slate-500">×{n}</span>}</span>
                {claim_id && !hideClaim && (nav ? <button className="num text-sm text-teal-300 hover:underline" onClick={() => nav(`/admin/claims/${claim_id}/summary`)}>{ref(claim_id)}</button>
                  : <span className="num text-sm text-slate-400">{ref(claim_id)}</span>)}
              </div>
              {d.detail && <p className="mt-1 text-sm text-slate-400">{d.detail}</p>}
              {Object.keys(payload).length > 0 && (
                <details className="mt-1.5 text-xs text-slate-500">
                  <summary className="cursor-pointer select-none hover:text-slate-300">Details</summary>
                  <pre className="num mt-2 overflow-x-auto whitespace-pre-wrap rounded-lg bg-black/30 p-3 text-[11px] leading-relaxed text-slate-400">{JSON.stringify(payload, null, 2)}</pre>
                </details>
              )}
            </div>
            <div className="shrink-0 text-right text-xs text-slate-500">
              <div className="flex items-center justify-end gap-1"><Clock3 className="h-3 w-3" />{when(time)}</div>
              <div className="mt-0.5">{actor}</div>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

export function AuditPage({ onAuth, nav }: { onAuth: () => void; nav: Nav }) {
  const { data, err } = useLoad<AuditRec[]>(adminApi.audit, [], onAuth);
  return <Card title="System and staff activity" sub="Append-only record written by the backend. Newest first.">{data ? <AuditList rows={data} nav={nav} /> : <State err={err} />}</Card>;
}
