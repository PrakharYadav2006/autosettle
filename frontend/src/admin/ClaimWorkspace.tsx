import { useState } from "react";
import { ArrowLeft, Bot, CheckCircle2, ExternalLink, MessageSquareDashed, ShieldAlert, User } from "lucide-react";
import { adminApi } from "./api";
import { Card, Kv, State, Td, Th, claimTone, reasonText, ref, useLoad, when } from "./kit";
import { AuditList } from "./pages";
import { cx, inr, methodLabel, shortDate, title } from "../format";
import { Pill, type Tone } from "../components/ui";
import { Evidence, type Sel } from "../components/Assessment";
import type { DamageFinding, FindingStatusCode, LineItem, Policy, PricingBreakdown, Quote } from "../types";

export const TABS = [
  ["summary", "Summary"], ["evidence", "Evidence & damage"], ["pricing", "Pricing"], ["negotiation", "Negotiation"], ["audit", "Audit trail"],
] as const;
export type Tab = (typeof TABS)[number][0];

interface Turn { time: string; user: string; intent: string; intent_source: string; outcome: string; state: string; tier: number; amount_inr: number | null; wording: string; reply: string;
  checks?: { check: string; passed: boolean; detail?: string }[]; template?: string | null; topic?: string | null }
interface StaffFinding {
  id: number; part: string; price_key: string | null; damage: string; status: FindingStatusCode;
  reason_code: string | null; reason: string | null; severity: string | null; verification: string | null;
  vlm_valid_views: number | null; video_status: string | null; frames_seen: number[] | null;
  region: { frame: string; bbox: number[] } | null; evidence: string | null; group?: number;
}
interface Blocker { code: string; level: "review" | "escalate"; text: string; finding?: number }
/** Staff quote = the full pricing_rules output (the customer copy has these removed). */
type StaffQuote = Omit<Quote, "plain_reasons" | "pending_items" | "ruled_out_count"> & {
  notes: string[]; findings?: StaffFinding[]; blockers?: Blocker[]; overlap_groups?: number[][];
  scoped_offer?: { finding_ids: number[]; excluded_finding_ids: number[] } | null;
};

const STATUS: Record<FindingStatusCode, { label: string; tone: Tone }> = {
  confirmed_priceable: { label: "Priced", tone: "teal" },
  uncertain_review: { label: "Uncertain · review", tone: "amber" },
  unsupported_review: { label: "Needs inspection", tone: "red" },
  rejected: { label: "Ruled out", tone: "neutral" },
  duplicate_or_overlap: { label: "Overlap · review", tone: "amber" },
};

/** Blockers grouped by code: "VLM_INCONCLUSIVE ×2 — front bumper-dent, door-dent". */
function Blockers({ q }: { q: StaffQuote }) {
  const bl = q.blockers ?? [];
  if (bl.length === 0) return null;
  const by = new Map<string, Blocker[]>();
  bl.forEach((b) => by.set(b.code, [...(by.get(b.code) ?? []), b]));
  return (
    <div className="mt-4 border-t border-white/10 pt-4">
      <div className="text-sm font-medium text-slate-200">Blockers to automated settlement</div>
      <ul className="mt-2 space-y-2 text-sm">
        {[...by.entries()].map(([code, bs]) => (
          <li key={code} className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
            <Pill tone={bs[0].level === "escalate" ? "red" : "amber"}>{code.replace(/_/g, " ").toLowerCase()}{bs.length > 1 ? ` ×${bs.length}` : ""}</Pill>
            <span className="text-slate-300">{reasonText(bs[0].text)}{bs.length > 1 ? ` (+${bs.length - 1} more)` : ""}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function FindingTable({ q }: { q: StaffQuote }) {
  const fs = (q.findings ?? []).filter((f) => f.part !== "photo" || f.status !== "confirmed_priceable");
  if (fs.length === 0) return <p className="text-slate-400">No findings were passed to pricing.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px]">
        <thead className="border-y border-white/[0.06] bg-white/[0.02]"><tr>
          <Th>Finding</Th><Th>Status</Th><Th>Verification</Th><Th>Reason</Th>
        </tr></thead>
        <tbody className="divide-y divide-white/[0.04]">
          {fs.map((f) => (
            <tr key={f.id}>
              <Td><div className="font-medium text-slate-100"><span className="num text-slate-500">#{f.id + 1}</span> {title(f.part)}</div><div className="text-xs text-slate-500">{f.damage}{f.severity ? ` · ${f.severity}` : ""}{f.region ? ` · ${f.region.frame === "photo" ? "photo" : "frame " + f.region.frame.split(/[\\/]/).pop()}` : ""}</div></Td>
              <Td><Pill tone={STATUS[f.status].tone}>{STATUS[f.status].label}</Pill>{f.group !== undefined && <div className="mt-1 text-xs text-slate-500">overlap group {f.group + 1}</div>}</Td>
              <Td className="text-sm text-slate-300">{f.verification ?? "—"}{typeof f.vlm_valid_views === "number" ? ` · ${f.vlm_valid_views} valid view${f.vlm_valid_views === 1 ? "" : "s"}` : ""}{f.video_status ? ` · video ${f.video_status}` : ""}</Td>
              <Td className="text-sm text-slate-400">{f.reason ?? (f.status === "confirmed_priceable" ? "passed all checks" : "—")}{f.reason_code && <div className="num text-xs text-slate-600">{f.reason_code}</div>}</Td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface AdminClaim {
  claim_id: string;
  meta: { created_at: string; last_activity: string; plate: string | null; media_type: string; vehicle_type: string; findings: number; mode?: string; demo_tier?: string | null;
    sample?: string | null; photo_integrity?: { hard: string[]; soft: string[]; software: string | null; camera: string | null; taken: string | null } | null };
  policy: Policy | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  analysis: any;
  quote: StaffQuote;
  negotiation: { state: string; ladder_inr: number[] | null; step: number | null; current_offer_inr: number | null; final_notice: boolean; settled_amount_inr: number | null; history: Turn[];
    settlement_scope?: "partial" | "full" | null; settled_item_ids?: string[] | null;
    evidence?: EvidenceRec[] };
  ledger?: { item_id: string; signature: string[]; time: string; partial: boolean }[];
  audit: Record<string, unknown>[];
}

interface EvidenceRec {
  n: number; kind: "photo" | "document"; time: string; file: string; result?: string; outcome?: string;
  offer_before?: number | null; offer_after?: number | null; note?: string; extraction?: string | null;
  check?: { ok: boolean; problems: string[]; cap: string; authenticity: string; accepted: number;
    lines: { description: string; part: string | null; type: string; amount: number; status: string; reason?: string; ours?: number; cap_amount?: number; for_review?: boolean }[] } | null;
}

function ProofRow({ claimId, e }: { claimId: string; e: EvidenceRec }) {
  const [err, setErr] = useState<string | null>(null);
  const ch = e.check;
  return (
    <li className="py-3 text-sm">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <div className="min-w-0">
          <span className="font-medium text-slate-100">#{e.n} {e.kind === "document" ? "Garage estimate" : "Proof photo"}</span>
          <span className="text-slate-400"> · {e.result ?? "—"}</span>
          <div className="num text-xs text-slate-500">{when(e.time)}{e.extraction ? ` · reader: ${e.extraction}` : ""}{e.note ? ` · "${e.note}"` : ""}</div>
        </div>
        <div className="flex items-center gap-3">
          <span className="num text-slate-300">{inr(e.offer_before)} → {inr(e.offer_after)}</span>
          <button className="btn-quiet h-8 px-2 text-xs" onClick={() => { setErr(null); adminApi.openEvidenceFile(claimId, e.n).catch((x) => setErr((x as Error).message)); }}>
            <ExternalLink className="h-3.5 w-3.5" /> Open file</button>
        </div>
      </div>
      {err && <div className="mt-1 text-xs text-rose-300">{err}</div>}
      {ch && (
        <div className="mt-2 rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
          <div className="text-xs text-slate-400">Line check · cap {Math.round(parseFloat(ch.cap) * 100)}% of our rate (demo assumption) · {ch.accepted} line(s) accepted ·
            <span className="text-amber-200"> authenticity {ch.authenticity}</span></div>
          {ch.problems.length > 0 && <div className="mt-1 text-xs text-rose-200">Rejected document: {ch.problems.join("; ")}</div>}
          {ch.lines.length > 0 && (
            <table className="mt-2 w-full text-xs">
              <thead><tr className="text-left text-slate-500"><th className="py-1 pr-2">Line</th><th className="pr-2">Part · type</th><th className="pr-2 text-right">Garage</th><th className="pr-2 text-right">Ours</th><th className="pr-2">Result</th></tr></thead>
              <tbody>{ch.lines.map((l, i) => (
                <tr key={i} className="border-t border-white/[0.05] align-top text-slate-300">
                  <td className="py-1 pr-2">{l.description}</td><td className="pr-2">{l.part ?? "—"} · {l.type}</td>
                  <td className="num pr-2 text-right">{inr(l.amount)}</td><td className="num pr-2 text-right">{l.ours != null ? inr(l.ours) : "—"}</td>
                  <td className="pr-2"><span className="font-medium">{l.status.replace(/_/g, " ")}</span>{l.reason ? <span className="text-slate-500"> — {l.reason}</span> : null}{l.for_review ? <span className="text-amber-200"> · for specialist</span> : null}</td>
                </tr>))}</tbody>
            </table>
          )}
        </div>
      )}
    </li>
  );
}

function decision(c: AdminClaim): { tone: Tone; head: string; sub: string } {
  const q = c.quote.status, s = c.negotiation.state;
  if (s === "SETTLED" && c.negotiation.settlement_scope === "partial") return { tone: "amber", head: `Partially settled at ${inr(c.negotiation.settled_amount_inr)}`, sub: `Customer accepted item(s) ${(c.negotiation.settled_item_ids ?? []).join(", ")} only (demo, no payment). The remaining items are still under review and were not accepted or waived.` };
  if (s === "SETTLED") return { tone: "teal", head: `Settled at ${inr(c.negotiation.settled_amount_inr)}`, sub: "Customer accepted the authorised offer (demo, no payment)." };
  if (s === "ESCALATED") return { tone: "red", head: "Escalated by customer", sub: "Customer asked for a person or declined the final offer." };
  if (q === "APPROVED") return { tone: "teal", head: "Auto-approved", sub: `Cash offer ${inr(c.negotiation.current_offer_inr)} is with the customer.` };
  if (c.quote.scoped_offer) return { tone: "amber", head: "Partial demo offer + review", sub: `Demo Mode: offer ${inr(c.negotiation.current_offer_inr)} covers finding(s) ${c.quote.scoped_offer.finding_ids.map((i) => `#${i + 1}`).join(", ")} only; the rest stay under review.` };
  if (q === "NO_PAYOUT") return { tone: "neutral", head: "No payout", sub: "Repair estimate is within the compulsory excess." };
  const part = (c.quote.totals as PricingBreakdown).estimate_scope === "partial";
  if (q === "REVIEW_REQUIRED") return { tone: "amber", head: "Human review required", sub: part ? "Some findings are priced; the rest are pending. No offer until every finding is resolved." : "Automatic settlement paused by the pricing engine." };
  return { tone: "red", head: "Escalated to surveyor", sub: "Automatic settlement blocked by the pricing engine." };
}
const TONE_BOX: Record<Tone, string> = {
  teal: "border-teal-300/25 bg-teal-400/[0.06]", amber: "border-amber-300/25 bg-amber-400/[0.06]",
  red: "border-rose-300/25 bg-rose-400/[0.06]", neutral: "border-white/10 bg-white/[0.03]",
};
const TONE_TEXT: Record<Tone, string> = { teal: "text-teal-200", amber: "text-amber-200", red: "text-rose-200", neutral: "text-slate-200" };

/* ------------------------------ Summary ------------------------------ */
function Summary({ c, go }: { c: AdminClaim; go: (t: Tab) => void }) {
  const d = decision(c);
  const t = c.quote.totals as PricingBreakdown;
  const damages: DamageFinding[] = c.analysis?.damages ?? [];
  const p = c.policy;
  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
      <div className="space-y-6">
        <section className={cx("rounded-2xl border p-6", TONE_BOX[d.tone])}>
          <div className={cx("flex items-center gap-2 text-sm font-medium", TONE_TEXT[d.tone])}>
            {d.tone === "teal" ? <CheckCircle2 className="h-4 w-4" /> : <ShieldAlert className="h-4 w-4" />} Decision
          </div>
          <div className="mt-2 text-2xl font-semibold text-slate-50">{d.head}</div>
          <p className="mt-1 text-slate-300">{d.sub}</p>
          {c.quote.blockers ? <Blockers q={c.quote} /> : c.quote.status !== "APPROVED" && c.quote.reasons.length > 0 && (
            <ul className="mt-4 space-y-2 border-t border-white/10 pt-4 text-sm text-slate-300">
              {c.quote.reasons.map((r) => <li key={r} className="flex gap-2.5"><span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-slate-400" />{reasonText(r)}</li>)}
            </ul>
          )}
        </section>

        {(c.negotiation.evidence ?? []).length > 0 && (
          <Card title="Customer proof" sub="The offer changes only when proof changes the assessment. Garage estimates are checked line by line; the document itself is not authenticated.">
            <ul className="divide-y divide-white/[0.05]">
              {(c.negotiation.evidence ?? []).map((e) => <ProofRow key={e.n} claimId={c.claim_id} e={e} />)}
            </ul>
          </Card>
        )}
        <Card title="Findings" sub="What pricing did with each finding, and why." pad={false}
          right={<button className="btn-quiet" onClick={() => go("evidence")}>Inspect evidence</button>}>
          <FindingTable q={c.quote} />
        </Card>
      </div>

      <div className="space-y-6">
        <Card title="Money" sub="As computed by the pricing engine.">
          {typeof t.garage_estimate_inr !== "number" ? <p className="text-slate-400">Nothing could be priced automatically.</p> : (
            <dl className="divide-y divide-white/[0.05]">
              <Kv k="Repair estimate" v={inr(t.garage_estimate_inr)} />
              <Kv k="Depreciation" v={`−${inr(t.depreciation_inr)}`} />
              <Kv k="Compulsory excess" v={`−${inr(t.deductible_inr)}`} />
              <Kv k={t.estimate_scope === "partial" ? `Provisional subtotal (${t.items_pending} pending)` : "Net payable"} v={inr(t.insurer_payable_inr)} strong />
              {c.negotiation.current_offer_inr !== null && <Kv k={c.negotiation.state === "SETTLED" ? "Accepted" : "Current offer"} v={inr(c.negotiation.settled_amount_inr ?? c.negotiation.current_offer_inr)} />}
            </dl>
          )}
        </Card>
        <Card title="Vehicle and policy">
          <dl className="divide-y divide-white/[0.05]">
            {c.meta.mode === "demo" && <Kv k="Mode" v="Demo Mode · no policy, make not identified" />}
            {c.meta.mode === "demo" && <Kv k="Demo price tier" v={`${c.meta.demo_tier ?? "auto"} → ${t.price_tier ?? "—"}${t.tier_source ? ` (${t.tier_source})` : ""}`} />}
            <Kv k="Vehicle" v={p ? `${title(p.make)} ${p.model} · ${p.year}` : title(c.meta.vehicle_type ?? "")} />
            <Kv k="Registration" v={c.meta.plate ?? (c.meta.mode === "demo" ? "not used in Demo Mode" : "—")} />
            {p && <Kv k="Policy" v={`${p.policy_no} · ${p.holder}`} />}
            {p && <Kv k="Excess · zero-dep" v={`${inr(p.deductible_inr)} · ${p.zero_dep ? "yes" : "no"}`} />}
            <Kv k="Vehicle check" v={c.analysis?.vehicle?.mismatch ?? "consistent with policy"} />
            <Kv k="Licence plate (OCR)" v={c.analysis?.plate?.status === "identified"
              ? `${c.analysis.plate.text} · ${Math.round((c.analysis.plate.confidence ?? 0) * 100)}%${c.analysis.plate.registration_check ? ` · ${c.analysis.plate.registration_check === "match" ? "matches" : "DIFFERS from"} policy registration` : ""}`
              : `not identified${c.analysis?.plate?.reason ? ` (${c.analysis.plate.reason})` : ""}`} />
          </dl>
        </Card>
        <Card title="Processing">
          <dl className="divide-y divide-white/[0.05]">
            <Kv k="Evidence" v={c.meta.media_type === "video" ? `Video · ${c.analysis?.video_info?.keyframes ?? "?"} keyframes` : "Single photo"} />
            <Kv k="Submitted" v={when(c.meta.created_at)} />
            <Kv k="Last activity" v={when(c.meta.last_activity)} />
            <Kv k="Severity source" v={Array.from(new Set(damages.map((f) => f.severity_source).filter(Boolean))).join(", ") || "—"} />
            {c.meta.photo_integrity && <Kv k="Photo metadata" v={[
              c.meta.photo_integrity.hard.length ? `EDITED: ${c.meta.photo_integrity.software}` : null,
              c.meta.photo_integrity.camera ? `camera ${c.meta.photo_integrity.camera}` : null,
              c.meta.photo_integrity.taken ? `taken ${c.meta.photo_integrity.taken}` : null,
              ...c.meta.photo_integrity.soft].filter(Boolean).join(" · ") || "—"} />}
            {c.meta.sample && <Kv k="Recorded sample" v={`${c.meta.sample} (vision output recorded earlier)`} />}
          </dl>
        </Card>
      </div>
    </div>
  );
}

function Verdict({ f }: { f: DamageFinding }) {
  if (f.status) return <Pill tone={STATUS[f.status].tone}>{STATUS[f.status].label}</Pill>;
  if (f.vlm_confirmed === false) return <Pill tone="neutral">VLM: not damage</Pill>;
  if (f.part_uncertain) return <Pill tone="amber">Panel uncertain</Pill>;
  if (f.needs_review) return <Pill tone="amber">Needs review</Pill>;
  if (f.vlm_confirmed === true) return <Pill tone="teal">VLM verified</Pill>;
  return <Pill tone="neutral">Detector only</Pill>;
}

/* ------------------------------ Evidence ------------------------------ */
const FACTS: [string, string][] = [
  ["damage_visible", "Damage visible"], ["part_matches", "Part matches"], ["paint_broken", "Paint broken"], ["material_torn", "Material torn"],
  ["dent_depth", "Dent depth"], ["sharp_crease", "Sharp crease"], ["lamp_lens_broken", "Lamp lens broken"], ["lamp_housing_broken", "Lamp housing broken"],
  ["glass_shattered", "Glass shattered"], ["extent", "Extent"], ["image_quality", "Image quality"], ["reflection_or_dirt_suspected", "Reflection / dirt"],
];

function EvidenceTab({ c }: { c: AdminClaim }) {
  const [sel, setSel] = useState<Sel>(null);
  const a = c.analysis ?? {};
  const damages: DamageFinding[] = a.damages ?? [];
  const f = sel !== null ? damages[sel] : null;
  return (
    <div className="grid gap-6 2xl:grid-cols-[minmax(0,1.6fr)_minmax(380px,1fr)]">
      {a.viewer ? <Evidence claim={{ viewer: a.viewer, damages, images: a.images ?? [] }} selected={sel} onSelect={setSel} staff /> : <State empty="No evidence image stored for this claim." />}
      <div className="space-y-6">
        <Card title="Findings" sub="Select one to highlight it and see the VLM's answers." pad={false}>
          <ul className="divide-y divide-white/[0.05] border-t border-white/[0.05]">
            {damages.map((d, i) => (
              <li key={i}>
                <button onClick={() => setSel(sel === i ? null : i)} aria-pressed={sel === i}
                  className={cx("w-full px-6 py-4 text-left transition", sel === i ? "bg-teal-400/[0.07]" : "hover:bg-white/[0.03]")}>
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="font-medium text-slate-100">{title(d.part)} · <span className="font-normal text-slate-300">{d.damage}</span></div>
                      <div className="num mt-1 text-xs text-slate-500">
                        detector {typeof d.detector_confidence === "number" ? d.detector_confidence.toFixed(2) : "—"} · severity {d.severity ?? "—"} ({d.severity_source ?? "—"})
                        {d.vlm_verification ? ` · VLM ${d.vlm_verification}` : ""}
                        {typeof d.vlm_valid_views === "number" ? ` · ${d.vlm_valid_views} valid view(s)` : ""}
                        {typeof d.vlm_agreement === "number" ? ` · ${Math.round(d.vlm_agreement * 100)}% of valid views agree` : ""}
                      </div>
                    </div>
                    <Verdict f={d} />
                  </div>
                </button>
              </li>
            ))}
            {damages.length === 0 && <li className="px-6 py-6 text-slate-400">No findings.</li>}
          </ul>
        </Card>
        {f && (
          <Card title={`${title(f.part)} — ${f.damage}`} sub={f.severity_reasons?.join("; ")}>
            {f.severity_facts ? (
              <dl className="grid grid-cols-2 gap-x-6 gap-y-1">
                {FACTS.map(([k, label]) => {
                  const v = f.severity_facts?.[k];
                  return (
                    <div key={k} className="flex items-center justify-between gap-3 py-1.5 text-sm">
                      <dt className="text-slate-400">{label}</dt>
                      <dd className={cx("num", v === true ? "text-teal-200" : v === false ? "text-slate-500" : "text-slate-200")}>{v === true ? "yes" : v === false ? "no" : String(v ?? "—")}</dd>
                    </div>
                  );
                })}
              </dl>
            ) : <p className="text-sm text-slate-400">{f.vlm_verification && f.vlm_verification !== "not_run" ? `No accepted VLM facts for this finding (verification: ${f.vlm_verification}).` : "The VLM did not run for this finding (severity from measured damage area)."}</p>}
            {(f as DamageFinding & { severity_votes?: VoteRec[] }).severity_votes && <Votes votes={(f as DamageFinding & { severity_votes: VoteRec[] }).severity_votes} />}
          </Card>
        )}
      </div>
    </div>
  );
}

interface VoteRec { view: number; visible?: boolean; error?: string; raw?: string[] }
const VIEW_NAME = ["tight crop", "normal crop", "wide crop"];
/** Raw VLM replies per crop - staff diagnostics only (stripped from customer responses). */
function Votes({ votes }: { votes: VoteRec[] }) {
  return (
    <details className="mt-4 border-t border-white/[0.06] pt-3 text-sm">
      <summary className="cursor-pointer select-none text-slate-300">VLM answers per crop ({votes.filter((v) => !v.error).length} of {votes.length} valid)</summary>
      <ul className="mt-2 space-y-3">
        {votes.map((v) => (
          <li key={v.view}>
            <div className={v.error ? "text-rose-200" : "text-slate-200"}>{VIEW_NAME[v.view] ?? `view ${v.view + 1}`}: {v.error ? `invalid — ${v.error}` : v.visible ? "damage visible" : "no damage"}</div>
            {v.raw?.map((r, i) => <pre key={i} className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded-lg bg-black/30 p-2 text-xs text-slate-400">{r}</pre>)}
          </li>
        ))}
      </ul>
    </details>
  );
}

/* ------------------------------ Pricing ------------------------------ */
function source(l: LineItem) {
  const s = l.part_price_source;
  if (s.type === "scraped") return { label: s.secondary_source ? "OEM price (secondary source)" : "OEM price", detail: `${s.price_type} · ${s.part_numbers?.join(", ") ?? ""} · ${s.sites?.join(", ")} · median of ${s.n} · ${shortDate(s.scraped_at)}${s.stale ? " · stale" : ""}`, href: s.urls?.[0] };
  if (s.type === "demo_estimate") return { label: "Rate card", detail: `demo tier price ${inr(s.tier_price)} × ${s.multiplier}` };
  return { label: "—", detail: "no new part" };
}

function PricingTab({ c }: { c: AdminClaim }) {
  const q = c.quote;
  const t = q.totals as PricingBreakdown;
  const excluded = (q.findings ?? []).filter((f) => f.status !== "confirmed_priceable");
  if (q.line_items.length === 0) return (
    <div className="space-y-6">
      <State empty="No line items were priced for this claim." />
      {excluded.length > 0 && <Card title="Not priced" sub="Findings kept out of the estimate, with the reason." pad={false}><FindingTable q={{ ...q, findings: excluded }} /></Card>}
    </div>
  );
  return (
    <div className="space-y-6">
      {t.estimate_scope === "partial" && (
        <div className="rounded-2xl border border-amber-300/25 bg-amber-400/[0.06] px-5 py-4 text-sm text-amber-100">
          Provisional partial estimate: {t.items_included} item(s) priced, {t.items_pending} pending review and not included. No offer is made until every finding is resolved.
        </div>
      )}
      <Card title="Line items" sub={`Price tier ${t.price_tier ?? "—"} · make ${t.make_used ?? "—"}`} pad={false}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[880px]">
            <thead className="border-y border-white/[0.06] bg-white/[0.02]"><tr>
              <Th>Part</Th><Th>Repair method</Th><Th r>Parts</Th><Th r>Labour</Th><Th r>Paint</Th><Th r>Line total</Th><Th r>Depreciation</Th><Th>Part price source</Th>
            </tr></thead>
            <tbody className="divide-y divide-white/[0.04]">
              {q.line_items.map((l) => {
                const s = source(l);
                return (
                  <tr key={l.part}>
                    <Td><div className="font-medium text-slate-100">{title(l.part)}</div><div className="mt-0.5 text-xs text-slate-500">{l.damage} · {l.severity} · {l.material}{l.also_covers?.length ? ` · covers ${l.also_covers.join(", ")}` : ""}</div></Td>
                    <Td className="text-slate-300">{methodLabel(l.method)}{l.lesser_of ? <div className="mt-0.5 text-xs text-amber-200/80">lesser-of: replacing (net {inr(l.lesser_of.replace_net)}) is cheaper than {methodLabel(l.lesser_of.instead_of).toLowerCase()} (net {inr(l.lesser_of.repair_net)})</div> : null}
                      {l.provenance && <div className="mt-1 flex flex-wrap gap-1">{Object.entries(l.provenance).map(([k, v]) => {
                        const tag = v.split(" ")[0];
                        return <span key={k} title={v} className={cx("rounded-full px-2 py-0.5 text-[11px]", tag === "OBSERVED" ? "bg-teal-400/15 text-teal-200" : tag === "RULE" ? "bg-sky-400/15 text-sky-200" : tag === "IMPUTED" ? "bg-amber-400/15 text-amber-200" : "bg-white/5 text-slate-500")}>{k.replace("_", " ")}: {tag}</span>;
                      })}</div>}
                    </Td>
                    <Td r>{inr(l.part_cost)}</Td><Td r>{inr(l.labour)}</Td><Td r>{inr(l.paint)}</Td>
                    <Td r className="font-medium text-slate-50">{inr(l.garage_cost)}</Td>
                    <Td r className="text-slate-300">{l.depreciation_pct}% · −{inr(l.depreciation)}</Td>
                    <Td>
                      <details className="text-sm">
                        <summary className={cx("cursor-pointer select-none", s.label === "OEM price" ? "text-sky-200" : "text-slate-300")}>{s.label}</summary>
                        <div className="mt-1.5 text-xs text-slate-500">{s.detail}{s.href && <a href={s.href} target="_blank" rel="noreferrer" className="ml-1 inline-flex items-center gap-0.5 text-teal-300 hover:underline">source <ExternalLink className="h-3 w-3" /></a>}</div>
                      </details>
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Totals">
          <dl className="divide-y divide-white/[0.05]">
            <Kv k="Gross repair estimate" v={inr(t.garage_estimate_inr)} />
            <Kv k={`Material depreciation${t.zero_depreciation_cover ? " (zero-dep cover)" : ""}`} v={`−${inr(t.depreciation_inr)}`} />
            <Kv k="Compulsory excess" v={`−${inr(t.deductible_inr)}`} />
            <Kv k={t.estimate_scope === "partial" ? "Provisional subtotal (partial)" : "Net payable"} v={inr(t.insurer_payable_inr)} strong />
          </dl>
        </Card>
        <Card title="Rules applied">
          <ul className="space-y-2.5 text-sm leading-relaxed text-slate-400">
            <li>One repair per part; the most expensive method covers the other damage on that part.</li>
            <li>Depreciation on new parts only: plastic/rubber 50%, glass 0%, metal by vehicle age; waived with zero-dep cover.</li>
            <li>Compulsory excess by vehicle type and engine capacity.</li>
            <li>{t.parts_priced_from_real_data} part price(s) from scraped OEM data, {t.parts_priced_from_demo_estimates} from the demo rate card; labour and paint from the demo rate card.</li>
            {(q.notes ?? []).map((n) => <li key={n} className="text-amber-200/80">{n}</li>)}
          </ul>
        </Card>
      </div>
      {excluded.length > 0 && <Card title="Not priced" sub="Findings kept out of the estimate, with the reason. Overlap groups are one damage region seen on two panels." pad={false}><FindingTable q={{ ...q, findings: excluded }} /></Card>}
    </div>
  );
}

/* ------------------------------ Negotiation ------------------------------ */
function NegotiationTab({ c }: { c: AdminClaim }) {
  const n = c.negotiation;
  const rng = (c.quote.totals as PricingBreakdown).settlement_range;
  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
      <div className="space-y-6">
        <Card title="Authorised offer" sub="Evidence-based: one fixed share of the assessed payable; it changes only when proof changes the assessment. Never shown as a range to the customer.">
          {!rng || !n.ladder_inr ? <p className="text-slate-400">No cash offer was authorised for this claim.</p> : (
            <>
              <ol className="space-y-3">
                {n.ladder_inr.map((v, i) => {
                  const reached = (n.step ?? 0) >= i;
                  const current = (n.step ?? 0) === i && n.state !== "SETTLED";
                  return (
                    <li key={i} className={cx("flex items-center justify-between rounded-xl border px-4 py-3", current ? "border-teal-300/40 bg-teal-400/[0.07]" : "border-white/[0.06]")}>
                      <span className={cx("text-sm", reached ? "text-slate-200" : "text-slate-500")}>{n.ladder_inr!.length === 1 ? "Cash offer (fixed share of payable)" : <>Step {i + 1}{i === 0 ? " · opening" : i === n.ladder_inr!.length - 1 ? " · ceiling" : ""}</>}</span>
                      <span className={cx("num font-medium", reached ? "text-slate-50" : "text-slate-500")}>{inr(v)}</span>
                    </li>
                  );
                })}
              </ol>
              <dl className="mt-5 divide-y divide-white/[0.05]">
                <Kv k="State" v={n.state} />
                <Kv k="Current offer" v={inr(n.current_offer_inr)} />
                <Kv k="Final-offer notice given" v={n.final_notice ? "yes" : "no"} />
                <Kv k="Accepted amount" v={inr(n.settled_amount_inr)} />
              </dl>
            </>
          )}
        </Card>
      </div>
      <Card title="Conversation" sub="Every reply's amount was authorised by the state machine before sending.">
        {n.history.length === 0 ? (
          <div className="flex flex-col items-center gap-3 py-12 text-center text-slate-400"><MessageSquareDashed className="h-6 w-6" />The customer has not sent any messages.</div>
        ) : (
          <ol className="space-y-5">
            {n.history.map((h, i) => (
              <li key={i} className="space-y-2">
                <div className="flex justify-end"><div className="max-w-[85%] rounded-2xl rounded-br-md bg-teal-400/12 bg-teal-400/[0.12] px-4 py-2.5 text-slate-100"><User className="mr-1.5 inline h-3.5 w-3.5 opacity-60" />{h.user}</div></div>
                <div className="flex"><div className="max-w-[85%] rounded-2xl rounded-bl-md bg-white/[0.05] px-4 py-2.5 text-slate-200"><Bot className="mr-1.5 inline h-3.5 w-3.5 opacity-60" />{h.reply}</div></div>
                <div className="num text-xs text-slate-500">{when(h.time)} · intent {h.intent} ({h.intent_source}) → {h.outcome.replace(/_/g, " ").toLowerCase()} · state {h.state}{h.amount_inr ? ` · amount ${inr(h.amount_inr)}` : ""} · {h.wording}</div>
                <details className="text-xs text-slate-400">
                  <summary className="cursor-pointer select-none text-slate-500">Behind this reply</summary>
                  <div className="mt-1.5 space-y-1 rounded-lg bg-white/[0.02] p-3">
                    <div>Intent from {h.intent_source}{h.topic ? ` · topic ${h.topic}` : ""} · amount set by the state machine, not by the text</div>
                    {h.wording === "template" ? <div>Wording: fixed template written by code (no language model).</div> : (
                      <>
                        <div>Wording: {h.wording}</div>
                        {h.template && <div>Engine's text: <span className="text-slate-300">{h.template}</span></div>}
                        {(h.checks ?? []).map((c) => (
                          <div key={c.check} className={c.passed ? "text-teal-200/90" : "text-rose-200"}>{c.passed ? "✓" : "✗"} {c.check.replace(/_/g, " ")}{c.detail ? <span className="text-slate-500"> — {c.detail}</span> : null}</div>
                        ))}
                      </>
                    )}
                  </div>
                </details>
              </li>
            ))}
          </ol>
        )}
      </Card>
    </div>
  );
}

/* ------------------------------ Workspace ------------------------------ */
export default function ClaimWorkspace({ id, tab, onAuth, nav }: { id: string; tab: Tab; onAuth: () => void; nav: (p: string) => void }) {
  const { data: c, err } = useLoad<AdminClaim>(() => adminApi.claim(id), [id], onAuth);
  if (!c) return <State err={err} />;
  const st = claimTone(c.quote.status, c.negotiation.state, c.negotiation.settlement_scope);
  const go = (t: Tab) => nav(`/admin/claims/${id}/${t}`);
  const p = c.policy;
  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
        <button onClick={() => nav("/admin/claims")} className="btn-quiet h-10 w-10 px-0 ring-1 ring-inset ring-white/10" aria-label="Back to claim queue"><ArrowLeft className="h-4 w-4" /></button>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="num text-2xl font-semibold text-slate-50">{ref(id)}</h1>
            <Pill tone={st.tone} dot>{st.label}</Pill>
          </div>
          <div className="mt-1 text-sm text-slate-400">{p ? `${title(p.make)} ${p.model} · ` : ""}{title(c.meta.vehicle_type ?? "")} · {c.meta.plate ?? "no plate"} · submitted {when(c.meta.created_at)}</div>
        </div>
      </div>

      <nav className="sticky top-0 z-20 -mx-5 mt-6 overflow-x-auto border-b border-white/[0.07] bg-ink-950/90 px-5 backdrop-blur lg:-mx-10 lg:px-10" aria-label="Claim sections">
        <div className="flex gap-1">
          {TABS.map(([k, label]) => (
            <button key={k} onClick={() => go(k)} aria-current={tab === k ? "page" : undefined}
              className={cx("relative whitespace-nowrap px-4 py-3.5 text-sm font-medium transition",
                tab === k ? "text-slate-50" : "text-slate-400 hover:text-slate-200")}>
              {label}
              {tab === k && <span className="absolute inset-x-3 -bottom-px h-0.5 rounded-full bg-teal-300" />}
            </button>
          ))}
        </div>
      </nav>

      <div className="fade-in pt-6" key={tab}>
        {tab === "summary" && <Summary c={c} go={go} />}
        {tab === "evidence" && <EvidenceTab c={c} />}
        {tab === "pricing" && <PricingTab c={c} />}
        {tab === "negotiation" && <NegotiationTab c={c} />}
        {tab === "audit" && <Card title="Audit trail" sub="Recorded by the backend for this claim. Newest first."><AuditList rows={[...c.audit].reverse() as never} hideClaim /></Card>}
      </div>
    </div>
  );
}
