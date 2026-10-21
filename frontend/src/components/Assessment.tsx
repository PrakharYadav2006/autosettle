import { useState } from "react";
import { AlertTriangle, ArrowRight, ImageOff } from "lucide-react";
import type { BoundingBox, ClaimState, DamageFinding, OfferBlock } from "../types";
import { cx, inr, title } from "../format";
import { findingStatus } from "../status";
import { Pill, Segmented, Toggle } from "./ui";

const BOX = {
  part: { stroke: "rgba(148,163,184,0.75)", label: "bg-slate-200 text-ink-950" },
  damage: { stroke: "#2dd4bf", label: "bg-teal-300 text-ink-950" },
  vlm: { stroke: "#fbbf24", label: "bg-amber-300 text-ink-950" },
};

/** Selection is per finding (index into claim.damages), so two findings on the same
 *  part (e.g. three scratches on one bumper) each highlight their own box and crop. */
export type Sel = number | null;

function boxFinding(b: BoundingBox, damages: DamageFinding[]): number | null {
  if (b.kind === "damage") { const i = Number(b.id.slice(1)); return Number.isFinite(i) && i < damages.length ? i : null; }
  const i = damages.findIndex((d) => d.price_key === b.price_key && (b.kind === "part" || d.severity_source === "vlm_only"));
  return i >= 0 ? i : null;
}

function cropFor(i: number, claim: Pick<ClaimState, "damages" | "images">) {
  const d = claim.damages[i];
  const name = `${d.part} - ${d.damage}`.toLowerCase();
  const nth = claim.damages.slice(0, i).filter((x) => `${x.part} - ${x.damage}`.toLowerCase() === name).length;
  return claim.images.filter((im) => im.image && im.name.toLowerCase().startsWith(name))[nth];
}

export function Evidence({ claim, selected, onSelect, staff }: {
  claim: Pick<ClaimState, "viewer" | "damages" | "images">; selected: Sel; onSelect: (i: Sel) => void; staff?: boolean;
}) {
  const [mode, setMode] = useState<"annotated" | "original">("annotated");
  const [parts, setParts] = useState(!!staff);
  const [damage, setDamage] = useState(true);
  const v = claim.viewer;
  const sel = selected !== null ? claim.damages[selected] : undefined;
  const crop = selected !== null && sel ? cropFor(selected, claim) : undefined;

  if (!v) {
    return (
      <section className="panel flex min-h-[320px] flex-col items-center justify-center gap-3 p-8 text-center text-slate-400">
        <ImageOff className="h-6 w-6" /> No evidence image is available for this claim.
      </section>
    );
  }
  const on = mode === "annotated";
  const boxes = v.boxes.filter((b) => on && ((b.kind === "part" && parts) || (b.kind !== "part" && damage)));
  const isHit = (b: BoundingBox) => {
    if (selected === null || !sel) return false;
    if (b.kind === "damage") return b.id === `d${selected}`;
    return b.price_key === sel.price_key && (b.kind === "part" || sel.severity_source === "vlm_only");
  };
  const showLabel = (b: BoundingBox) => b.kind !== "part" || isHit(b) || !!staff;
  const ratio = v.width / Math.max(v.height, 1);

  return (
    <section className="panel overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-4">
        <Segmented size="sm" value={mode} onChange={setMode}
          options={[{ value: "annotated", label: "Annotated" }, { value: "original", label: "Original" }]} />
        <div className={cx("flex items-center gap-5", !on && "pointer-events-none opacity-40")}>
          <Toggle checked={damage} onChange={setDamage} label="Damage" />
          <Toggle checked={parts} onChange={setParts} label="Parts" />
        </div>
      </div>

      {/* The frame keeps the photo's aspect ratio and always fills the available width
          (up to 64vh tall), so small source photos are scaled up instead of rendering at
          their tiny native size. The overlay uses the same frame, so boxes stay aligned. */}
      <div className="flex justify-center bg-black/40" onClick={() => onSelect(null)}>
        <div className="relative" style={{ aspectRatio: `${v.width} / ${v.height}`, width: `min(100%, calc(64vh * ${ratio.toFixed(4)}))` }}>
          <img src={v.image} alt="Vehicle evidence" draggable={false} className="absolute inset-0 h-full w-full select-none object-fill" />
          <svg className="absolute inset-0 h-full w-full" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden>
            {boxes.map((b) => {
              const [x1, y1, x2, y2] = b.bbox.map((n) => n * 1000);
              const hit = isHit(b);
              const dim = selected !== null && !hit;
              return (
                <rect key={b.id} x={x1} y={y1} width={x2 - x1} height={y2 - y1} rx={4}
                  fill={hit ? "rgba(45,212,191,0.14)" : "transparent"} stroke={BOX[b.kind].stroke}
                  strokeWidth={hit ? 3 : b.kind === "part" ? 1 : 2} vectorEffect="non-scaling-stroke"
                  strokeDasharray={b.kind === "vlm" ? "6 4" : b.kind === "part" ? "4 3" : undefined}
                  opacity={dim ? 0.3 : 1} className="cursor-pointer"
                  onClick={(e) => { e.stopPropagation(); onSelect(boxFinding(b, claim.damages)); }} />
              );
            })}
          </svg>
          {boxes.filter(showLabel).map((b) => (
            <span key={b.id + "l"}
              style={b.bbox[0] > 0.6 ? { right: `${(1 - b.bbox[2]) * 100}%`, top: `${b.bbox[1] * 100}%` } : { left: `${b.bbox[0] * 100}%`, top: `${b.bbox[1] * 100}%` }}
              className={cx("pointer-events-none absolute -translate-y-[calc(100%+3px)] whitespace-nowrap rounded-md px-1.5 py-0.5 text-xs font-semibold shadow",
                BOX[b.kind].label, selected !== null && !isHit(b) && "opacity-40",
                // phones: only the selected label, so labels never pile up over the photo
                !isHit(b) && "hidden sm:block")}>
              {b.kind === "part" ? title(b.label) : `${title(b.label)} · ${title(b.part ?? "")}`}
              {staff && b.confidence !== null && <span className="ml-1 font-normal opacity-70">{b.confidence.toFixed(2)}</span>}
            </span>
          ))}
        </div>
      </div>

      <div className="border-t border-white/[0.06] p-5">
        {sel ? (
          <div className="fade-in grid gap-5 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] sm:items-start">
            {crop?.image ? (
              <div className="flex items-center justify-center overflow-hidden rounded-xl bg-black/40">
                <img src={crop.image} alt={`Close-up: ${sel.part}, ${sel.damage}`} className="max-h-64 w-full object-contain" />
              </div>
            ) : (
              <div className="flex h-40 items-center justify-center rounded-xl bg-white/[0.03] px-4 text-center text-sm text-slate-400">
                No close-up for this finding. Its area is highlighted on the photo above.
              </div>
            )}
            <div className="min-w-0">
              <div className="text-lg font-semibold text-slate-50">{title(sel.part)}</div>
              <div className="mt-0.5 text-slate-300">{plain(sel)}</div>
              <p className="mt-3 text-sm leading-relaxed text-slate-400">{describe(sel)}</p>
              <button onClick={() => onSelect(null)} className="btn-quiet -ml-3 mt-2">Clear selection</button>
            </div>
          </div>
        ) : (
          <p className="text-sm text-slate-400">
            {v.caption ? `Best frame: ${v.caption}. ` : ""}Select a finding or a highlighted area to inspect it up close.
          </p>
        )}
      </div>
    </section>
  );
}

const DAMAGE_WORD: Record<string, string> = {
  dent: "Dent", scratch: "Scratches", crack: "Crack", "lamp broken": "Broken lamp", "glass shatter": "Shattered glass", "tire flat": "Flat tyre",
};

/** "Crack detected on the front bumper · minor" - plain language from real fields only. */
function plain(d: DamageFinding): string {
  const what = d.damage ? (DAMAGE_WORD[d.damage] ?? title(d.damage)) : "Possible damage";
  const sev = d.severity ? ({ low: "minor", medium: "moderate", high: "severe" } as const)[d.severity] : null;
  return `${what} ${d.vlm_confirmed === false ? "suspected" : "detected"} on the ${d.part.toLowerCase()}${sev ? ` · ${sev}` : ""}`;
}

function describe(d: DamageFinding): string {
  const sev = d.severity ? `${d.severity[0].toUpperCase()}${d.severity.slice(1)} severity` : "Severity not graded";
  const why = d.severity_reasons?.filter(Boolean).join("; ");
  return why ? `${sev} — ${why}.` : `${sev}.`;
}

function Findings({ claim, selected, onSelect }: { claim: ClaimState; selected: Sel; onSelect: (i: Sel) => void }) {
  return (
    <section className="panel">
      <div className="flex items-baseline justify-between px-5 pb-2 pt-5">
        <h2 className="text-base font-semibold text-slate-50">Findings</h2>
        <span className="text-sm text-slate-400">{claim.damages.length} item{claim.damages.length === 1 ? "" : "s"}</span>
      </div>
      {claim.vehicle.mismatch && (
        <div className="mx-5 mb-2 mt-1 flex gap-3 rounded-xl bg-amber-400/10 px-4 py-3 text-sm text-amber-100">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> Vehicle check: {claim.vehicle.mismatch}.
        </div>
      )}
      {claim.damages.length === 0 ? (
        <p className="px-5 pb-6 pt-2 text-sm text-slate-400">No damage was located on an identifiable vehicle part.</p>
      ) : (
        <ul className="px-2 pb-2">
          {claim.damages.map((d, i) => {
            const st = findingStatus(d);
            const active = selected === i;
            return (
              <li key={i}>
                <button onClick={() => onSelect(active ? null : i)} aria-pressed={active}
                  className={cx("w-full rounded-xl px-3 py-3.5 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-300/60",
                    active ? "bg-teal-400/[0.08]" : "hover:bg-white/[0.03]")}>
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="font-medium text-slate-100">{title(d.part)}</div>
                      <div className="mt-0.5 text-sm text-slate-400">{plain(d)}</div>
                    </div>
                    <Pill tone={st.tone}>{st.label}</Pill>
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

function SummaryTeaser({ claim, offer, onOpen }: { claim: ClaimState; offer: OfferBlock; onOpen: () => void }) {
  const settled = offer.state === "SETTLED";
  const live = offer.state === "OPEN" || offer.state === "CONFIRMING";
  return (
    <section className="panel p-5 sm:p-6">
      {settled || live ? (
        <>
          <div className="text-sm text-slate-400">{settled ? (offer.settlement_scope === "partial" ? "Partially settled — remaining items under review" : "Offer accepted") : claim.quote.offer_scope?.partial ? "Partial demo offer · listed items only" : claim.demo ? "Your demo cash offer" : "Your cash settlement offer"}</div>
          <div className="num mt-1 text-3xl font-semibold text-slate-50">{inr(settled ? offer.settled_amount_inr : offer.current_offer_inr)}</div>
          <p className="mt-2 text-slate-400">{settled ? (offer.settlement_scope === "partial" ? "Recorded for the listed items only; the rest stays under review." : "Your decision is recorded.") : "See how it was calculated before you decide."}</p>
        </>
      ) : (
        <>
          <div className="font-medium text-slate-100">{claim.quote.status === "NO_PAYOUT" ? "No payable amount" : "Specialist review"}</div>
          <p className="mt-1.5 text-slate-400">{claim.quote.status === "NO_PAYOUT" ? "The repair estimate is within your policy excess."
            : (claim.quote.pending_items?.length ?? 0) > 0 ? `Part of the damage could be estimated; ${claim.quote.pending_items.length} item(s) need a specialist's check. No offer is made until the review is done.`
            : "An instant offer isn't available while a specialist reviews this claim."}</p>
        </>
      )}
      <button className="btn-primary mt-5 w-full" onClick={onOpen}>{live ? "Review estimate & offer" : "View estimate"} <ArrowRight className="h-4 w-4" /></button>
    </section>
  );
}

export default function Assessment({ claim, offer, onOpenSettlement }: { claim: ClaimState; offer: OfferBlock; onOpenSettlement: () => void }) {
  const [selected, setSelected] = useState<Sel>(null);
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.75fr)_minmax(320px,1fr)] lg:items-start">
      <Evidence claim={claim} selected={selected} onSelect={setSelected} />
      <div className="space-y-6">
        <Findings claim={claim} selected={selected} onSelect={setSelected} />
        <SummaryTeaser claim={claim} offer={offer} onOpen={onOpenSettlement} />
      </div>
    </div>
  );
}
