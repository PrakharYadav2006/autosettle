import { useEffect, useRef } from "react";
import type { ClaimState } from "../types";
import { inr, title } from "../format";

export default function SettlementModal({ claim, onClose }: { claim: ClaimState; onClose: () => void }) {
  const btn = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    btn.current?.focus();
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);
  const sc = claim.quote.offer_scope;
  const partial = claim.offer.settlement_scope === "partial" || !!sc?.partial;
  const at = claim.offer.decision_recorded_at ? new Date(claim.offer.decision_recorded_at).toLocaleString("en-IN") : "—";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-5 backdrop-blur-sm" onClick={onClose}>
      <div role="dialog" aria-modal="true" aria-labelledby="settle-title" className="panel fade-in w-full max-w-md overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="bg-amber-300 px-4 py-2 text-center text-xs font-bold uppercase tracking-[0.14em] text-ink-950">Demo — No money transferred</div>
        <div className="p-6 sm:p-8">
          <h2 id="settle-title" className="text-xl font-semibold text-slate-50">Your decision is recorded</h2>
          <div className="num mt-5 text-4xl font-semibold text-slate-50">{inr(claim.offer.settled_amount_inr)}</div>
          {partial ? (
            <p className="mt-2 text-slate-300">Your acceptance of {inr(claim.offer.settled_amount_inr)} is recorded for the listed items only. The remaining items stay under review.</p>
          ) : <p className="mt-1 text-slate-400">accepted as full and final settlement</p>}
          {partial && sc && (
            <div className="mt-4 space-y-1 rounded-xl border border-amber-300/20 bg-amber-400/[0.05] px-4 py-3 text-sm text-slate-300">
              <p><span className="text-slate-400">Settled:</span> {sc.included.map((x) => `${title(x.part)} (${x.damage})`).join(", ")}</p>
              {sc.excluded.length > 0 && <p><span className="text-slate-400">Still under review (not accepted, not waived):</span> {sc.excluded.map((x) => `${title(x.part)} (${x.damage})`).join(", ")}</p>}
            </div>
          )}
          <dl className="mt-6 divide-y divide-white/[0.06]">
            {[
              ["Claim reference", `AS-${claim.claim_id.slice(0, 6).toUpperCase()}`],
              ["Decision", partial ? "Partially settled — remaining items under review" : "Offer accepted"],
              ["Recorded at", at],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-4 py-3 text-sm"><dt className="text-slate-400">{k}</dt><dd className="num text-right text-slate-200">{v}</dd></div>
            ))}
          </dl>
          <p className="mt-5 text-sm leading-relaxed text-slate-400">
            Next step in a live service: the insurer verifies your bank details and releases the payment. In this demo nothing is paid and no bank is contacted.
          </p>
          <button ref={btn} className="btn-primary mt-6 w-full" onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  );
}
