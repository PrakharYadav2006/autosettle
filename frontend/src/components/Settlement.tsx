import { useEffect, useRef, useState } from "react";
import { MessageCircleQuestion, RefreshCcw, RotateCw, SendHorizontal, UserRound } from "lucide-react";
import type { ChatMessage, ClaimState, LineItem, OfferBlock, PricingBreakdown } from "../types";
import { cx, inr, methodLabel, prettyReason, shortDate, title } from "../format";
import { findingStatus } from "../status";
import { Disclosure, Money, Pill } from "./ui";

/* ------------------------------------------------------------------ */
/* Repair estimate - a financial statement, values exactly as returned */
/* ------------------------------------------------------------------ */

function sourceNote(l: LineItem): string | null {
  const s = l.part_price_source;
  if (s.type === "scraped") {
    const site = s.sites?.[0]?.replace(/^www\./, "") ?? "the manufacturer";
    return `${title(l.part)}: manufacturer list price from ${site} (${shortDate(s.scraped_at)}).`;
  }
  return null;
}

function Statement({ claim, id }: { claim: ClaimState; id: string }) {
  const q = claim.quote;
  const t = q.totals as PricingBreakdown;
  const has = typeof t.garage_estimate_inr === "number";
  const deps = q.line_items.filter((l) => l.depreciation > 0);
  const sources = q.line_items.map(sourceNote).filter(Boolean) as string[];
  const cc = claim.policy?.engine_cc;

  return (
    <section id={id} className="panel scroll-mt-24 p-5 sm:p-7">
      <h2 className="text-lg font-semibold text-slate-50">Repair estimate</h2>
      <p className="mt-1 text-slate-400">What a garage would charge for these repairs, and what your policy covers.</p>

      {q.line_items.length === 0 ? (
        <p className="mt-6 rounded-xl bg-white/[0.03] px-4 py-5 text-slate-400">No repair could be priced automatically for this claim.</p>
      ) : (
        <ul className="mt-6 divide-y divide-white/[0.06] border-y border-white/[0.06]">
          {q.line_items.map((l) => (
            <li key={l.part} className="flex items-start justify-between gap-4 py-4">
              <div className="min-w-0">
                <div className="font-medium text-slate-100">{title(l.part)}</div>
                <div className="mt-0.5 text-sm text-slate-400">
                  {methodLabel(l.method)}{l.also_covers?.length ? ` (also covers ${l.also_covers.join(", ")})` : ""}
                </div>
                <div className="num mt-1 text-sm text-slate-500">
                  {[l.part_cost ? `Part ${inr(l.part_cost)}` : null, `Labour ${inr(l.labour)}`, l.paint ? `Paint ${inr(l.paint)}` : null].filter(Boolean).join(" · ")}
                </div>
              </div>
              <Money v={l.garage_cost} className="shrink-0 text-slate-100" />
            </li>
          ))}
        </ul>
      )}

      {has && (
        <dl className="mt-1">
          <div className="flex items-baseline justify-between gap-4 py-3.5">
            <dt className="font-medium text-slate-100">Total repair estimate</dt>
            <dd><Money v={t.garage_estimate_inr} className="font-medium text-slate-100" /></dd>
          </div>
          <div className="flex items-baseline justify-between gap-4 py-3">
            <dt>
              <div className="text-slate-300">Depreciation</div>
              <div className="mt-0.5 text-sm text-slate-500">
                {t.zero_depreciation_cover ? "Waived by your zero-depreciation cover"
                  : deps.length ? deps.map((l) => `${l.depreciation_pct}% on the new ${l.material} ${title(l.part).toLowerCase()}`).join(", ")
                  : "Not applicable to these repairs"}
              </div>
            </dt>
            <dd className="num text-slate-300">−{inr(t.depreciation_inr)}</dd>
          </div>
          <div className="flex items-baseline justify-between gap-4 py-3">
            <dt>
              <div className="text-slate-300">Compulsory excess</div>
              <div className="mt-0.5 text-sm text-slate-500">Set by your policy{cc ? ` for a ${cc} cc ${t.vehicle_type ?? "vehicle"}` : ""}</div>
            </dt>
            <dd className="num text-slate-300">−{inr(t.deductible_inr)}</dd>
          </div>
          <div className="mt-1 flex items-baseline justify-between gap-4 border-t border-white/10 pt-4">
            <dt>
              <div className="text-lg font-semibold text-slate-50">Net payable</div>
              <div className="mt-0.5 text-sm text-slate-500">What your policy covers for these repairs</div>
            </dt>
            <dd><Money v={t.insurer_payable_inr} className="text-2xl font-semibold text-slate-50" /></dd>
          </div>
        </dl>
      )}

      {has && (
        <div className="mt-5 border-t border-white/[0.06] pt-2">
          <Disclosure title="How was this calculated?">
            <ul className="space-y-3 leading-relaxed text-slate-400">
              <li><span className="text-slate-200">Repair method.</span> Each damaged part gets one repair method from fixed rules on the type and severity of damage. For example, a broken lamp is replaced and a small dent can be repaired without repainting.</li>
              <li><span className="text-slate-200">Depreciation.</span> {t.zero_depreciation_cover ? "Your zero-depreciation cover means nothing is deducted for wear."
                : "Only new parts are depreciated: 50% for plastic and rubber, nothing for glass, and an age-based rate for metal."}</li>
              <li><span className="text-slate-200">Excess.</span> The compulsory excess of {inr(t.deductible_inr)} is fixed by your policy.</li>
              {sources.length > 0 && <li><span className="text-slate-200">Part prices.</span> {sources.join(" ")} Other prices and labour use our standard rate card.</li>}
              <li><span className="text-slate-200">Estimate, net payable and cash offer.</span> The repair estimate is the garage bill. Net payable is what your policy covers after depreciation and excess. A cash settlement is paid to you instead of the insurer arranging the repair, so it is a separate amount.</li>
            </ul>
          </Disclosure>
        </div>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Offer card - current authorised offer only. No range, no ceiling.   */
/* ------------------------------------------------------------------ */

function OfferCard({ offer, busy, onSend, onShowReceipt, onReviewEstimate }: {
  offer: OfferBlock; busy: boolean; onSend: (t: string) => void; onShowReceipt: () => void; onReviewEstimate: () => void;
}) {
  const amount = offer.state === "SETTLED" ? offer.settled_amount_inr : offer.current_offer_inr;

  if (offer.state === "SETTLED") {
    return (
      <section className="panel p-5 sm:p-7">
        <div className="flex items-start justify-between gap-3">
          <div className="text-sm text-slate-400">Offer accepted</div>
          <Pill tone="teal" dot>Decision recorded</Pill>
        </div>
        <Money v={amount} className="mt-1 block text-4xl font-semibold text-slate-50" />
        <p className="mt-3 text-slate-400">Your acceptance is recorded against this claim. This is a demo, so no money is transferred.</p>
        <button className="btn-secondary mt-6 w-full" onClick={onShowReceipt}>View confirmation</button>
      </section>
    );
  }

  return (
    <section className="panel p-5 sm:p-7">
      <div className="text-sm text-slate-400">Your cash settlement offer</div>
      <Money v={amount} className="mt-1 block text-4xl font-semibold text-slate-50" />
      <p className="mt-3 text-slate-400">
        Accepting closes the claim for this amount, paid to you instead of a garage repair arranged by us.{" "}
        <button onClick={onReviewEstimate} className="text-teal-300 underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-300/60">Review the estimate</button>
      </p>

      {offer.state === "CONFIRMING" ? (
        <div className="mt-6 rounded-xl bg-white/[0.04] p-4">
          <p className="text-slate-200">Accept <span className="num font-semibold">{inr(amount)}</span> as full and final settlement of this claim?</p>
          <div className="mt-4 flex flex-col gap-3 sm:flex-row">
            <button className="btn-primary flex-1" disabled={busy} onClick={() => onSend("CONFIRM")}>Yes, accept</button>
            <button className="btn-secondary" disabled={busy} onClick={() => onSend("No, not yet")}>Go back</button>
          </div>
        </div>
      ) : (
        <>
          <button className="btn-primary mt-6 w-full" disabled={busy || !offer.can_accept} onClick={() => onSend("I accept this offer")}>Accept offer</button>
          <div className="mt-3 grid gap-2 sm:grid-cols-3">
            <button className="btn-secondary h-10 px-3 text-sm" disabled={busy} onClick={() => onSend("Please explain how my estimate was calculated")}>
              <MessageCircleQuestion className="h-4 w-4" /> Explain
            </button>
            <button className="btn-secondary h-10 px-3 text-sm" disabled={busy || !offer.can_reconsider} onClick={() => onSend("Please reconsider the offer")}>
              <RefreshCcw className="h-4 w-4" /> Reconsider
            </button>
            <button className="btn-secondary h-10 px-3 text-sm" disabled={busy} onClick={() => onSend("I want to talk to a human")}>
              <UserRound className="h-4 w-4" /> Talk to a person
            </button>
          </div>
          {!offer.can_reconsider && <p className="mt-3 text-sm text-slate-500">This is the best offer available for an instant settlement. You can accept it or ask a person to review your claim.</p>}
        </>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Specialist review - automated settlement is stopped                 */
/* ------------------------------------------------------------------ */

function ReviewPanel({ claim, offer, onActivity }: { claim: ClaimState; offer: OfferBlock; onActivity: () => void }) {
  const q = claim.quote;
  if (q.status === "NO_PAYOUT" && offer.state !== "ESCALATED") {
    return (
      <section className="panel p-5 sm:p-7">
        <h2 className="text-lg font-semibold text-slate-50">No payable amount</h2>
        <p className="mt-2 text-slate-400">The repair estimate is within your policy's compulsory excess, so there is nothing to pay on this claim. You can still ask a question or speak to a person.</p>
      </section>
    );
  }
  // The engine writes one reason per detection; repeated identical reasons are shown once with a count.
  const counted = new Map<string, number>();
  (q.status === "APPROVED" || q.status === "NO_PAYOUT" ? [] : q.reasons).forEach((r) => {
    const t = prettyReason(r); counted.set(t, (counted.get(t) ?? 0) + 1);
  });
  const reasons = [...counted.entries()].map(([t, n]) => (n > 1 ? `${t} (${n} detections)` : t));
  const customerAsked = offer.state === "ESCALATED";
  const uncertain = claim.damages.filter((d) => findingStatus(d).tone === "amber");

  return (
    <section className="panel p-5 sm:p-7">
      <Pill tone="amber" dot>Specialist review</Pill>
      <h2 className="mt-3 text-lg font-semibold text-slate-50">A claims specialist will review your claim</h2>
      <p className="mt-1.5 text-slate-400">An instant cash settlement isn't available while the review is open. Nothing has been decided against you.</p>
      <dl className="mt-6 space-y-5">
        <div>
          <dt className="font-medium text-slate-200">Why</dt>
          <dd className="mt-1.5 space-y-1 text-slate-400">
            {reasons.map((r) => <p key={r}>{r}</p>)}
            {customerAsked && reasons.length === 0 && <p>You asked for a person to review the claim.</p>}
          </dd>
        </div>
        {uncertain.length > 0 && (
          <div>
            <dt className="font-medium text-slate-200">Still to be confirmed</dt>
            <dd className="mt-1.5 text-slate-400">{[...new Set(uncertain.map((d) => `${title(d.part)} (${d.damage ?? "possible damage"})`))].join(", ")}</dd>
          </div>
        )}
        <div>
          <dt className="font-medium text-slate-200">Not available during review</dt>
          <dd className="mt-1.5 text-slate-400">Accepting or reconsidering a cash offer.</dd>
        </div>
        <div>
          <dt className="font-medium text-slate-200">What happens next</dt>
          <dd className="mt-1.5 text-slate-400">
            A specialist checks the claim and contacts you.{claim.evidence_retained ? " Your photos or video and this estimate are stored with the claim." : ""}
          </dd>
        </div>
      </dl>
      <button className="btn-secondary mt-6" onClick={onActivity}>View claim activity</button>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Conversation                                                       */
/* ------------------------------------------------------------------ */

function Conversation({ messages, offer, busy, onSend, onRetry }: {
  messages: ChatMessage[]; offer: OfferBlock; busy: boolean; onSend: (t: string) => void; onRetry: (i: number) => void;
}) {
  const [text, setText] = useState("");
  const list = useRef<HTMLDivElement>(null);
  useEffect(() => { list.current?.scrollTo({ top: list.current.scrollHeight, behavior: "smooth" }); }, [messages, busy]);
  const closed = offer.state === "SETTLED" || offer.state === "ESCALATED";
  function send() { if (!text.trim() || busy || closed) return; onSend(text.trim()); setText(""); }

  return (
    <section className="panel flex flex-col">
      <h2 className="px-5 pt-5 text-base font-semibold text-slate-50 sm:px-7">Messages</h2>
      <div ref={list} className="mt-3 max-h-[420px] min-h-[160px] space-y-4 overflow-y-auto px-5 py-2 sm:px-7" aria-live="polite">
        {messages.length === 0 && (
          <p className="py-8 text-center text-slate-500">No messages yet. Ask anything about your claim, your estimate or your offer.</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={cx("fade-in flex", m.from === "user" ? "justify-end" : "justify-start")}>
            <div className="max-w-[88%]">
              <div className={cx("rounded-2xl px-4 py-3 leading-relaxed",
                m.from === "user" ? "rounded-br-md bg-teal-400/15 text-teal-50" : "rounded-bl-md bg-white/[0.05] text-slate-100")}>
                {m.text}
              </div>
              {m.failed ? (
                <button onClick={() => onRetry(i)} className="mt-1 flex items-center gap-1 text-sm text-rose-300 hover:text-rose-200">
                  <RotateCw className="h-3.5 w-3.5" /> Not sent — tap to retry
                </button>
              ) : m.time ? (
                <div className={cx("mt-1 text-xs text-slate-500", m.from === "user" && "text-right")}>{m.time}</div>
              ) : null}
            </div>
          </div>
        ))}
        {busy && <div className="flex gap-1.5 px-1 py-2" aria-label="Waiting for reply">{[0, 1, 2].map((i) => <span key={i} className="h-2 w-2 animate-bounce rounded-full bg-slate-500" style={{ animationDelay: `${i * 140}ms` }} />)}</div>}
      </div>
      {closed ? (
        <p className="border-t border-white/[0.06] px-5 py-4 text-sm text-slate-400 sm:px-7">
          {offer.state === "SETTLED" ? "This claim is settled, so messages are closed." : "A specialist is handling your claim, so automated messages are closed. They will contact you directly."}
        </p>
      ) : (
        <form onSubmit={(e) => { e.preventDefault(); send(); }} className="flex gap-2 border-t border-white/[0.06] p-4 sm:px-7">
          <input className="input" value={text} onChange={(e) => setText(e.target.value)} placeholder="Ask about your claim" maxLength={500} aria-label="Message" enterKeyHint="send" />
          <button className="btn-primary w-11 shrink-0 px-0" disabled={busy || !text.trim()} aria-label="Send"><SendHorizontal className="h-4 w-4" /></button>
        </form>
      )}
    </section>
  );
}

export default function Settlement({ claim, offer, messages, busy, onSend, onRetry, onShowReceipt, onActivity }: {
  claim: ClaimState; offer: OfferBlock; messages: ChatMessage[]; busy: boolean;
  onSend: (t: string) => void; onRetry: (i: number) => void; onShowReceipt: () => void; onActivity: () => void;
}) {
  const showOffer = offer.state === "OPEN" || offer.state === "CONFIRMING" || offer.state === "SETTLED";
  const review = () => document.getElementById("estimate")?.scrollIntoView({ behavior: "smooth", block: "start" });
  return (
    // Phone: decision first, then the estimate, then messages. Desktop: estimate left, decision + messages right.
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] lg:grid-rows-[auto_1fr] lg:items-start">
      <div className="lg:col-start-2 lg:row-start-1">
        {showOffer ? <OfferCard offer={offer} busy={busy} onSend={onSend} onShowReceipt={onShowReceipt} onReviewEstimate={review} />
          : <ReviewPanel claim={claim} offer={offer} onActivity={onActivity} />}
      </div>
      <div className="lg:col-start-1 lg:row-span-2 lg:row-start-1"><Statement claim={claim} id="estimate" /></div>
      <div className="lg:col-start-2 lg:row-start-2"><Conversation messages={messages} offer={offer} busy={busy} onSend={onSend} onRetry={onRetry} /></div>
    </div>
  );
}
