import { ArrowLeft } from "lucide-react";
import type { ChatMessage, ClaimPhase, ClaimState, OfferBlock, Session } from "../types";
import { title } from "../format";
import { claimStatus } from "../status";
import { Pill, Segmented } from "./ui";
import Assessment from "./Assessment";
import Settlement, { type ProofProps } from "./Settlement";
import Activity from "./Activity";

export type Tab = "assessment" | "settlement" | "activity";

export default function ClaimView(p: {
  claim: ClaimState; session: Session; phase: ClaimPhase; offer: OfferBlock; messages: ChatMessage[]; busy: boolean;
  tab: Tab; setTab: (t: Tab) => void; onSend: (t: string) => void; onRetry: (i: number) => void;
  onBack: () => void; onShowReceipt: () => void; proof: ProofProps;
}) {
  const st = claimStatus(p.phase, p.claim.quote.status);
  const go = (t: Tab) => { p.setTab(t); window.scrollTo({ top: 0 }); };
  return (
    <div className="mx-auto max-w-[1280px] px-5 pb-16 pt-6 sm:pt-8">
      <div className="flex items-start gap-4">
        <button onClick={p.onBack} className="btn-quiet mt-0.5 h-10 w-10 shrink-0 px-0 ring-1 ring-inset ring-white/10" aria-label="Back to home">
          <ArrowLeft className="h-4 w-4" />
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <h1 className="text-xl font-semibold text-slate-50 sm:text-2xl">Claim AS-{p.claim.claim_id.slice(0, 6).toUpperCase()}</h1>
            <Pill tone={st.tone} dot>{st.label}</Pill>
          </div>
          <div className="mt-1 text-slate-400">{p.session.policy ? `${title(p.session.policy.make)} ${p.session.policy.model} · ${p.session.policy.plate}`
            : `Demo Mode · ${p.claim.demo_pricing?.tier_requested_label ?? "illustrative"} pricing · ${title(p.claim.vehicle.type)} detected`}</div>
        </div>
      </div>

      {p.claim.recorded_sample && (
        <div className="mt-4 rounded-xl border border-amber-300/20 bg-amber-400/[0.05] px-4 py-3 text-sm text-amber-100">
          Recorded sample: the damage findings were produced earlier by the live GPU pipeline{p.claim.recorded_sample.recorded_at ? ` (${new Date(p.claim.recorded_sample.recorded_at).toLocaleDateString()})` : ""}. Pricing, the offer, chat and any proof or estimate you upload are processed live now.
        </div>
      )}
      <nav className="mt-6 lg:sticky lg:top-16 lg:z-30 lg:-mx-5 lg:bg-ink-950/90 lg:px-5 lg:py-3 lg:backdrop-blur">
        <Segmented value={p.tab} onChange={go} options={[
          { value: "assessment", label: "Assessment" },
          { value: "settlement", label: "Settlement" },
          { value: "activity", label: "Activity" },
        ]} />
      </nav>

      <div className="fade-in mt-6" key={p.tab}>
        {p.tab === "assessment" && <Assessment claim={p.claim} offer={p.offer} onOpenSettlement={() => go("settlement")} />}
        {p.tab === "settlement" && (
          <Settlement claim={p.claim} offer={p.offer} messages={p.messages} busy={p.busy} onSend={p.onSend}
            onRetry={p.onRetry} onShowReceipt={p.onShowReceipt} onActivity={() => go("activity")} proof={p.proof} />
        )}
        {p.tab === "activity" && <Activity events={p.claim.events} session={p.session} claim={p.claim} />}
      </div>
    </div>
  );
}
