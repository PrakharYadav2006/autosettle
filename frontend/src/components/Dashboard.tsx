import { ArrowRight, Camera } from "lucide-react";
import type { ClaimPhase, ClaimState, Policy } from "../types";
import { inr, shortDate, title } from "../format";
import { claimStatus } from "../status";
import { Pill } from "./ui";

export function PolicySummary({ p, onSwitch, all }: { p: Policy; all?: Policy[]; onSwitch?: (p: Policy) => void }) {
  const rows: [string, string][] = [
    ["Registration", p.plate],
    ["Vehicle", `${title(p.make)} ${p.model} · ${p.year} · ${p.engine_cc} cc`],
    ["Policy period", `${shortDate(p.policy_start)} – ${shortDate(p.policy_end)}`],
    ["Compulsory excess", inr(p.deductible_inr)],
    ["Zero depreciation", p.zero_dep ? "Included" : "Not included"],
  ];
  return (
    <section className="panel p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="eyebrow">Your policy</div>
          <div className="mt-1.5 text-lg font-semibold text-slate-50">{title(p.make)} {p.model}</div>
          <div className="mt-0.5 text-sm text-slate-400">Policy {p.policy_no}</div>
        </div>
        <Pill tone={p.policy_active ? "teal" : "red"} dot>{p.policy_active ? "Active" : "Lapsed"}</Pill>
      </div>
      <dl className="mt-5 divide-y divide-white/[0.06]">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-baseline justify-between gap-6 py-3">
            <dt className="text-sm text-slate-400">{k}</dt>
            <dd className="num text-right text-[0.95rem] text-slate-100">{v}</dd>
          </div>
        ))}
      </dl>
      {all && onSwitch && all.length > 1 && (
        <label className="mt-4 flex items-center justify-between gap-3 border-t border-white/[0.06] pt-4 text-sm text-slate-400">
          Switch vehicle
          <select className="input h-9 w-auto max-w-[60%] py-0 text-sm" value={p.plate}
            onChange={(e) => { const n = all.find((x) => x.plate === e.target.value); if (n) onSwitch(n); }}>
            {all.map((x) => <option key={x.plate} value={x.plate}>{title(x.make)} {x.model} ({x.plate})</option>)}
          </select>
        </label>
      )}
    </section>
  );
}

export default function Dashboard({ policy, all, claim, phase, onSwitch, onFile, onOpenClaim }: {
  policy: Policy; all: Policy[]; claim: ClaimState | null; phase: ClaimPhase;
  onSwitch: (p: Policy) => void; onFile: () => void; onOpenClaim: () => void;
}) {
  const st = claimStatus(phase, claim?.quote.status);
  const offer = claim?.offer.current_offer_inr;
  return (
    <div className="mx-auto max-w-5xl px-5 py-8 sm:py-12">
      <h1 className="text-2xl font-semibold text-slate-50 sm:text-3xl">Hello, {policy.holder}</h1>
      <p className="mt-2 text-slate-400">Had an accident? File a claim with a short video or a few photos.</p>

      <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_1.1fr] lg:items-start">
        <section className="panel flex flex-col p-6">
          {claim ? (
            <>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="eyebrow">Current claim</div>
                  <div className="num mt-1.5 text-lg font-semibold text-slate-50">AS-{claim.claim_id.slice(0, 6).toUpperCase()}</div>
                </div>
                <Pill tone={st.tone} dot>{st.label}</Pill>
              </div>
              <div className="mt-6">
                {offer || claim.quote.status === "NO_PAYOUT" ? (
                  <>
                    <div className="text-sm text-slate-400">{offer ? "Current cash offer" : "Net payable under policy"}</div>
                    <div className="num mt-1 text-3xl font-semibold text-slate-50">{inr(offer ?? claim.offer.insurer_payable_inr)}</div>
                  </>
                ) : (
                  <>
                    <div className="text-sm text-slate-400">Status</div>
                    <div className="mt-1 text-lg font-medium text-slate-100">With a claims specialist — no offer yet</div>
                  </>
                )}
              </div>
              <div className="mt-8 flex flex-col gap-3 sm:flex-row">
                <button className="btn-primary flex-1" onClick={onOpenClaim}>Open claim <ArrowRight className="h-4 w-4" /></button>
                <button className="btn-secondary" onClick={onFile}>New claim</button>
              </div>
            </>
          ) : (
            <>
              <div className="eyebrow">New claim</div>
              <div className="mt-1.5 text-lg font-semibold text-slate-50">Get an assessment in about a minute</div>
              <ol className="mt-5 space-y-3 text-slate-300">
                <li className="flex gap-3"><span className="num text-teal-300">1</span> Record the number plate and a slow walk-around.</li>
                <li className="flex gap-3"><span className="num text-teal-300">2</span> We locate damage and price it against your policy.</li>
                <li className="flex gap-3"><span className="num text-teal-300">3</span> Accept a cash offer, or ask for a surveyor.</li>
              </ol>
              <div className="mt-8">
                <button className="btn-primary w-full" onClick={onFile}><Camera className="h-4 w-4" /> File a claim</button>
              </div>
            </>
          )}
        </section>
        <PolicySummary p={policy} all={all} onSwitch={onSwitch} />
      </div>
    </div>
  );
}
