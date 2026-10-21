import type { ClaimEvent, Policy } from "../types";
import { cx } from "../format";
import { PolicySummary } from "./Dashboard";

const DOT = { neutral: "bg-slate-500", teal: "bg-teal-300", amber: "bg-amber-300", red: "bg-rose-300" };

function when(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString("en-IN", { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
}

export default function Activity({ events, policy }: { events: ClaimEvent[]; policy: Policy }) {
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)] lg:items-start">
      <section className="panel p-5 sm:p-7">
        <h2 className="text-lg font-semibold text-slate-50">Claim activity</h2>
        <ol className="relative mt-6 space-y-6 pl-6">
          <span className="absolute bottom-1 left-[5px] top-1 w-px bg-white/10" aria-hidden />
          {(events ?? []).length === 0 && <li className="text-slate-500">No activity recorded yet.</li>}
          {(events ?? []).map((e, i) => (
            <li key={i} className="relative">
              <span className={cx("absolute -left-6 top-1.5 h-[11px] w-[11px] rounded-full ring-4 ring-ink-900", DOT[e.tone])} />
              <div className="flex flex-wrap items-baseline justify-between gap-x-4">
                <div className="font-medium text-slate-100">{e.title}</div>
                <div className="text-sm text-slate-500">{when(e.time)}</div>
              </div>
              {e.detail && <p className="mt-1 text-sm text-slate-400">{e.detail}</p>}
            </li>
          ))}
        </ol>
      </section>
      <PolicySummary p={policy} />
    </div>
  );
}
