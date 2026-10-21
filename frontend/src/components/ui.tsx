import type { ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { cx } from "../format";

export type Tone = "neutral" | "teal" | "amber" | "red";

const PILL: Record<Tone, string> = {
  neutral: "bg-white/[0.06] text-slate-300",
  teal: "bg-teal-400/10 text-teal-200",
  amber: "bg-amber-400/10 text-amber-200",
  red: "bg-rose-400/10 text-rose-200",
};
const DOT: Record<Tone, string> = { neutral: "bg-slate-400", teal: "bg-teal-300", amber: "bg-amber-300", red: "bg-rose-300" };

export function Pill({ tone = "neutral", children, dot }: { tone?: Tone; children: ReactNode; dot?: boolean }) {
  return (
    <span className={cx("inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-medium", PILL[tone])}>
      {dot && <span className={cx("h-1.5 w-1.5 rounded-full", DOT[tone])} />}
      {children}
    </span>
  );
}

export function Disclosure({ title, children, defaultOpen }: { title: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="group" open={defaultOpen}>
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 rounded-lg py-2 text-sm font-medium text-slate-300 hover:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-300/60 [&::-webkit-details-marker]:hidden">
        {title}
        <ChevronDown className="h-4 w-4 text-slate-500 transition group-open:rotate-180" />
      </summary>
      <div className="pb-2 pt-2">{children}</div>
    </details>
  );
}

export function Segmented<T extends string>({ value, options, onChange, size = "md" }: {
  value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void; size?: "sm" | "md";
}) {
  return (
    <div role="tablist" className="inline-flex rounded-xl bg-white/[0.04] p-1 ring-1 ring-inset ring-white/[0.06]">
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={value === o.value} onClick={() => onChange(o.value)}
          className={cx("rounded-lg font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-300/60",
            size === "sm" ? "px-3 py-1.5 text-xs" : "px-4 py-2 text-sm",
            value === o.value ? "bg-ink-700 text-slate-50 shadow-sm" : "text-slate-400 hover:text-slate-200")}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-2 text-sm text-slate-300">
      <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}
        className={cx("relative h-5 w-9 rounded-full transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-300/60", checked ? "bg-teal-400" : "bg-white/15")}>
        <span className={cx("absolute top-0.5 h-4 w-4 rounded-full bg-white transition", checked ? "left-[18px]" : "left-0.5")} />
      </button>
      {label}
    </label>
  );
}

export function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <img src="/icon.svg" alt="" className="h-8 w-8" />
      <span className="text-[1.05rem] font-semibold tracking-tight text-slate-50">Auto-Settle AI</span>
    </div>
  );
}

export function Money({ v, className }: { v: number | null | undefined; className?: string }) {
  return <span className={cx("num", className)}>{v === null || v === undefined ? "—" : "₹" + Math.round(v).toLocaleString("en-IN")}</span>;
}
