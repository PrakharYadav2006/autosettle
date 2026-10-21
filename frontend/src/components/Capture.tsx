import { useEffect, useRef, useState } from "react";
import { Upload, X } from "lucide-react";

const GUIDE = [
  ["Number plate first", "So we can match the vehicle to your policy."],
  ["Slow walk-around", "10–20 seconds, whole vehicle in frame. Blurry frames are dropped automatically."],
  ["Close-up of each damaged area", "Hold for about two seconds on each one."],
];

export function GuidanceModal({ onClose, onFile }: { onClose: () => void; onFile: (f: File) => void }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 backdrop-blur-sm sm:items-center sm:p-6" onClick={onClose}>
      <div role="dialog" aria-modal="true" aria-labelledby="guide-title"
        className="panel fade-in w-full max-w-md rounded-b-none p-6 sm:rounded-2xl sm:p-8" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-4">
          <h2 id="guide-title" className="text-xl font-semibold text-slate-50">Capture the damage</h2>
          <button onClick={onClose} className="btn-quiet -mr-2 -mt-1 h-9 w-9 px-0" aria-label="Close"><X className="h-5 w-5" /></button>
        </div>
        <ol className="mt-6 space-y-5">
          {GUIDE.map(([h, t], i) => (
            <li key={h} className="flex gap-4">
              <span className="num flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-teal-400/10 text-sm text-teal-200">{i + 1}</span>
              <div><div className="font-medium text-slate-100">{h}</div><p className="mt-0.5 text-sm text-slate-400">{t}</p></div>
            </li>
          ))}
        </ol>
        <input ref={input} type="file" accept="image/*,video/*" className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
        <button className="btn-primary mt-8 w-full" onClick={() => input.current?.click()}>
          <Upload className="h-4 w-4" /> Upload video or photo
        </button>
        <p className="mt-3 text-center text-xs text-slate-500">MP4, MOV, JPG or PNG · up to 200 MB</p>
      </div>
    </div>
  );
}

/** Honest processing state: the backend replies once, so no fake per-stage checkmarks. */
export function Processing({ file, error, onBack }: { file: File; error: string | null; onBack: () => void }) {
  const video = file.type.startsWith("video") || /\.(mp4|mov|avi|mkv|webm|3gp|m4v)$/i.test(file.name);
  const [secs, setSecs] = useState(0);
  useEffect(() => { if (error) return; const id = setInterval(() => setSecs((s) => s + 1), 1000); return () => clearInterval(id); }, [error]);
  const preview = useRef<string | null>(null);
  if (!preview.current && !video) preview.current = URL.createObjectURL(file);
  const stages = [
    video ? "Selecting sharp keyframes" : "Reading the photo",
    "Identifying the vehicle and its parts",
    "Locating damage",
    "Judging how serious each damage is",
    "Pricing against your policy",
  ];

  return (
    <div className="mx-auto flex min-h-[calc(100vh-64px)] max-w-lg flex-col justify-center px-5 py-10">
      {preview.current && <img src={preview.current} alt="" className="mb-8 max-h-56 w-full rounded-2xl object-cover opacity-60" />}
      {error ? (
        <div role="alert">
          <h1 className="text-xl font-semibold text-slate-50">We couldn't assess this file</h1>
          <p className="mt-2 text-slate-400">{error}</p>
          <button className="btn-secondary mt-6" onClick={onBack}>Back to dashboard</button>
        </div>
      ) : (
        <>
          <h1 className="text-xl font-semibold text-slate-50">Assessing your claim</h1>
          <p className="mt-2 text-slate-400">{file.name} · {secs}s elapsed · usually under a minute{video ? " for a video" : ""}</p>
          <div className="mt-6 h-1 overflow-hidden rounded-full bg-white/[0.06]" role="progressbar" aria-label="Assessment in progress">
            <div className="indeterminate h-full w-2/5 rounded-full bg-teal-400/70" />
          </div>
          <p className="mt-8 text-sm font-medium text-slate-300">What happens during assessment</p>
          <ul className="mt-3 space-y-2.5 text-sm text-slate-400">
            {stages.map((s) => <li key={s} className="flex gap-3"><span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-slate-500" />{s}</li>)}
          </ul>
        </>
      )}
    </div>
  );
}
