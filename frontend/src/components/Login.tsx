import { useState } from "react";
import { ArrowRight, Settings2 } from "lucide-react";
import { api, getApiUrl, setApiUrl } from "../api";
import type { DemoSession } from "../types";
import { Disclosure, Logo } from "./ui";

export default function Login({ onSession }: { onSession: (s: DemoSession) => void }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [url, setUrl] = useState(getApiUrl());

  async function launch() {
    setBusy(true); setErr(null);
    try { onSession(await api.demoSession()); }
    catch (e) { setErr((e as Error).message); }
    finally { setBusy(false); }
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-5 py-12">
      <div className="w-full max-w-[420px]">
        <div className="mb-10 flex flex-col items-center text-center">
          <Logo />
          <p className="mt-4 text-slate-400">Motor claims, assessed from a photo or video — with every rupee explained.</p>
        </div>

        <div className="panel p-6 sm:p-8">
          <h1 className="text-xl font-semibold text-slate-50">Start a demo</h1>
          <p className="mt-1.5 text-slate-400">Opens a synthetic demo policy (Demo Customer A, 2021 Maruti Swift). No real customer data is used.</p>
          <button onClick={launch} disabled={busy} className="btn-primary mt-6 w-full">
            {busy ? "Starting…" : "Launch demo session"} {!busy && <ArrowRight className="h-4 w-4" />}
          </button>
          {err && <p role="alert" className="mt-4 rounded-xl bg-rose-400/10 px-4 py-3 text-sm text-rose-200">{err}</p>}

          <div className="mt-6 border-t border-white/[0.06] pt-4">
            <Disclosure title="Sign in with policy number">
              <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); setErr("Customer sign-in is not enabled in this prototype. Use the demo session."); }}>
                <input className="input" placeholder="Policy number or mobile" autoComplete="username" aria-label="Policy number or mobile" />
                <input className="input" type="password" placeholder="Password" autoComplete="current-password" aria-label="Password" />
                <button className="btn-secondary w-full" type="submit">Sign in</button>
              </form>
            </Disclosure>
          </div>
        </div>

        <div className="mt-6">
          <Disclosure title={<span className="flex items-center gap-2 text-slate-400"><Settings2 className="h-4 w-4" /> Server connection</span>}>
            <div className="flex gap-2">
              <input className="input num text-sm" value={url} onChange={(e) => setUrl(e.target.value)} aria-label="Backend URL" />
              <button className="btn-secondary" onClick={() => { setApiUrl(url); setUrl(getApiUrl()); }}>Save</button>
            </div>
            <p className="mt-2 text-xs text-slate-500">FastAPI backend, e.g. your cloudflared tunnel URL.</p>
          </Disclosure>
        </div>
      </div>
    </div>
  );
}
