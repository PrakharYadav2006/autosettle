import { useEffect, useState } from "react";
import { ArrowRight, FlaskConical, Settings2 } from "lucide-react";
import { api, getApiUrl, setApiUrl } from "../api";
import { DEMO_TIERS, type DemoTier, type Session } from "../types";
import { Disclosure, Logo } from "./ui";

export default function Login({ onSession }: { onSession: (s: Session) => void }) {
  const [busy, setBusy] = useState<"policy" | "demo" | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [url, setUrl] = useState(getApiUrl());
  const [policyNo, setPolicyNo] = useState("");
  const [plate, setPlate] = useState("");
  const [demoOn, setDemoOn] = useState(false);       // server decides; hidden unless AUTOSETTLE_DEMO_MODE=1
  const [tier, setTier] = useState<DemoTier>("auto");

  useEffect(() => { api.health().then((h) => setDemoOn(!!h.demo_mode)).catch(() => setDemoOn(false)); }, [url]);

  async function signIn(e: React.FormEvent) {
    e.preventDefault();
    setBusy("policy"); setErr(null);
    try { onSession(await api.policySession(policyNo.trim(), plate.trim())); }
    catch (e2) { setErr((e2 as Error).message); }
    finally { setBusy(null); }
  }
  async function startDemo() {
    setBusy("demo"); setErr(null);
    try { onSession(await api.demoSession(tier)); }
    catch (e2) { setErr((e2 as Error).message); }
    finally { setBusy(null); }
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-5 py-12">
      <div className="w-full max-w-[420px]">
        <div className="mb-10 flex flex-col items-center text-center">
          <Logo />
          <p className="mt-4 text-slate-400">Motor claims, assessed from a photo or video — with every rupee explained.</p>
        </div>

        <div className="panel p-6 sm:p-8">
          <h1 className="text-xl font-semibold text-slate-50">Sign in</h1>
          <p className="mt-1.5 text-slate-400">Your vehicle, cover and excess come from your policy record.</p>
          <form className="mt-6 space-y-3" onSubmit={signIn}>
            <input className="input num" placeholder="Policy number" value={policyNo} onChange={(e) => setPolicyNo(e.target.value)} autoComplete="username" aria-label="Policy number" required />
            <input className="input num uppercase" placeholder="Registration number" value={plate} onChange={(e) => setPlate(e.target.value)} aria-label="Registration number" required />
            <button className="btn-primary w-full" type="submit" disabled={!!busy}>
              {busy === "policy" ? "Signing in…" : "Sign in"} {busy !== "policy" && <ArrowRight className="h-4 w-4" />}
            </button>
          </form>
          {err && <p role="alert" className="mt-4 rounded-xl bg-rose-400/10 px-4 py-3 text-sm text-rose-200">{err}</p>}

          {demoOn && (
            <div className="mt-6 border-t border-white/[0.06] pt-5">
              <div className="flex items-center gap-2 font-medium text-slate-100"><FlaskConical className="h-4 w-4 text-amber-300" /> Demo Mode</div>
              <p className="mt-1.5 text-sm text-slate-400">Upload any vehicle photo. No policy is attached and the vehicle's make is not identified, so pricing uses an illustrative rate card you choose. No real payment is made.</p>
              <label className="mt-4 block text-sm text-slate-400" htmlFor="tier">Price tier (demo assumption)</label>
              <select id="tier" className="input mt-1.5" value={tier} onChange={(e) => setTier(e.target.value as DemoTier)}>
                {DEMO_TIERS.map((t) => <option key={t.value} value={t.value}>{t.label}{t.value === "auto" ? " — generic mid-range" : ""}</option>)}
              </select>
              <button onClick={startDemo} disabled={!!busy} className="btn-secondary mt-4 w-full">
                {busy === "demo" ? "Starting…" : "Start demo"}
              </button>
            </div>
          )}
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
