import { useCallback, useEffect, useState } from "react";
import { ClipboardList, LayoutGrid, LogOut, ScrollText } from "lucide-react";
import { adminApi, adminToken } from "./api";
import { usePath } from "./kit";
import { AuditPage, OverviewPage, QueuePage } from "./pages";
import ClaimWorkspace, { TABS, type Tab } from "./ClaimWorkspace";
import { cx } from "../format";

function AdminLogin({ onDone }: { onDone: () => void }) {
  const [u, setU] = useState("admin");
  const [p, setP] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  async function submit(e: React.FormEvent) {
    e.preventDefault(); setBusy(true); setErr(null);
    try { const r = await adminApi.login(u, p); adminToken.set(r.token); onDone(); }
    catch (x) { setErr((x as Error).message); }
    finally { setBusy(false); }
  }
  return (
    <div className="flex min-h-screen items-center justify-center px-5">
      <form onSubmit={submit} className="panel w-full max-w-sm p-8">
        <div className="flex items-center gap-2.5"><img src="/icon.svg" alt="" className="h-8 w-8" /><span className="font-semibold text-slate-50">Auto-Settle AI</span><span className="text-sm text-slate-500">Staff</span></div>
        <h1 className="mt-8 text-xl font-semibold text-slate-50">Employee sign-in</h1>
        <p className="mt-1.5 text-sm text-slate-400">For authorised staff. Customer sessions cannot access this console.</p>
        <label className="mt-6 block text-sm text-slate-300">Username<input className="input mt-1.5" value={u} onChange={(e) => setU(e.target.value)} autoComplete="username" /></label>
        <label className="mt-4 block text-sm text-slate-300">Password<input className="input mt-1.5" type="password" value={p} onChange={(e) => setP(e.target.value)} autoComplete="current-password" /></label>
        {err && <p role="alert" className="mt-4 rounded-xl bg-rose-400/10 px-4 py-3 text-sm text-rose-200">{err}</p>}
        <button className="btn-primary mt-6 w-full" disabled={busy || !p}>{busy ? "Signing in…" : "Sign in"}</button>
        <p className="mt-5 text-xs leading-relaxed text-slate-500">Verified by the server. Set AUTOSETTLE_ADMIN_PASSWORD, or use the password printed in the server terminal at start-up.</p>
      </form>
    </div>
  );
}

const NAV = [
  { k: "overview", label: "Overview", href: "/admin", icon: LayoutGrid },
  { k: "claims", label: "Claim queue", href: "/admin/claims", icon: ClipboardList },
  { k: "audit", label: "Audit log", href: "/admin/audit", icon: ScrollText },
] as const;

const PAGE: Record<string, [string, string]> = {
  overview: ["Overview", "Operational summary from claims stored on this server."],
  claims: ["Claim queue", "Search and filter every claim. Open one to investigate."],
  audit: ["Audit log", "What the system and staff did, in order."],
};

export default function AdminApp() {
  const [path, search, nav, replace] = usePath();
  const [authed, setAuthed] = useState(!!adminToken.get());
  const out = useCallback(() => { adminToken.clear(); setAuthed(false); }, []);
  useEffect(() => { document.title = "Auto-Settle AI · Staff"; }, []);
  if (!authed) return <AdminLogin onDone={() => setAuthed(true)} />;

  const m = path.match(/^\/admin\/claims\/([a-z0-9]+)(?:\/([a-z]+))?/);
  const claimId = m?.[1];
  const tab: Tab = (TABS.map(([k]) => k) as string[]).includes(m?.[2] ?? "") ? (m![2] as Tab) : "summary";
  const section = claimId || path.startsWith("/admin/claims") ? "claims" : path.startsWith("/admin/audit") ? "audit" : "overview";
  const signOut = () => { adminApi.logout(); out(); };

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[248px_minmax(0,1fr)]">
      {/* desktop sidebar */}
      <aside className="hidden border-r border-white/[0.06] bg-ink-900/50 lg:sticky lg:top-0 lg:flex lg:h-screen lg:flex-col">
        <div className="flex items-center gap-2.5 px-6 py-6">
          <img src="/icon.svg" alt="" className="h-8 w-8" />
          <div><div className="font-semibold leading-tight text-slate-50">Auto-Settle AI</div><div className="text-xs text-slate-500">Staff console</div></div>
        </div>
        <nav className="mt-4 flex flex-col gap-1 px-3" aria-label="Main">
          {NAV.map(({ k, label, href, icon: I }) => (
            <button key={k} onClick={() => nav(href)} aria-current={section === k ? "page" : undefined}
              className={cx("relative flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-sm font-medium transition",
                section === k ? "bg-white/[0.06] text-slate-50" : "text-slate-400 hover:bg-white/[0.03] hover:text-slate-200")}>
              {section === k && <span className="absolute -left-3 top-2 bottom-2 w-1 rounded-r-full bg-teal-300" />}
              <I className={cx("h-[18px] w-[18px]", section === k ? "text-teal-300" : "")} />{label}
            </button>
          ))}
        </nav>
        <div className="mt-auto border-t border-white/[0.06] px-6 py-5">
          <div className="text-sm text-slate-300">admin</div>
          <button onClick={signOut} className="btn-quiet -ml-3 mt-1"><LogOut className="h-4 w-4" /> Sign out</button>
        </div>
      </aside>

      {/* mobile / tablet top bar */}
      <header className="border-b border-white/[0.06] bg-ink-900/60 lg:hidden">
        <div className="flex items-center justify-between px-5 py-3">
          <div className="flex items-center gap-2"><img src="/icon.svg" alt="" className="h-7 w-7" /><span className="font-semibold text-slate-50">Staff console</span></div>
          <button onClick={signOut} className="btn-quiet" aria-label="Sign out"><LogOut className="h-4 w-4" /></button>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-2" aria-label="Main">
          {NAV.map(({ k, label, href, icon: I }) => (
            <button key={k} onClick={() => nav(href)} className={cx("flex shrink-0 items-center gap-2 rounded-lg px-3 py-2 text-sm", section === k ? "bg-white/[0.07] text-slate-50" : "text-slate-400")}>
              <I className="h-4 w-4" />{label}
            </button>
          ))}
        </nav>
      </header>

      <main className="min-w-0 px-5 py-8 lg:px-10 lg:py-10">
        {!claimId && (
          <div className="mb-8">
            <h1 className="text-2xl font-semibold text-slate-50">{PAGE[section][0]}</h1>
            <p className="mt-1.5 text-slate-400">{PAGE[section][1]}</p>
          </div>
        )}
        {claimId ? <ClaimWorkspace id={claimId} tab={tab} onAuth={out} nav={nav} />
          : section === "claims" ? <QueuePage onAuth={out} nav={nav} search={search} replace={replace} />
          : section === "audit" ? <AuditPage onAuth={out} nav={nav} />
          : <OverviewPage onAuth={out} nav={nav} />}
      </main>
    </div>
  );
}
