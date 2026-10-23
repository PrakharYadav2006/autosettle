import { useEffect, useRef, useState } from "react";
import { RotateCcw } from "lucide-react";
import { api, ApiError, store } from "./api";
import type { ChatMessage, ClaimPhase, ClaimState, Session } from "./types";
import { nowTime } from "./status";
import Login from "./components/Login";
import Dashboard from "./components/Dashboard";
import { GuidanceModal, Processing } from "./components/Capture";
import ClaimView, { type Tab } from "./components/ClaimView";
import SettlementModal from "./components/SettlementModal";
import { Logo } from "./components/ui";

type Screen = "home" | "processing" | "claim";

function fmtTime(t: string | null): string | null {
  if (!t) return null;
  const d = new Date(t);
  return Number.isNaN(d.getTime()) ? t : d.toLocaleTimeString("en-IN", { hour: "numeric", minute: "2-digit" });
}

export default function CustomerApp() {
  // Older stored sessions (pre Demo Mode) have no "mode" - they are dropped, not reused.
  const [session, setSession] = useState<Session | null>(() => { const s = store.get<Session>("session"); return s?.mode ? s : null; });
  const [screen, setScreen] = useState<Screen>("home");
  const [tab, setTab] = useState<Tab>(() => store.get<Tab>("tab") ?? "assessment");
  const [guide, setGuide] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [procErr, setProcErr] = useState<string | null>(null);
  const [claim, setClaim] = useState<ClaimState | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [receipt, setReceipt] = useState(false);
  const [restoring, setRestoring] = useState(() => !!store.get<string>("claimId"));
  const inFlight = useRef(false);

  useEffect(() => { store.set("tab", tab); }, [tab]);

  function loadClaim(c: ClaimState) {
    // An older api_server.py answers without these fields - say so instead of crashing.
    if (!c?.chat?.messages || !Array.isArray(c.events) || !c.offer || !("state" in c.offer)) {
      throw new ApiError("The server is running an older api_server.py that this website can't use. Replace api_server.py and negotiation.py with the latest versions and restart the server.", 0);
    }
    setClaim(c);
    setMessages(c.chat.messages.map((m) => ({ ...m, time: fmtTime(m.time) })));
    store.set("claimId", c.claim_id);
  }

  // Page refresh: reload the claim from the server - never from cached amounts.
  useEffect(() => {
    const id = store.get<string>("claimId");
    if (!id || !session) { setRestoring(false); return; }
    api.claim(id)
      .then((c) => { loadClaim(c); setScreen("claim"); })
      .catch((e) => { if (e instanceof ApiError && (e.status === 401 || e.status === 404)) store.clear("claimId"); })
      .finally(() => setRestoring(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function start(s: Session) {
    store.clear(); store.set("token", s.token); store.set("session", s);
    setSession(s);
  }
  function clearClaim() {
    setClaim(null); setMessages([]); setReceipt(false); setTab("assessment"); store.clear("claimId");
  }
  function resetDemo() { clearClaim(); store.clear(); setSession(null); setScreen("home"); }

  async function submit(f: File) {
    if (!session || inFlight.current) return;
    inFlight.current = true;
    setGuide(false); clearClaim(); setFile(f); setProcErr(null); setScreen("processing");
    try {
      loadClaim(await api.analyzeClaim(f));
      setScreen("claim");
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) { resetDemo(); return; }
      setProcErr((e as Error).message);
    } finally {
      inFlight.current = false;
    }
  }

  // Proof for reconsideration: re-assessed on the server; the reply arrives in the conversation.
  const [proofBusy, setProofBusy] = useState(false);
  const [proofKind, setProofKind] = useState<"photo" | "document" | null>(null);
  const [proofErr, setProofErr] = useState<string | null>(null);
  async function uploadProof(f: File, kind: "photo" | "document") {
    if (!claim || inFlight.current) return;
    inFlight.current = true;
    setProofBusy(true); setProofKind(kind); setProofErr(null);
    try {
      loadClaim(await api.submitEvidence(claim.claim_id, f, kind));
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) { resetDemo(); return; }
      setProofErr((e as Error).message);
    } finally {
      setProofBusy(false);
      inFlight.current = false;
    }
  }

  async function send(text: string, retryIndex?: number) {
    if (!claim || inFlight.current) return;      // one request at a time - no duplicate decisions
    inFlight.current = true;
    setMessages((m) => {
      const next = [...m];
      if (retryIndex !== undefined) next.splice(retryIndex, 1);
      return [...next, { from: "user", text, time: nowTime() }];
    });
    setBusy(true);
    try {
      const r = await api.chat(claim.claim_id, text);
      setMessages((m) => [...m, { from: "bot", text: r.reply, time: nowTime(), outcome: r.outcome }]);
      setClaim((c) => (c ? { ...c, offer: r.offer, events: r.events, chat: { ...c.chat, state: r.state } } : c));
      if (r.outcome === "SETTLED") setReceipt(true);
    } catch (e) {
      setMessages((m) => m.map((x, i) => (i === m.length - 1 ? { ...x, failed: true } : x)));
      if (e instanceof ApiError && e.status === 401) resetDemo();
    } finally {
      setBusy(false);
      inFlight.current = false;
    }
  }

  if (!session) return <Login onSession={start} />;
  const demo = session.mode === "demo";
  if (restoring) return <div className="flex min-h-screen items-center justify-center text-slate-400">Loading your claim…</div>;

  const offerState = claim?.offer.state;
  const phase: ClaimPhase =
    screen === "processing" ? "PROCESSING"
      : !claim ? "DRAFT"
      : offerState === "SETTLED" ? (claim.offer.settlement_scope === "partial" ? "PARTIALLY_SETTLED" : "SETTLED")
      : offerState === "OPEN" || offerState === "CONFIRMING" ? "AWAITING_DECISION"   // incl. a scoped demo offer
      : offerState === "ESCALATED" || claim.quote.status === "ESCALATED" || claim.quote.status === "REVIEW_REQUIRED" ? "ESCALATED"
      : "ASSESSED";

  return (
    <div className="min-h-screen">
      <header className="border-b border-white/[0.06] bg-ink-950/90 backdrop-blur sm:sticky sm:top-0 sm:z-40">
        <div className="mx-auto flex h-16 max-w-[1280px] items-center justify-between gap-4 px-5">
          <button onClick={() => setScreen("home")} aria-label="Home"><Logo /></button>
          <div className="flex items-center gap-2">
            {demo && <span className="rounded-full border border-amber-300/25 px-2.5 py-0.5 text-xs font-medium text-amber-200/90" title="Illustrative demo pricing - no policy, no real payment">Demo Mode</span>}
            <button onClick={resetDemo} className="btn-quiet text-slate-400" title={demo ? "Clear this demo session and start again" : "Sign out"}>
              <RotateCcw className="h-4 w-4" /><span className="hidden sm:inline">{demo ? "Reset" : "Sign out"}</span>
            </button>
          </div>
        </div>
      </header>

      {screen === "home" && (
        <Dashboard session={session} claim={claim} phase={phase}
          onFile={() => setGuide(true)} onOpenClaim={() => setScreen("claim")} />
      )}
      {screen === "processing" && file && <Processing file={file} error={procErr} onBack={() => setScreen("home")} />}
      {screen === "claim" && claim && (
        <ClaimView claim={claim} session={session} phase={phase} offer={claim.offer} messages={messages} busy={busy}
          tab={tab} setTab={setTab} onSend={(t) => send(t)} onRetry={(i) => send(messages[i].text, i)}
          proof={{ busy: proofBusy, busyKind: proofKind, error: proofErr, upload: uploadProof }}
          onBack={() => setScreen("home")} onShowReceipt={() => setReceipt(true)} />
      )}

      {guide && <GuidanceModal onClose={() => setGuide(false)} onFile={submit} />}
      {receipt && claim && claim.offer.state === "SETTLED" && <SettlementModal claim={claim} onClose={() => setReceipt(false)} />}
    </div>
  );
}
