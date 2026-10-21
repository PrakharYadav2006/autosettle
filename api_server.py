"""
Auto-Settle AI - FastAPI backend for the custom (Vercel) frontend
=================================================================

CUSTOMER (header X-Session-Token from /api/demo_session; claims are tied to that session)
    POST /api/demo_session     one-click demo login -> signed session token + demo policy
    POST /api/analyze_claim    photo or video -> findings, estimate, CURRENT offer only
    POST /api/chat_negotiate   {claim_id, message} -> reply + current offer
    GET  /api/claim/{id}       full customer view again (page refresh)
    GET  /api/health

ADMIN (header Authorization: Bearer <token> from /api/admin/login; checked server-side)
    POST /api/admin/login      username + password (AUTOSETTLE_ADMIN_USER / _PASSWORD;
                               if no password is set, a random one is printed at startup)
    GET  /api/admin/overview   counts from stored claims + health
    GET  /api/admin/claims     claim queue (?status=...)
    GET  /api/admin/claims/{id} everything: limits, confidences, provenance, negotiation log
    GET  /api/admin/audit      append-only audit log

The customer API never returns the opening/maximum settlement limits, the offer ladder
or the negotiation step.  Those exist only in admin responses.

Security choice (differs from the "frontend sends offer_tier" idea on purpose):
the negotiation state and the quote live ON THE SERVER, keyed by claim_id.
The browser only sends the claim_id and the text.  If the frontend sent
offer_tier, anyone could edit the request to jump straight to the 95 % offer.
Extra fields such as offer_tier are accepted but ignored.

Run (from C:\\autosettle, with the Python that has torch + cv2):
    python -m pip install fastapi uvicorn python-multipart
    python api_server.py                       # http://localhost:8000/docs to try it
Options (environment variables):
    AUTOSETTLE_SEVERITY=vlm   load the Qwen VISION model as a damage witness (default: rules = no VLM).
                              With the VLM loaded, only VLM-confirmed damage can be priced
                              (fail-closed).  AUTOSETTLE_REQUIRE_VLM=0 switches that gate off.
    AUTOSETTLE_VLM_MODEL=...  vision model id (default Qwen/Qwen2.5-VL-3B-Instruct)
    AUTOSETTLE_DINO=tiny      faster, less accurate parts (default: base)
    AUTOSETTLE_LLM=1          use a local Ollama TEXT model for chat intent + wording (default: rules +
                              templates).  Separate from the vision model; never sees images or sets amounts.
    AUTOSETTLE_MAX_VIDEO_S=90 longest accepted video
    AUTOSETTLE_DEMO_MODE=1    show the hidden Demo Mode entry (judges upload any photo; illustrative
                              demo price tier; scoped partial demo offers).  Off by default.
    AUTOSETTLE_CORS=https://your-site.vercel.app   (default: *)

Frontend on Vercel + backend on your laptop: the browser must reach the laptop.
Easiest for the demo is a tunnel:  cloudflared tunnel --url http://localhost:8000
(or ngrok http 8000) and put that https URL in the frontend.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import io
import json
import secrets
import time
from datetime import datetime, timezone
import os
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import negotiation
import pricing_rules
from vehicles import family, norm_type

CLAIMS_DIR = Path(os.environ.get("AUTOSETTLE_CLAIMS_DIR", "claims"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".3gp"}
MAX_UPLOAD_MB = 200
MAX_VIDEO_S = float(os.environ.get("AUTOSETTLE_MAX_VIDEO_S", "90"))
MAX_IMAGES_RETURNED = 8
USE_LLM = os.environ.get("AUTOSETTLE_LLM", "0") == "1"
DEMO_MODE = os.environ.get("AUTOSETTLE_DEMO_MODE", "0") == "1"

MODELS: Dict = {}            # loaded once at startup
CLAIMS: Dict[str, Dict] = {}  # claim_id -> {"quote", "session"}  (also saved to disk)
GPU_LOCK = threading.Lock()   # one claim at a time on the GPU
CHAT_LOCK = threading.Lock()


# =============================================================================
# Startup: load DINO + YOLO (+ VLM) once
# =============================================================================
def load_models() -> None:
    from grounding_dino_parts import PartDetector
    from severity import SeverityAssessor, load_vlm
    from yolo_damage import DamageDetector

    sev = os.environ.get("AUTOSETTLE_SEVERITY", "rules")
    print(f"Loading models (DINO {os.environ.get('AUTOSETTLE_DINO', 'base')}, severity={sev}) ...")
    MODELS["parts"] = PartDetector(model_size=os.environ.get("AUTOSETTLE_DINO", "base"))
    MODELS["damage"] = DamageDetector(os.environ.get("AUTOSETTLE_WEIGHTS"), 0.25)
    vlm = load_vlm("qwen" if sev == "vlm" else "none")
    MODELS["vlm"] = vlm
    MODELS["assessor"] = SeverityAssessor(vlm)
    # Fail-closed: when the vision witness is available, nothing is priced without it.
    pricing_rules.REQUIRE_VLM_CONFIRMATION = vlm is not None and os.environ.get("AUTOSETTLE_REQUIRE_VLM", "1") != "0"
    print(f"Models ready. VLM approval gate: {'ON' if pricing_rules.REQUIRE_VLM_CONFIRMATION else 'OFF'}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("AUTOSETTLE_SKIP_MODELS") != "1":   # tests set this
        load_models()
    yield


app = FastAPI(title="Auto-Settle AI", version="1.0", lifespan=lifespan)
@app.middleware("http")
async def body_size_limit(request, call_next):
    """FastAPI spools a whole multipart upload to a temp file BEFORE the endpoint runs, so the
    size must be refused here, from Content-Length, before the body is read.  (A request
    without Content-Length is still capped by the byte count inside analyze_claim.)"""
    if request.url.path == "/api/analyze_claim":
        try:
            n = int(request.headers.get("content-length") or 0)
        except ValueError:
            n = 0
        if n > (MAX_UPLOAD_MB + 1) * 1024 * 1024:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": f"file larger than {MAX_UPLOAD_MB} MB"}, status_code=413)
    return await call_next(request)


app.add_middleware(CORSMiddleware,
                   allow_origins=[o.strip() for o in os.environ.get("AUTOSETTLE_CORS", "*").split(",")],
                   allow_methods=["*"], allow_headers=["*"])


# =============================================================================
# Helpers
# =============================================================================
def b64(path: Path) -> Optional[str]:
    try:
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return None


def save_claim(claim_id: str) -> None:
    c = CLAIMS[claim_id]
    d = CLAIMS_DIR / claim_id
    c["meta"]["last_activity"] = now_iso()
    (d / "quote.json").write_text(json.dumps(c["quote"], indent=2))
    (d / "session.json").write_text(json.dumps(c["session"], indent=2))
    (d / "meta.json").write_text(json.dumps(c["meta"], indent=2))


def get_claim(claim_id: str) -> Dict:
    if claim_id in CLAIMS:
        return CLAIMS[claim_id]
    d = CLAIMS_DIR / claim_id              # survive a server restart
    if not claim_id.isalnum() or not (d / "meta.json").is_file():
        raise HTTPException(404, "unknown claim")
    CLAIMS[claim_id] = {"quote": json.loads((d / "quote.json").read_text()),
                        "session": json.loads((d / "session.json").read_text()),
                        "meta": json.loads((d / "meta.json").read_text())}
    return CLAIMS[claim_id]


def load_analysis(claim_id: str) -> Dict:
    try:
        return json.loads((CLAIMS_DIR / claim_id / "analysis.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =============================================================================
# Audit log (append-only JSONL, admin-readable only)
# =============================================================================
AUDIT_LOCK = threading.Lock()


def audit(actor: str, action: str, claim_id: Optional[str] = None, **detail) -> None:
    rec = {"time": now_iso(), "actor": actor, "action": action, "claim_id": claim_id, **detail}
    try:
        CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOCK, open(CLAIMS_DIR / "audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as exc:   # never break a claim over logging, but never hide it either
        import sys
        print(f"[audit] WRITE FAILED ({exc}) for {action} {claim_id or ''}", file=sys.stderr)


# =============================================================================
# Customer sessions: HMAC-signed tokens (survive a restart, cannot be forged)
# =============================================================================
def _secret() -> bytes:
    env = os.environ.get("AUTOSETTLE_SECRET")
    if env:
        return env.encode()
    f = CLAIMS_DIR / ".secret"
    try:
        return f.read_bytes()
    except OSError:
        CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
        key = secrets.token_bytes(32)
        f.write_bytes(key)
        return key


def sign(value: str) -> str:
    return hmac.new(_secret(), value.encode(), hashlib.sha256).hexdigest()[:32]


def new_customer_token(kind: str, value: str) -> str:
    """kind 'cs' = policyholder (value = the policy's plate), 'dm' = Demo Mode (value = price tier).
    The value is inside the signature, so the browser cannot switch policy or tier."""
    sid = secrets.token_hex(12)
    body = f"{kind}.{sid}.{value}"
    return f"{body}.{sign(body)}"


def customer(x_session_token: Optional[str] = Header(None)) -> Dict:
    """Dependency: a valid customer token -> {sid, mode, plate | tier}, else 401."""
    parts = (x_session_token or "").split(".")
    if len(parts) != 4 or parts[0] not in ("cs", "dm") or \
            not hmac.compare_digest(sign(".".join(parts[:3])), parts[3]):
        raise HTTPException(401, "customer session missing or invalid - start a new session")
    kind, sid, value = parts[:3]
    if kind == "dm":
        if not DEMO_MODE or value not in pricing_rules.DEMO_TIERS:
            raise HTTPException(401, "demo mode is not available - start a new session")
        return {"sid": sid, "mode": "demo", "tier": value, "plate": None}
    return {"sid": sid, "mode": "policyholder", "tier": None, "plate": value}


def owned_claim(claim_id: str, s: Dict) -> Dict:
    c = get_claim(claim_id)
    if not hmac.compare_digest(c["meta"].get("owner", ""), s["sid"]):
        raise HTTPException(404, "unknown claim")     # same answer as 'does not exist'
    return c


# =============================================================================
# Admin auth: server-side password check -> short-lived bearer token
# =============================================================================
ADMIN_USER = os.environ.get("AUTOSETTLE_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("AUTOSETTLE_ADMIN_PASSWORD") or secrets.token_urlsafe(9)
ADMIN_TOKENS: Dict[str, float] = {}      # token -> expiry (epoch seconds); in memory only
ADMIN_TTL_S = 8 * 3600
FAILED_LOGINS: Dict[str, List[float]] = {}   # source -> failure times (last 5 minutes)
LOGIN_WINDOW_S, LOGIN_MAX_PER_SOURCE, LOGIN_MAX_TOTAL = 300, 10, 100


def admin(authorization: Optional[str] = Header(None)) -> str:
    t = (authorization or "").removeprefix("Bearer ").strip()
    exp = ADMIN_TOKENS.get(t)
    if not t or not exp or exp < time.time():
        ADMIN_TOKENS.pop(t, None)
        raise HTTPException(401, "admin authentication required")
    return ADMIN_USER


# =============================================================================
# Customer-safe views: no settlement limits, ladder, negotiation step or model internals
# =============================================================================
PENDING = ("uncertain_review", "unsupported_review", "duplicate_or_overlap")
LINE_STAFF_ONLY = ("confidence", "low_confidence", "evidence", "finding", "severity_source")


def customer_quote(q: Dict) -> Dict:
    """No settlement limits, confidences, file paths, reason codes or overlap internals."""
    q = copy.deepcopy(q)
    (q.get("totals") or {}).pop("settlement_range", None)
    for l in q.get("line_items", []):
        for k in LINE_STAFF_ONLY:
            l.pop(k, None)
    q["pending_items"] = [{"part": f["part"], "price_key": f.get("price_key"), "damage": f["damage"],
                           "status": f["status"], "reason": negotiation.PLAIN.get(f.get("reason_code") or "",
                                                                                   "needs a specialist's check")}
                          for f in q.get("findings", []) if f["status"] in PENDING]
    q["ruled_out_count"] = sum(1 for f in q.get("findings", []) if f["status"] == "rejected")
    q["plain_reasons"] = negotiation.plain_blockers(q)
    so = q.get("scoped_offer")
    if so:
        fs = {f["id"]: f for f in q.get("findings", [])}
        q["offer_scope"] = {"partial": True,
                            "included": [{"part": fs[i]["part"], "damage": fs[i]["damage"]} for i in so["finding_ids"] if i in fs],
                            "excluded": [{"part": fs[i]["part"], "damage": fs[i]["damage"]}
                                         for i in so["excluded_finding_ids"] if i in fs and fs[i]["status"] in PENDING]}
    for k in ("findings", "blockers", "overlap_groups", "notes", "pricing_data", "scoped_offer"):
        q.pop(k, None)
    return q


def customer_offer(quote: Dict, session: Dict) -> Dict:
    st = session["state"]
    live = st in ("OPEN", "CONFIRMING", "SETTLED")
    return {"status": quote["status"], "state": st,
            "current_offer_inr": session.get("offer_inr") if live else None,
            "insurer_payable_inr": (quote.get("totals") or {}).get("insurer_payable_inr"),
            "estimate_scope": (quote.get("totals") or {}).get("estimate_scope", "none"),
            "offer_scope": ("partial" if quote.get("scoped_offer") else "complete") if live else None,
            # 'partial' = the listed items are settled; the rest of the claim is still under review
            "settlement_scope": session.get("settlement_scope") if st == "SETTLED" else None,
            "can_accept": st in ("OPEN", "CONFIRMING"),
            "can_reconsider": negotiation.can_reconsider(session),
            "settled_amount_inr": session.get("settled_amount_inr"),
            "decision_recorded_at": session.get("decided_at")}


def customer_analysis(result: Dict) -> Dict:
    r = copy.deepcopy({k: v for k, v in result.items() if k != "quote"})
    for d in r.get("damages", []):
        for k in ("detector_confidence", "vlm_agreement", "severity_facts", "vlm_verification",
                  "vlm_valid_views", "status_code", "severity_votes"):   # staff-only detail
            d.pop(k, None)
    if r.get("viewer"):
        for b in r["viewer"]["boxes"]:
            b["confidence"] = None
    v = r.get("vehicle") or {}
    r["vehicle"] = {"type": v.get("type"), "mismatch": v.get("mismatch")}
    r.pop("video_info", None)
    return r


def customer_messages(session: Dict) -> List[Dict]:
    out = [{"from": "bot", "text": session.get("opening", ""), "time": session.get("created_at")}]
    for t in session["history"]:
        out.append({"from": "user", "text": t["user"], "time": t["time"]})
        out.append({"from": "bot", "text": t["reply"], "time": t["time"], "outcome": t["outcome"]})
    return out


def customer_events(c: Dict) -> List[Dict]:
    """Customer activity: only events that really happened, from server records."""
    m, q, s = c["meta"], c["quote"], c["session"]
    ev = [{"time": m["created_at"], "title": "Evidence submitted", "tone": "neutral",
           "detail": "Video" if m.get("media_type") == "video" else "Photo"},
          {"time": m["created_at"], "title": "Assessment completed", "tone": "neutral",
           "detail": f"{m.get('findings', 0)} finding(s)"}]
    if q["status"] == "APPROVED":
        ev.append({"time": m["created_at"], "title": "Estimate and offer presented", "tone": "teal"})
    elif q.get("scoped_offer"):
        ev.append({"time": m["created_at"], "title": "Partial demo offer presented - other items under review",
                   "tone": "teal", "detail": f"{len(q['scoped_offer']['finding_ids'])} item(s) covered"})
    elif q["status"] == "NO_PAYOUT":
        ev.append({"time": m["created_at"], "title": "Estimate generated - no payable amount", "tone": "neutral"})
    elif (q.get("totals") or {}).get("estimate_scope") == "partial":
        ev.append({"time": m["created_at"], "title": "Provisional partial estimate - some items sent for review",
                   "tone": "amber", "detail": f"{q['totals']['items_pending']} item(s) pending"})
    else:
        ev.append({"time": m["created_at"], "title": "Sent for specialist review", "tone": "amber"})
    for t in s["history"]:
        if t["outcome"] in ("COUNTER_OFFER", "FINAL_OFFER"):
            ev.append({"time": t["time"], "title": "Reconsideration requested",
                       "detail": "Revised offer presented" if t["outcome"] == "COUNTER_OFFER" else "Offer unchanged",
                       "tone": "neutral"})
        elif t["outcome"] == "SETTLED" and s.get("settlement_scope") == "partial":
            ev.append({"time": t["time"], "title": "Partially settled - remaining items under review", "tone": "amber",
                       "detail": "Listed items accepted; the remaining items were not accepted or waived. Demo - no money transferred"})
        elif t["outcome"] == "SETTLED":
            ev.append({"time": t["time"], "title": "Offer accepted - decision recorded", "tone": "teal",
                       "detail": "Demo - no money transferred"})
        elif t["outcome"] == "ESCALATED":
            ev.append({"time": t["time"], "title": "Specialist review requested", "tone": "amber"})
    return ev


def customer_claim(claim_id: str, c: Dict) -> Dict:
    return {"claim_id": claim_id, **customer_analysis(load_analysis(claim_id)),
            "quote": customer_quote(c["quote"]), "offer": customer_offer(c["quote"], c["session"]),
            "chat": {"state": c["session"]["state"], "messages": customer_messages(c["session"])},
            "events": customer_events(c), "evidence_retained": (CLAIMS_DIR / claim_id).is_dir(),
            "mode": c["meta"].get("mode", "policyholder"), "demo": c["meta"].get("mode") == "demo",
            "demo_pricing": demo_info(c["quote"]) if c["meta"].get("mode") == "demo" else None}


def demo_info(q: Dict) -> Dict:
    t = q.get("totals") or {}
    req = (q.get("vehicle") or {}).get("demo_tier") or "auto"
    return {"tier_requested": req, "tier_requested_label": pricing_rules.DEMO_TIER_LABELS.get(req, req),
            "tier_used": t.get("price_tier"), "tier_source": t.get("tier_source"),
            "note": "Demo Mode: illustrative rate card, vehicle make/model not identified, no real payment."}


VIEWER_MAX_SIDE = 1600


def viewer_block(image_path: Path, parts: List[Dict], fused_like: Dict) -> Dict:
    """Clean (un-annotated) image + every box in 0..1 coordinates, so the frontend can
    draw, hide and highlight boxes itself (click a receipt line -> its box lights up)."""
    from yolo_damage import load_image
    img = load_image(str(image_path))        # same EXIF rotation the models used
    W, H = img.size
    small = img.copy()
    small.thumbnail((VIEWER_MAX_SIDE, VIEWER_MAX_SIDE))
    buf = io.BytesIO()
    small.convert("RGB").save(buf, "JPEG", quality=85)

    def nb(b):
        return [round(b[0] / W, 4), round(b[1] / H, 4), round(b[2] / W, 4), round(b[3] / H, 4)]

    def key(label):
        return pricing_rules.PART_ALIASES.get(pricing_rules.norm(label))

    boxes = []
    for i, p in enumerate(parts):
        boxes.append({"id": f"p{i}", "kind": "part", "label": p["label"], "price_key": key(p["label"]),
                      "confidence": round(p["confidence"], 3), "bbox": nb(p["bbox"])})
    for i, m in enumerate(fused_like.get("matched", [])):
        boxes.append({"id": f"d{i}", "kind": "damage", "label": m["damage"], "part": m["part"],
                      "price_key": key(m["part"]), "confidence": round(m["damage_confidence"], 3),
                      "severity": m.get("severity") or m.get("severity_hint"), "bbox": nb(m["damage_bbox"])})
    for i, ins in enumerate(fused_like.get("inspect", [])):
        if ins.get("vlm_found_damage"):
            boxes.append({"id": f"v{i}", "kind": "vlm", "label": ins.get("vlm_damage") or "damage?",
                          "part": ins["part"], "price_key": key(ins["part"]), "confidence": None,
                          "severity": ins.get("vlm_severity"), "bbox": nb(ins["bbox"])})
    return {"image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
            "width": W, "height": H, "boxes": boxes}


def policy_view(plate: str) -> Optional[Dict]:
    v = pricing_rules.lookup_policy(plate)
    if not v:
        return None
    vtype = norm_type(v.get("vehicle_type")) or "car"
    return {"plate": "".join(ch for ch in plate.upper() if ch.isalnum()), **v, "vehicle_type": vtype,
            "deductible_inr": pricing_rules.deductible_for(v),
            "endorsements": (["Zero Depreciation"] if v.get("zero_dep") else []),
            "auto_approval_limit_inr": pricing_rules.AUTO_APPROVAL_THRESHOLD_INR}


def part_key(label: str) -> Optional[str]:
    """DINO label -> pricing key, so findings, boxes and receipt lines can be linked."""
    return pricing_rules.PART_ALIASES.get(pricing_rules.norm(label))


def attach_status(damages: List[Dict], quote: Dict) -> None:
    """Pricing findings are in the same order as the damage list (quality items aside)."""
    fs = [f for f in quote.get("findings", []) if f.get("part") != "photo"]
    for d, f in zip(damages, fs):
        d["status"] = f["status"]
        d["status_code"] = f.get("reason_code")
        d["needs_review"] = f["status"] in PENDING


def damage_summary(fused: Dict) -> List[Dict]:
    out = []
    for m in fused.get("matched", []):
        out.append({"part": m["part"], "damage": m["damage"], "price_key": part_key(m["part"]),
                    "detector_confidence": m["damage_confidence"],
                    "severity": m.get("severity") or m.get("severity_hint"),
                    "severity_source": m.get("severity_source"),
                    "severity_reasons": m.get("severity_reasons"),
                    "vlm_confirmed": m.get("vlm_confirmed"), "vlm_verification": m.get("vlm_verification"),
                    "vlm_valid_views": m.get("vlm_valid_views"), "severity_votes": m.get("severity_votes"),
                    "vlm_agreement": m.get("vlm_agreement"), "severity_facts": m.get("severity_facts"),
                    "part_uncertain": bool(m.get("part_uncertain")),
                    "needs_review": bool(m.get("severity_review") or m.get("part_uncertain"))})
    for i in fused.get("inspect", []):
        if i.get("vlm_found_damage"):
            out.append({"part": i["part"], "damage": i.get("vlm_damage"), "price_key": part_key(i["part"]),
                        "severity": i.get("vlm_severity"),
                        "severity_source": "vlm_only", "needs_review": True})
    return out


# =============================================================================
# The two pipelines
# =============================================================================
def analyze_photo(path: Path, out_root: Path, form: Dict) -> Dict:
    from run_pipeline import run_one

    if form.get("demo_tier"):
        vehicle, suspicious = None, []      # Demo Mode: no policy, no plate, no make/model
    else:
        vehicle, suspicious = pricing_rules.resolve_vehicle(plate=form["plate"])
    expected = (vehicle or {}).get("vehicle_type") if form["vehicle_type"] == "auto" else None
    with GPU_LOCK:
        row, fused, parts_res = run_one(path, MODELS["parts"], MODELS["damage"], out_root, 0.30, details=True,
                                assessor=MODELS["assessor"], vehicle_type=form["vehicle_type"],
                                expected_type=expected)
    if form.get("demo_tier"):
        v, sus = pricing_rules.demo_vehicle(form["demo_tier"], fused["vehicle_type"]), []
    else:
        v = dict(vehicle or {})
        v["vehicle_type"] = v.get("vehicle_type") or fused["vehicle_type"]
        sus = list(suspicious) + ([fused["vehicle_mismatch"]] if fused["vehicle_mismatch"] else [])
    quote = pricing_rules.price_claim(pricing_rules.items_from_fused(fused), v, sus, settled=form.get("settled"),
                                      repeat_parts=form.get("repeat_parts"))
    damages = damage_summary(fused)
    attach_status(damages, quote)

    folder = out_root / path.stem
    images = [{"name": "annotated", "image": b64(folder / "fused_annotated.jpg")}]
    for m in fused["matched"][:MAX_IMAGES_RETURNED - 1]:
        if m.get("crop_path"):
            images.append({"name": f"{m['part']} - {m['damage']}", "image": b64(Path(m["crop_path"]))})
    return {"media_type": "photo", "quote": quote, "images": images,
            "viewer": viewer_block(path, parts_res["parts"], fused),
            "damages": damages,
            "vehicle": {"type": fused["vehicle_type"], "detection": fused.get("vehicle_type_detection"),
                        "mismatch": fused["vehicle_mismatch"], "policy": vehicle},
            "rejected_detections": len(fused["rejected"])}


def analyze_video_claim(path: Path, out_root: Path, form: Dict) -> Dict:
    from video_pipeline import analyze_video
    from run_pipeline import MISMATCH_CONF

    pre, _ = (None, []) if form.get("demo_tier") else pricing_rules.resolve_vehicle(plate=form["plate"])
    with GPU_LOCK:
        rep = analyze_video(str(path), MODELS["parts"], MODELS["damage"], out_root, 2.0, 12, 0.30,
                            plate=True, vlm=MODELS["vlm"], vehicle_type=form["vehicle_type"],
                            expected_type=(pre or {}).get("vehicle_type"))
    plate = form["plate"] or (None if form.get("demo_tier") else rep["plate"].get("text"))
    vehicle, suspicious = (None, []) if form.get("demo_tier") else pricing_rules.resolve_vehicle(plate=plate)
    v = dict(vehicle or {})
    ti = rep.get("vehicle_type_detection") or {}
    mismatch = None
    if (v.get("vehicle_type") and ti.get("vehicle_type") and (ti.get("confidence") or 0) >= MISMATCH_CONF
            and family(v["vehicle_type"]) != family(ti["vehicle_type"])):
        mismatch = f"video shows a {ti['vehicle_type']} but the policy is for a {norm_type(v['vehicle_type'])}"
    v["vehicle_type"] = v.get("vehicle_type") or rep["vehicle_type"]
    sus = list(suspicious) + ([mismatch] if mismatch else [])
    if form.get("demo_tier"):
        v, sus = pricing_rules.demo_vehicle(form["demo_tier"], rep["vehicle_type"]), []
    quote = pricing_rules.price_claim(pricing_rules.items_from_video(rep), v, sus, settled=form.get("settled"),
                                      repeat_parts=form.get("repeat_parts"))

    vid_out = out_root / path.stem
    images, damages = [], []
    for d in rep["damages"]:
        ev = d["best_evidence"]
        damages.append({"part": d["part"], "damage": d["damage"], "price_key": part_key(d["part"]),
                        "video_status": d["status"],
                        "frames_seen": d["frames_seen"], "detector_confidence": d["best_confidence"],
                        "severity": d.get("severity") or ev.get("severity"),
                        "severity_reasons": d.get("severity_reasons") or ev.get("severity_reasons"),
                        "vlm_confirmed": d.get("vlm_confirmed"), "vlm_verification": d.get("vlm_verification"),
                        "vlm_valid_views": d.get("vlm_valid_views"), "severity_votes": d.get("severity_votes"),
                        "vlm_agreement": d.get("vlm_agreement"), "severity_facts": d.get("severity_facts"),
                        "needs_review": d["status"] in ("inconsistent", "weak") or bool(d.get("severity_review"))})
        if len(images) < MAX_IMAGES_RETURNED:
            frame_annot = vid_out / Path(ev["frame_path"]).stem / "fused_annotated.jpg"
            images.append({"name": f"{d['part']} - {d['damage']} (t={ev['time_s']}s)", "image": b64(frame_annot)})
    viewer = None
    shown = [d for d in rep["damages"] if d["status"] in ("confirmed", "single_view")] or rep["damages"]
    if shown:   # the best frame of the most important damage, with every damage seen on that frame
        fp = shown[0]["best_evidence"]["frame_path"]
        same = [d for d in rep["damages"] if d["best_evidence"]["frame_path"] == fp]
        pj = vid_out / Path(fp).stem / "parts.json"
        frame_parts = json.loads(pj.read_text())["parts"] if pj.is_file() else []
        fake = {"matched": [{"part": d["part"], "damage": d["damage"], "damage_confidence": d["best_confidence"],
                             "severity": d.get("severity") or d["best_evidence"].get("severity"),
                             "damage_bbox": d["best_evidence"]["bbox"]} for d in same]}
        viewer = viewer_block(Path(fp), frame_parts, fake)
        viewer["caption"] = f"keyframe at t={shown[0]['best_evidence']['time_s']}s"
    attach_status(damages, quote)
    if rep["plate"].get("crop_path"):
        images.append({"name": "licence plate", "image": b64(Path(rep["plate"]["crop_path"]))})
    return {"media_type": "video", "quote": quote, "images": [i for i in images if i["image"]],
            "viewer": viewer,
            "damages": damages,
            "vehicle": {"type": rep["vehicle_type"], "detection": ti or None, "mismatch": mismatch,
                        "policy": vehicle, "plate_read": rep["plate"].get("text")},
            "video_info": rep["video_info"]}


# =============================================================================
# Endpoints - customer
# =============================================================================
FAILED_ASSESSMENTS: List[Dict] = []   # since server start (not persisted)


@app.get("/api/health")
def health():
    return {"ok": True, "models_loaded": bool(MODELS), "demo_mode": DEMO_MODE}


def video_duration_s(path: Path) -> Optional[float]:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return n / fps if fps > 0 and n > 0 else None
    except Exception:  # noqa: BLE001 - unreadable -> the pipeline reports it
        return None


# =============================================================================
# Settled-items ledger: every item ever accepted, in ANY session or mode, append-only.
#   item_id   = "<claim_id>#<finding index>"  (stable: findings are stored with the claim)
#   signature = (canonical part, canonical damage)
#   image_sha = SHA-256 of the uploaded file; image_dhash = 64-bit perceptual hash, so a
#               re-saved / resized copy of the same photo is also recognised.
# A new claim cannot be paid again for an item already paid on the same image, and on a
# policy, the same part + damage from a different photo goes to a person first.
# Limits (said plainly): a DIFFERENT photo of the same damage in Demo Mode cannot be matched,
# because Demo Mode has no vehicle identity.
# =============================================================================
LEDGER_LOCK = threading.Lock()
DHASH_MAX_DISTANCE = 6        # of 64 bits: near-identical copies only


def _ledger_path() -> Path:
    return CLAIMS_DIR / "settled_ledger.jsonl"


def read_ledger() -> List[Dict]:
    try:
        return [json.loads(x) for x in _ledger_path().read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def image_dhash(path: Path) -> Optional[str]:
    """Difference hash of a photo (None for video / unreadable)."""
    try:
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(path)).convert("L").resize((9, 8))
        px = list(img.tobytes())          # one byte per pixel in "L" mode
        bits = "".join("1" if px[r * 9 + c] > px[r * 9 + c + 1] else "0" for r in range(8) for c in range(8))
        return f"{int(bits, 2):016x}"
    except Exception:  # noqa: BLE001
        return None


def _same_image(e: Dict, sha: str, dh: Optional[str]) -> bool:
    if e.get("image_sha") == sha:
        return True
    if dh and e.get("image_dhash"):
        return bin(int(dh, 16) ^ int(e["image_dhash"], 16)).count("1") <= DHASH_MAX_DISTANCE
    return False


def prior_settlements(sha: str, dh: Optional[str], plate: Optional[str]):
    """(signatures paid on this same image, signatures paid on this policy from other photos)"""
    same, policy = set(), set()
    for e in read_ledger():
        sig = tuple(e["signature"])
        if _same_image(e, sha, dh):
            same.add(sig)
        elif plate and e.get("plate") == plate:
            policy.add(sig)
    return same, policy - same


def record_settlement(claim_id: str, c: Dict, items: List[Dict], amount: Optional[int], partial: bool) -> List[str]:
    m = c["meta"]
    ids = []
    with LEDGER_LOCK:
        CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_ledger_path(), "a", encoding="utf-8") as f:
            for it in items:
                item_id = f"{claim_id}#{it['id']}"
                ids.append(item_id)
                f.write(json.dumps({"time": now_iso(), "item_id": item_id, "claim_id": claim_id,
                                    "finding_id": it["id"], "signature": list(pricing_rules.finding_signature(it)),
                                    "image_sha": m.get("image_sha"), "image_dhash": m.get("image_dhash"),
                                    "plate": m.get("plate"), "mode": m.get("mode"),
                                    "claim_amount_inr": amount, "partial": partial}) + "\n")
    return ids


class DemoIn(BaseModel):
    tier: str = "auto"


@app.post("/api/demo_session")
def demo_session(body: Optional[DemoIn] = None):
    """Demo Mode (only when AUTOSETTLE_DEMO_MODE=1): no policy, no plate, no Maruti.  The
    price tier is an operator-chosen assumption, signed into the token."""
    if not DEMO_MODE:
        raise HTTPException(404, "Not Found")          # hidden from normal policyholders
    tier = (body.tier if body else "auto").lower()
    if tier not in pricing_rules.DEMO_TIERS:
        raise HTTPException(422, f"tier must be one of {', '.join(pricing_rules.DEMO_TIERS)}")
    return {"token": new_customer_token("dm", tier), "mode": "demo", "policy": None,
            "demo": {"tier": tier, "tier_label": pricing_rules.DEMO_TIER_LABELS[tier],
                     "note": "Demo Mode: prices are illustrative and no real payment is made."}}


class PolicyLogin(BaseModel):
    policy_no: str = Field(..., max_length=40)
    plate: str = Field(..., max_length=20)


SESSION_FAILS: Dict[str, List[float]] = {}


@app.post("/api/session")
def policyholder_session(body: PolicyLogin, request: Request):
    """Policyholder sign-in: policy number + registration must match the policy record.
    The vehicle, cover and excess then come only from that record."""
    now, src = time.time(), client_source(request)
    SESSION_FAILS[src] = [t for t in SESSION_FAILS.get(src, []) if now - t < 300]
    if len(SESSION_FAILS[src]) >= 20:
        raise HTTPException(429, "too many attempts - wait 5 minutes")
    plate = "".join(ch for ch in body.plate.upper() if ch.isalnum())
    pol = pricing_rules.lookup_policy(plate)
    if not pol or not hmac.compare_digest(pol["policy_no"].upper(), body.policy_no.strip().upper()):
        SESSION_FAILS[src].append(now)
        raise HTTPException(401, "policy number and registration do not match a policy")
    return {"token": new_customer_token("cs", plate), "mode": "policyholder", "policy": policy_view(plate)}


@app.get("/api/policy/{plate}")
def policy(plate: str, s: Dict = Depends(customer)):
    v = policy_view(plate)
    if not v or s["plate"] != v["plate"]:
        raise HTTPException(404, "no active policy for this plate")
    return v


@app.post("/api/analyze_claim")
def analyze_claim(file: UploadFile = File(...),
                  plate: Optional[str] = Form(None), make: Optional[str] = Form(None),
                  model: Optional[str] = Form(None), year: Optional[int] = Form(None),
                  cc: Optional[int] = Form(None), zero_dep: bool = Form(False),
                  vehicle_type: str = Form("auto"), demo_tier: Optional[str] = Form(None),
                  s: Dict = Depends(customer)):
    """Upload a claim photo/video (+ plate) -> findings, estimate and the current offer.

    The vehicle and policy terms come ONLY from the policy record found by the plate.
    make / model / year / cc / zero_dep are still accepted (old clients) but ignored, so a
    request cannot price a claim against self-chosen vehicle or cover details."""
    if not MODELS:
        raise HTTPException(503, "the assessment service is not ready yet - try again in a minute")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        raise HTTPException(415, f"unsupported file type '{ext}' - use a photo or a video")
    if vehicle_type not in ("auto", "car", "motorcycle", "scooter"):
        raise HTTPException(422, "vehicle_type must be auto / car / motorcycle / scooter")

    claim_id = uuid.uuid4().hex[:12]
    folder = CLAIMS_DIR / claim_id
    folder.mkdir(parents=True, exist_ok=True)
    src = folder / f"upload{ext}"
    limit, total = MAX_UPLOAD_MB * 1024 * 1024, 0
    file_sha = hashlib.sha256()
    try:   # count bytes WHILE reading: an oversized upload is stopped at the limit
        with open(src, "wb") as f:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
                file_sha.update(chunk)
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    if total == 0:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(422, "the uploaded file is empty")
    if ext in VIDEO_EXTS:
        dur = video_duration_s(src)
        if dur is not None and dur > MAX_VIDEO_S:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(413, f"video is {dur:.0f} s long - please keep it under {MAX_VIDEO_S:.0f} s")

    if any(x not in (None, False) for x in (make, model, year, cc)) or zero_dep or demo_tier:
        audit("customer", "ignored_vehicle_fields", claim_id)   # policy record / signed token wins
    sha, dh = file_sha.hexdigest(), (image_dhash(src) if ext in IMAGE_EXTS else None)
    if s["mode"] == "demo":
        form = {"plate": None, "vehicle_type": vehicle_type, "demo_tier": s["tier"]}
    else:   # the plate comes from the signed session, never from the form
        if plate and "".join(ch for ch in plate.upper() if ch.isalnum()) != s["plate"]:
            audit("customer", "ignored_plate_field", claim_id)
        form = {"plate": s["plate"], "vehicle_type": vehicle_type}
    form["settled"], form["repeat_parts"] = prior_settlements(sha, dh, form["plate"])
    plate = form["plate"]
    sid = s["sid"]
    try:
        result = (analyze_video_claim if ext in VIDEO_EXTS else analyze_photo)(src, folder / "results", form)
    except Exception as exc:  # noqa: BLE001 - record every failure for the admin view
        FAILED_ASSESSMENTS.append({"time": now_iso(), "claim_id": claim_id, "error": str(exc)[:300]})
        audit("system", "assessment_failed", claim_id, error=str(exc)[:300])
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(422, "we could not assess this file - try a clearer photo or video") from exc

    if plate:
        result["policy"] = policy_view(plate)
    quote = result["quote"]
    session = negotiation.new_session(claim_id, quote)
    session["created_at"] = now_iso()
    session["opening"] = negotiation.opening_message(session, quote)
    meta = {"claim_id": claim_id, "owner": sid, "mode": s["mode"], "demo_tier": s["tier"],
            "image_sha": sha, "image_dhash": dh,
            "created_at": session["created_at"],
            "plate": plate, "media_type": result["media_type"], "vehicle_type": result["vehicle"]["type"],
            "findings": len(result["damages"])}
    (folder / "analysis.json").write_text(json.dumps(result))
    with CHAT_LOCK:
        CLAIMS[claim_id] = {"quote": quote, "session": session, "meta": meta}
        save_claim(claim_id)
    audit("system", "claim_assessed", claim_id, status=quote["status"], reasons=quote["reasons"],
          offer_inr=session.get("offer_inr"), mode=s["mode"],
          demo_tier=(quote.get("totals") or {}).get("price_tier") if s["mode"] == "demo" else None,
          tier_source=(quote.get("totals") or {}).get("tier_source"),
          scoped_offer=(quote.get("scoped_offer") or {}).get("finding_ids"),
          demo_rules=[{"finding": f["id"], "rule": f["demo_rule"]} for f in quote.get("findings", []) if f.get("demo_rule")],
          excluded_already_paid=[f["id"] for f in quote.get("findings", []) if f.get("reason_code") == "ALREADY_SETTLED"])
    return customer_claim(claim_id, CLAIMS[claim_id])


class ChatIn(BaseModel):
    claim_id: str
    message: str = Field(..., min_length=1, max_length=500)


@app.post("/api/chat_negotiate")
def chat_negotiate(body: ChatIn, s: Dict = Depends(customer)):
    """Intent router + Python state machine. Returns only customer-safe fields."""
    with CHAT_LOCK:
        c = owned_claim(body.claim_id, s)
        r = negotiation.handle_message(c["session"], c["quote"], body.message, USE_LLM)
        accepted_ids = None
        if r["state"] == "SETTLED" and not c["session"].get("decided_at"):   # exactly once per claim
            c["session"]["decided_at"] = now_iso()
            q = c["quote"]
            covered = [f for f in q.get("findings", []) if f["status"] == "confirmed_priceable"]
            partial = bool(q.get("scoped_offer"))
            c["session"]["settled_finding_ids"] = [f["id"] for f in covered]
            c["session"]["settlement_scope"] = "partial" if partial else "full"
            accepted_ids = record_settlement(body.claim_id, c, covered, r["settled_amount_inr"], partial)
            c["session"]["settled_item_ids"] = accepted_ids
        save_claim(body.claim_id)
    audit("customer", "chat_turn", body.claim_id, intent=r["intent"], outcome=r["outcome"],
          state=r["state"], step=r["tier"], offer_inr=r["offer_inr"])
    if r["outcome"] == "SETTLED" and accepted_ids is not None:
        audit("customer", "offer_accepted", body.claim_id, amount_inr=r["settled_amount_inr"],
              item_ids=accepted_ids, scope=c["session"]["settlement_scope"],
              still_under_review=[f["id"] for f in c["quote"].get("findings", []) if f["status"] in PENDING])
    elif r["outcome"] == "ESCALATED":
        audit("customer", "escalated_to_human", body.claim_id)
    return {"claim_id": body.claim_id, "reply": r["reply"], "outcome": r["outcome"],
            "state": r["state"], "offer": customer_offer(c["quote"], c["session"]),
            "events": customer_events(c)}


@app.get("/api/claim/{claim_id}")
def claim_detail(claim_id: str, s: Dict = Depends(customer)):
    return customer_claim(claim_id, owned_claim(claim_id, s))


# =============================================================================
# Endpoints - admin (separate auth; customer tokens are rejected)
# =============================================================================
class AdminLogin(BaseModel):
    username: str = Field(..., max_length=64)
    password: str = Field(..., max_length=128)


def client_source(request: Request) -> str:
    """Behind the Cloudflare tunnel every request comes from localhost, so the tunnel's
    client-IP header is used when present."""
    h = request.headers
    ip = h.get("cf-connecting-ip") or (h.get("x-forwarded-for") or "").split(",")[0].strip()
    return ip or (request.client.host if request.client else "unknown")


@app.post("/api/admin/login")
def admin_login(body: AdminLogin, request: Request):
    now = time.time()
    src = client_source(request)
    for k in list(FAILED_LOGINS):
        FAILED_LOGINS[k] = [t for t in FAILED_LOGINS[k] if now - t < LOGIN_WINDOW_S]
        if not FAILED_LOGINS[k]:
            del FAILED_LOGINS[k]
    # per source: one guesser is blocked without locking out staff elsewhere;
    # a high overall cap still slows a distributed attack.
    if len(FAILED_LOGINS.get(src, [])) >= LOGIN_MAX_PER_SOURCE or \
            sum(len(v) for v in FAILED_LOGINS.values()) >= LOGIN_MAX_TOTAL:
        raise HTTPException(429, "too many failed attempts - wait 5 minutes")
    ok = hmac.compare_digest(body.username.encode(), ADMIN_USER.encode()) & hmac.compare_digest(body.password.encode(), ADMIN_PASSWORD.encode())
    if not ok:
        FAILED_LOGINS.setdefault(src, []).append(now)
        audit(body.username[:64], "admin_login_failed")
        raise HTTPException(401, "invalid username or password")
    token = secrets.token_urlsafe(32)
    ADMIN_TOKENS[token] = now + ADMIN_TTL_S
    audit(ADMIN_USER, "admin_login")
    return {"token": token, "user": ADMIN_USER,
            "expires_at": datetime.fromtimestamp(now + ADMIN_TTL_S, timezone.utc).isoformat(timespec="seconds")}


@app.post("/api/admin/logout")
def admin_logout(authorization: Optional[str] = Header(None), user: str = Depends(admin)):
    ADMIN_TOKENS.pop((authorization or "").removeprefix("Bearer ").strip(), None)
    return {"ok": True}


def all_claims() -> List[Dict]:
    out = []
    if not CLAIMS_DIR.is_dir():
        return out
    for d in CLAIMS_DIR.iterdir():
        if d.is_dir() and (d / "meta.json").is_file():
            try:
                c = get_claim(d.name)
            except HTTPException:
                continue
            q, s, m = c["quote"], c["session"], c["meta"]
            out.append({"claim_id": d.name, "created_at": m.get("created_at"),
                        "last_activity": m.get("last_activity"), "plate": m.get("plate"),
                        "vehicle_type": m.get("vehicle_type"), "media_type": m.get("media_type"),
                        "mode": m.get("mode", "policyholder"), "demo_tier": m.get("demo_tier"),
                        "scoped_offer": bool(q.get("scoped_offer")),
                        "settlement_scope": s.get("settlement_scope"),
                        "findings": m.get("findings"), "quote_status": q["status"], "chat_state": s["state"],
                        "current_offer_inr": s.get("offer_inr"), "settled_amount_inr": s.get("settled_amount_inr"),
                        "garage_estimate_inr": (q.get("totals") or {}).get("garage_estimate_inr"),
                        "reasons": q["reasons"] if q["status"] != "APPROVED" else [],
                        "blocker_codes": sorted({b["code"] for b in q.get("blockers", [])}),
                        "estimate_scope": (q.get("totals") or {}).get("estimate_scope"),
                        "items_pending": (q.get("totals") or {}).get("items_pending", 0),
                        "needs_review": q["status"] in ("REVIEW_REQUIRED", "ESCALATED") or s["state"] == "ESCALATED"})
    return sorted(out, key=lambda x: x.get("last_activity") or "", reverse=True)


@app.get("/api/admin/overview")
def admin_overview(user: str = Depends(admin)):
    cl = all_claims()
    count = lambda f: sum(1 for c in cl if f(c))  # noqa: E731
    return {"claims_total": len(cl),
            "awaiting_customer": count(lambda c: c["chat_state"] in ("OPEN", "CONFIRMING")),
            "needs_review": count(lambda c: c["needs_review"]),
            "settled": count(lambda c: c["chat_state"] == "SETTLED"),
            "no_payout": count(lambda c: c["quote_status"] == "NO_PAYOUT"),
            "failed_assessments_since_start": len(FAILED_ASSESSMENTS),
            "recent_failures": FAILED_ASSESSMENTS[-5:],
            "health": {"models_loaded": bool(MODELS),
                       "severity": "vlm" if MODELS.get("vlm") else "rules",
                       "vision_model": getattr(MODELS.get("vlm"), "name", None),
                       "vlm_approval_gate": pricing_rules.REQUIRE_VLM_CONFIRMATION,
                       "chat_llm": USE_LLM,
                       "chat_llm_model": negotiation.OLLAMA_MODEL if USE_LLM else None,
                       "dino": os.environ.get("AUTOSETTLE_DINO", "base"), "demo_mode": DEMO_MODE},
            "missing_capabilities": [
                "No employee roles/permissions - a single admin account only",
                "No override API: staff cannot change an offer or decision from the portal",
                "Failed-assessment count resets when the server restarts",
                "No live per-stage processing events (assessment is one blocking call)"]}


@app.get("/api/admin/claims")
def admin_claims(status: Optional[str] = None, user: str = Depends(admin)):
    cl = all_claims()
    if status == "review":
        cl = [c for c in cl if c["needs_review"]]
    elif status:
        cl = [c for c in cl if c["quote_status"] == status or c["chat_state"] == status]
    return cl


@app.get("/api/admin/claims/{claim_id}")
def admin_claim(claim_id: str, user: str = Depends(admin)):
    c = get_claim(claim_id)
    audit(user, "admin_view_claim", claim_id)
    s = c["session"]
    return {"claim_id": claim_id, "meta": {k: v for k, v in c["meta"].items() if k != "owner"},
            "policy": policy_view(c["meta"]["plate"]) if c["meta"].get("plate") else None,
            "analysis": load_analysis(claim_id), "quote": c["quote"],
            "negotiation": {"state": s["state"], "ladder_inr": s.get("ladder_inr"), "step": s.get("tier"),
                            "current_offer_inr": s.get("offer_inr"), "final_notice": s.get("final_notice", False),
                            "settled_amount_inr": s.get("settled_amount_inr"), "history": s["history"],
                            "settled_finding_ids": s.get("settled_finding_ids"),
                            "settled_item_ids": s.get("settled_item_ids"),
                            "settlement_scope": s.get("settlement_scope")},
            "ledger": [e for e in read_ledger() if e.get("claim_id") == claim_id],
            "audit": [a for a in read_audit(2000) if a.get("claim_id") == claim_id]}


def read_audit(limit: int) -> List[Dict]:
    try:
        lines = (CLAIMS_DIR / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(x) for x in lines if x.strip()]
    except (OSError, json.JSONDecodeError):
        return []


@app.get("/api/admin/audit")
def admin_audit(limit: int = 200, user: str = Depends(admin)):
    return list(reversed(read_audit(min(max(limit, 1), 2000))))


if __name__ == "__main__":
    import uvicorn
    if not os.environ.get("AUTOSETTLE_ADMIN_PASSWORD"):
        print(f"\n  Admin portal login:  user '{ADMIN_USER}'  password '{ADMIN_PASSWORD}'"
              "\n  (set AUTOSETTLE_ADMIN_PASSWORD to choose your own)\n")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
