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
    AUTOSETTLE_SEVERITY=vlm   use Qwen2.5-VL for severity (default: rules)
    AUTOSETTLE_DINO=tiny      faster, less accurate parts (default: base)
    AUTOSETTLE_LLM=1          use Ollama for chat intent + wording (default: rules + templates)
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

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import negotiation
import pricing_rules
from vehicles import family, norm_type

CLAIMS_DIR = Path(os.environ.get("AUTOSETTLE_CLAIMS_DIR", "claims"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".3gp"}
MAX_UPLOAD_MB = 200
MAX_IMAGES_RETURNED = 8
USE_LLM = os.environ.get("AUTOSETTLE_LLM", "0") == "1"

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
    print("Models ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("AUTOSETTLE_SKIP_MODELS") != "1":   # tests set this
        load_models()
    yield


app = FastAPI(title="Auto-Settle AI", version="1.0", lifespan=lifespan)
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
    except OSError:
        pass


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


def new_customer_token() -> str:
    sid = secrets.token_hex(12)
    return f"cs.{sid}.{sign('cs.' + sid)}"


def customer(x_session_token: Optional[str] = Header(None)) -> str:
    """Dependency: returns the session id of a valid CUSTOMER token, else 401."""
    t = x_session_token or ""
    parts = t.split(".")
    if len(parts) != 3 or parts[0] != "cs" or not hmac.compare_digest(sign("cs." + parts[1]), parts[2]):
        raise HTTPException(401, "customer session missing or invalid - start a new session")
    return parts[1]


def owned_claim(claim_id: str, sid: str) -> Dict:
    c = get_claim(claim_id)
    if not hmac.compare_digest(c["meta"].get("owner", ""), sid):
        raise HTTPException(404, "unknown claim")     # same answer as 'does not exist'
    return c


# =============================================================================
# Admin auth: server-side password check -> short-lived bearer token
# =============================================================================
ADMIN_USER = os.environ.get("AUTOSETTLE_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("AUTOSETTLE_ADMIN_PASSWORD") or secrets.token_urlsafe(9)
ADMIN_TOKENS: Dict[str, float] = {}      # token -> expiry (epoch seconds); in memory only
ADMIN_TTL_S = 8 * 3600
FAILED_LOGINS: List[float] = []


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
def customer_quote(q: Dict) -> Dict:
    q = copy.deepcopy(q)
    (q.get("totals") or {}).pop("settlement_range", None)
    return q


def customer_offer(quote: Dict, session: Dict) -> Dict:
    st = session["state"]
    live = st in ("OPEN", "CONFIRMING", "SETTLED")
    return {"status": quote["status"], "state": st,
            "current_offer_inr": session.get("offer_inr") if live else None,
            "insurer_payable_inr": (quote.get("totals") or {}).get("insurer_payable_inr"),
            "can_accept": st in ("OPEN", "CONFIRMING"),
            "can_reconsider": negotiation.can_reconsider(session),
            "settled_amount_inr": session.get("settled_amount_inr"),
            "decision_recorded_at": session.get("decided_at")}


def customer_analysis(result: Dict) -> Dict:
    r = copy.deepcopy({k: v for k, v in result.items() if k != "quote"})
    for d in r.get("damages", []):
        for k in ("detector_confidence", "vlm_agreement", "severity_facts"):   # staff-only detail
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
    elif q["status"] == "NO_PAYOUT":
        ev.append({"time": m["created_at"], "title": "Estimate generated - no payable amount", "tone": "neutral"})
    else:
        ev.append({"time": m["created_at"], "title": "Sent for specialist review", "tone": "amber"})
    for t in s["history"]:
        if t["outcome"] in ("COUNTER_OFFER", "FINAL_OFFER"):
            ev.append({"time": t["time"], "title": "Reconsideration requested",
                       "detail": "Revised offer presented" if t["outcome"] == "COUNTER_OFFER" else "Offer unchanged",
                       "tone": "neutral"})
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
            "demo": True}


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


def damage_summary(fused: Dict) -> List[Dict]:
    out = []
    for m in fused.get("matched", []):
        out.append({"part": m["part"], "damage": m["damage"], "price_key": part_key(m["part"]),
                    "detector_confidence": m["damage_confidence"],
                    "severity": m.get("severity") or m.get("severity_hint"),
                    "severity_source": m.get("severity_source"),
                    "severity_reasons": m.get("severity_reasons"),
                    "vlm_confirmed": m.get("vlm_confirmed"),
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

    vehicle, suspicious = pricing_rules.resolve_vehicle(
        form["make"], form["model"], form["year"], form["cc"], form["zero_dep"], form["plate"],
        vehicle_type=form["vehicle_type"] if form["vehicle_type"] != "auto" else None)
    expected = (vehicle or {}).get("vehicle_type") if form["vehicle_type"] == "auto" else None
    with GPU_LOCK:
        row, fused, parts_res = run_one(path, MODELS["parts"], MODELS["damage"], out_root, 0.30, details=True,
                                assessor=MODELS["assessor"], vehicle_type=form["vehicle_type"],
                                expected_type=expected)
    v = dict(vehicle or {})
    v["vehicle_type"] = v.get("vehicle_type") or fused["vehicle_type"]
    sus = list(suspicious) + ([fused["vehicle_mismatch"]] if fused["vehicle_mismatch"] else [])
    quote = pricing_rules.price_claim(pricing_rules.items_from_fused(fused), v, sus)

    folder = out_root / path.stem
    images = [{"name": "annotated", "image": b64(folder / "fused_annotated.jpg")}]
    for m in fused["matched"][:MAX_IMAGES_RETURNED - 1]:
        if m.get("crop_path"):
            images.append({"name": f"{m['part']} - {m['damage']}", "image": b64(Path(m["crop_path"]))})
    return {"media_type": "photo", "quote": quote, "images": images,
            "viewer": viewer_block(path, parts_res["parts"], fused),
            "damages": damage_summary(fused),
            "vehicle": {"type": fused["vehicle_type"], "detection": fused.get("vehicle_type_detection"),
                        "mismatch": fused["vehicle_mismatch"], "policy": vehicle},
            "rejected_detections": len(fused["rejected"])}


def analyze_video_claim(path: Path, out_root: Path, form: Dict) -> Dict:
    from video_pipeline import analyze_video
    from run_pipeline import MISMATCH_CONF

    pre, _ = pricing_rules.resolve_vehicle(
        form["make"], form["model"], form["year"], form["cc"], form["zero_dep"], form["plate"],
        vehicle_type=form["vehicle_type"] if form["vehicle_type"] != "auto" else None)
    with GPU_LOCK:
        rep = analyze_video(str(path), MODELS["parts"], MODELS["damage"], out_root, 2.0, 12, 0.30,
                            plate=True, vlm=MODELS["vlm"], vehicle_type=form["vehicle_type"],
                            expected_type=(pre or {}).get("vehicle_type"))
    plate = form["plate"] or rep["plate"].get("text")
    vehicle, suspicious = pricing_rules.resolve_vehicle(
        form["make"], form["model"], form["year"], form["cc"], form["zero_dep"], plate,
        vehicle_type=form["vehicle_type"] if form["vehicle_type"] != "auto" else None)
    v = dict(vehicle or {})
    ti = rep.get("vehicle_type_detection") or {}
    mismatch = None
    if (v.get("vehicle_type") and ti.get("vehicle_type") and (ti.get("confidence") or 0) >= MISMATCH_CONF
            and family(v["vehicle_type"]) != family(ti["vehicle_type"])):
        mismatch = f"video shows a {ti['vehicle_type']} but the policy is for a {norm_type(v['vehicle_type'])}"
    v["vehicle_type"] = v.get("vehicle_type") or rep["vehicle_type"]
    sus = list(suspicious) + ([mismatch] if mismatch else [])
    quote = pricing_rules.price_claim(pricing_rules.items_from_video(rep), v, sus)

    vid_out = out_root / path.stem
    images, damages = [], []
    for d in rep["damages"]:
        ev = d["best_evidence"]
        damages.append({"part": d["part"], "damage": d["damage"], "price_key": part_key(d["part"]),
                        "video_status": d["status"],
                        "frames_seen": d["frames_seen"], "detector_confidence": d["best_confidence"],
                        "severity": d.get("severity") or ev.get("severity"),
                        "severity_reasons": d.get("severity_reasons") or ev.get("severity_reasons"),
                        "vlm_confirmed": d.get("vlm_confirmed"),
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
    return {"ok": True, "models_loaded": bool(MODELS)}


@app.post("/api/demo_session")
def demo_session():
    """One-click demo login: a signed customer token + a synthetic policy (2021 Maruti Swift)."""
    return {"token": new_customer_token(), "mode": "demo",
            "policy": policy_view("DL3CAB1234"),
            "other_demo_policies": [policy_view(p) for p in pricing_rules.DEMO_POLICIES if p != "DL3CAB1234"]}


@app.get("/api/policy/{plate}")
def policy(plate: str, sid: str = Depends(customer)):
    v = policy_view(plate)
    if not v:
        raise HTTPException(404, "no active policy for this plate")
    return v


@app.post("/api/analyze_claim")
def analyze_claim(file: UploadFile = File(...),
                  plate: Optional[str] = Form(None), make: Optional[str] = Form(None),
                  model: Optional[str] = Form(None), year: Optional[int] = Form(None),
                  cc: Optional[int] = Form(None), zero_dep: bool = Form(False),
                  vehicle_type: str = Form("auto"), sid: str = Depends(customer)):
    """Upload a claim photo/video (+ plate) -> findings, estimate and the current offer."""
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
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if src.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")

    form = {"plate": plate, "make": make, "model": model, "year": year, "cc": cc,
            "zero_dep": zero_dep, "vehicle_type": vehicle_type}
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
    meta = {"claim_id": claim_id, "owner": sid, "created_at": session["created_at"],
            "plate": plate, "media_type": result["media_type"], "vehicle_type": result["vehicle"]["type"],
            "findings": len(result["damages"])}
    (folder / "analysis.json").write_text(json.dumps(result))
    with CHAT_LOCK:
        CLAIMS[claim_id] = {"quote": quote, "session": session, "meta": meta}
        save_claim(claim_id)
    audit("system", "claim_assessed", claim_id, status=quote["status"], reasons=quote["reasons"],
          offer_inr=session.get("offer_inr"))
    return customer_claim(claim_id, CLAIMS[claim_id])


class ChatIn(BaseModel):
    claim_id: str
    message: str = Field(..., min_length=1, max_length=500)


@app.post("/api/chat_negotiate")
def chat_negotiate(body: ChatIn, sid: str = Depends(customer)):
    """Intent router + Python state machine. Returns only customer-safe fields."""
    with CHAT_LOCK:
        c = owned_claim(body.claim_id, sid)
        r = negotiation.handle_message(c["session"], c["quote"], body.message, USE_LLM)
        if r["state"] == "SETTLED" and not c["session"].get("decided_at"):
            c["session"]["decided_at"] = now_iso()
        save_claim(body.claim_id)
    audit("customer", "chat_turn", body.claim_id, intent=r["intent"], outcome=r["outcome"],
          state=r["state"], step=r["tier"], offer_inr=r["offer_inr"])
    if r["outcome"] in ("SETTLED", "ESCALATED"):
        audit("customer", "offer_accepted" if r["outcome"] == "SETTLED" else "escalated_to_human",
              body.claim_id, amount_inr=r["settled_amount_inr"])
    return {"claim_id": body.claim_id, "reply": r["reply"], "outcome": r["outcome"],
            "state": r["state"], "offer": customer_offer(c["quote"], c["session"]),
            "events": customer_events(c)}


@app.get("/api/claim/{claim_id}")
def claim_detail(claim_id: str, sid: str = Depends(customer)):
    return customer_claim(claim_id, owned_claim(claim_id, sid))


# =============================================================================
# Endpoints - admin (separate auth; customer tokens are rejected)
# =============================================================================
class AdminLogin(BaseModel):
    username: str = Field(..., max_length=64)
    password: str = Field(..., max_length=128)


@app.post("/api/admin/login")
def admin_login(body: AdminLogin):
    now = time.time()
    FAILED_LOGINS[:] = [t for t in FAILED_LOGINS if now - t < 300]
    if len(FAILED_LOGINS) >= 10:
        raise HTTPException(429, "too many failed attempts - wait 5 minutes")
    ok = hmac.compare_digest(body.username.encode(), ADMIN_USER.encode()) & hmac.compare_digest(body.password.encode(), ADMIN_PASSWORD.encode())
    if not ok:
        FAILED_LOGINS.append(now)
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
                        "findings": m.get("findings"), "quote_status": q["status"], "chat_state": s["state"],
                        "current_offer_inr": s.get("offer_inr"), "settled_amount_inr": s.get("settled_amount_inr"),
                        "garage_estimate_inr": (q.get("totals") or {}).get("garage_estimate_inr"),
                        "reasons": q["reasons"] if q["status"] != "APPROVED" else [],
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
                       "chat_llm": USE_LLM, "dino": os.environ.get("AUTOSETTLE_DINO", "base")},
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
                            "settled_amount_inr": s.get("settled_amount_inr"), "history": s["history"]},
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
