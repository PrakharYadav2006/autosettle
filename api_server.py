"""
Auto-Settle AI - FastAPI backend for the custom (Vercel) frontend
=================================================================

    POST /api/analyze_claim    photo or video  -> damage, severity, quote, first offer
    POST /api/chat_negotiate   {claim_id, message} -> Stage 6 negotiation reply
    GET  /api/claim/{id}       quote + full chat history (audit)
    GET  /api/health

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
import json
import os
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
    (d / "quote.json").write_text(json.dumps(c["quote"], indent=2))
    (d / "session.json").write_text(json.dumps(c["session"], indent=2))


def get_claim(claim_id: str) -> Dict:
    if claim_id in CLAIMS:
        return CLAIMS[claim_id]
    d = CLAIMS_DIR / claim_id              # survive a server restart
    if not claim_id.isalnum() or not (d / "session.json").is_file():
        raise HTTPException(404, "unknown claim_id")
    CLAIMS[claim_id] = {"quote": json.loads((d / "quote.json").read_text()),
                        "session": json.loads((d / "session.json").read_text())}
    return CLAIMS[claim_id]


def offer_block(quote: Dict, session: Dict) -> Dict:
    rng = (quote.get("totals") or {}).get("settlement_range")
    return {"status": quote["status"], "can_negotiate": session["state"] == "OPEN",
            "current_offer_inr": session.get("offer_inr"),
            "opening_offer_inr": rng["opening_offer_inr"] if rng else None,
            "maximum_offer_inr": rng["maximum_offer_inr"] if rng else None,
            "insurer_payable_inr": (quote.get("totals") or {}).get("insurer_payable_inr")}


def damage_summary(fused: Dict) -> List[Dict]:
    out = []
    for m in fused.get("matched", []):
        out.append({"part": m["part"], "damage": m["damage"],
                    "detector_confidence": m["damage_confidence"],
                    "severity": m.get("severity") or m.get("severity_hint"),
                    "severity_source": m.get("severity_source"),
                    "severity_reasons": m.get("severity_reasons"),
                    "vlm_confirmed": m.get("vlm_confirmed"),
                    "needs_review": bool(m.get("severity_review") or m.get("part_uncertain"))})
    for i in fused.get("inspect", []):
        if i.get("vlm_found_damage"):
            out.append({"part": i["part"], "damage": i.get("vlm_damage"), "severity": i.get("vlm_severity"),
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
        row, fused, _ = run_one(path, MODELS["parts"], MODELS["damage"], out_root, 0.30, details=True,
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
        damages.append({"part": d["part"], "damage": d["damage"], "video_status": d["status"],
                        "frames_seen": d["frames_seen"], "detector_confidence": d["best_confidence"],
                        "severity": d.get("severity") or ev.get("severity"),
                        "severity_reasons": d.get("severity_reasons") or ev.get("severity_reasons"),
                        "vlm_confirmed": d.get("vlm_confirmed"),
                        "needs_review": d["status"] in ("inconsistent", "weak") or bool(d.get("severity_review"))})
        if len(images) < MAX_IMAGES_RETURNED:
            frame_annot = vid_out / Path(ev["frame_path"]).stem / "fused_annotated.jpg"
            images.append({"name": f"{d['part']} - {d['damage']} (t={ev['time_s']}s)", "image": b64(frame_annot)})
    if rep["plate"].get("crop_path"):
        images.append({"name": "licence plate", "image": b64(Path(rep["plate"]["crop_path"]))})
    return {"media_type": "video", "quote": quote, "images": [i for i in images if i["image"]],
            "damages": damages,
            "vehicle": {"type": rep["vehicle_type"], "detection": ti or None, "mismatch": mismatch,
                        "policy": vehicle, "plate_read": rep["plate"].get("text")},
            "video_info": rep["video_info"]}


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/api/health")
def health():
    return {"ok": True, "models_loaded": bool(MODELS), "severity": "vlm" if MODELS.get("vlm") else "rules",
            "chat_llm": USE_LLM}


@app.post("/api/analyze_claim")
def analyze_claim(file: UploadFile = File(...),
                  plate: Optional[str] = Form(None), make: Optional[str] = Form(None),
                  model: Optional[str] = Form(None), year: Optional[int] = Form(None),
                  cc: Optional[int] = Form(None), zero_dep: bool = Form(False),
                  vehicle_type: str = Form("auto")):
    """Upload a claim photo/video (+ plate or vehicle details) -> damage, quote, first offer."""
    if not MODELS:
        raise HTTPException(503, "models not loaded")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        raise HTTPException(415, f"unsupported file type '{ext}'")
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
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(422, f"could not analyse the file: {exc}") from exc

    quote = result["quote"]
    session = negotiation.new_session(claim_id, quote)
    with CHAT_LOCK:
        CLAIMS[claim_id] = {"quote": quote, "session": session}
        save_claim(claim_id)
    return {"claim_id": claim_id, **result,
            "offer": offer_block(quote, session),
            "chat": {"opening_message": negotiation.opening_message(session, quote), "state": session["state"]}}


class ChatIn(BaseModel):
    claim_id: str
    message: str = Field(..., min_length=1, max_length=500)
    offer_tier: Optional[int] = None   # accepted for compatibility, IGNORED (server keeps the state)


@app.post("/api/chat_negotiate")
def chat_negotiate(body: ChatIn):
    """Stage 6: intent router + Python state machine.  The reply never contains an
    amount that Python did not authorise."""
    with CHAT_LOCK:
        c = get_claim(body.claim_id)
        r = negotiation.handle_message(c["session"], c["quote"], body.message, USE_LLM)
        save_claim(body.claim_id)
    return {"claim_id": body.claim_id, **r, "offer": offer_block(c["quote"], c["session"])}


@app.get("/api/claim/{claim_id}")
def claim_detail(claim_id: str):
    c = get_claim(claim_id)
    return {"claim_id": claim_id, "quote": c["quote"], "offer": offer_block(c["quote"], c["session"]),
            "chat_state": c["session"]["state"], "history": c["session"]["history"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
