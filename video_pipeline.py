"""
Auto-Settle AI — VIDEO mode: walk-around video -> verified damage list
=====================================================================

    video -> pick sharp, distinct keyframes -> Stages 1-3 on each frame
          -> group the same damage across frames -> multi-frame consensus
          -> per-part claims + list of what needs VLM / human review

Why multi-frame consensus matters: a reflection that looks like a dent in one
frame usually disappears when the camera moves. Real damage stays put. So:

    CONFIRMED     seen in >= 2 frames                        -> claim
    SINGLE VIEW   seen in 1 frame, but its part was only
                  visible in <= 2 frames (no chance to
                  re-check) and confidence is decent         -> claim, VLM must confirm
    INCONSISTENT  seen in 1 frame although its part was
                  clearly visible in >= 3 frames             -> likely glare -> review
    WEAK          seen in 1 frame with low confidence        -> review

Licence plate: we DETECT and crop the plate. If `easyocr` is installed we also
read it; otherwise the crop is saved for a human / later step. We never invent
a plate number or claim a policy check we didn't do.

Usage
-----
    python video_pipeline.py claim.mp4
    python run_pipeline.py claim.mp4                 # same thing
    python video_pipeline.py claim.mp4 --fps 2 --max-frames 12 --out-dir results

Outputs in <out-dir>\\<video name>\\:
    frames\\frame_XX_t<sec>s.jpg      the chosen keyframes
    frame_XX_...\\                    full Stage 1-3 output per frame
    best_<n>_<part>-<damage>.jpg     best crop of each damage (for the VLM)
    plate.jpg                        licence plate crop (if found)
    video_report.json                everything, incl. per-damage evidence
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from fuse import SEVERITY_RANK, crop
from grounding_dino_parts import PartDetector
from run_pipeline import MISMATCH_CONF, decide_vehicle_type, run_one
from vehicles import family, norm_type
from yolo_damage import load_image
from severity import SeverityAssessor, load_vlm
from yolo_damage import DamageDetector
import pricing_rules

# ----------------------------------------------------------------------------- #
# Keyframe selection
# ----------------------------------------------------------------------------- #
BLUR_RATIO = 0.4        # drop frames less than 40% as sharp as the video's median frame
DUPLICATE_DIFF = 4.0    # mean pixel change (0-255) below which two frames count as "same view"
MIN_CONF_SINGLE = 0.5   # single-frame damage needs at least this confidence to be claimed
VISIBLE_FOR_RECHECK = 3 # part seen in >= this many frames -> one-frame damage is suspicious


def sharpness(gray_small: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_small, cv2.CV_64F).var())


def thumb(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)


def extract_keyframes(video_path: str, out_dir: Path, sample_fps: float = 2.0,
                      max_frames: int = 12) -> Tuple[List[Dict], Dict]:
    """Best (sharpest) frame per time window, minus blurry and near-duplicate frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    window = max(1, int(round(fps / sample_fps)))   # frames per time window
    step = 2 if fps > 40 else 1                     # skip every other frame on 60fps phones

    best: Dict[int, Tuple[float, int, np.ndarray]] = {}
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                scale = 480 / max(h, w)
                small = cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale))),
                                     cv2.COLOR_BGR2GRAY) if scale < 1 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sh = sharpness(small)
                win = idx // window
                if win not in best or sh > best[win][0]:
                    best[win] = (sh, idx, frame.copy())
        idx += 1
    cap.release()
    if not best:
        raise ValueError("No frames could be read from the video")

    info = {"fps": round(fps, 2), "frames_in_video": total or idx,
            "duration_s": round((total or idx) / fps, 2), "windows": len(best)}

    cands = [best[k] for k in sorted(best)]
    med = float(np.median([c[0] for c in cands]))
    sharp = [c for c in cands if c[0] >= BLUR_RATIO * med]
    info["dropped_blurry"] = len(cands) - len(sharp)

    distinct, last = [], None
    for c in sharp:
        t = thumb(c[2])
        if last is not None and float(np.mean(cv2.absdiff(t, last))) < DUPLICATE_DIFF:
            continue  # camera barely moved - same view, no new information
        distinct.append(c)
        last = t
    info["dropped_duplicates"] = len(sharp) - len(distinct)

    if len(distinct) > max_frames:  # keep an even spread across the whole video
        keep = np.linspace(0, len(distinct) - 1, max_frames).round().astype(int)
        distinct = [distinct[i] for i in sorted(set(keep))]

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for n, (sh, fidx, frame) in enumerate(distinct, 1):
        t = fidx / fps
        p = frames_dir / f"frame_{n:02d}_t{t:05.1f}s.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append({"frame": n, "time_s": round(t, 2), "sharpness": round(sh, 1), "path": str(p)})
    info["keyframes"] = len(out)
    return out, info


# ----------------------------------------------------------------------------- #
# Licence plate (detect + optional OCR, never faked)
# ----------------------------------------------------------------------------- #
def read_plate(parts_det: PartDetector, frames: List[Dict], out_dir: Path) -> Dict:
    best = None
    for f in frames:
        img = Image.open(f["path"]).convert("RGB")
        for d in parts_det.find(img, ["license plate"]):
            if best is None or d["confidence"] > best[0]["confidence"]:
                best = (d, img, f)
    if best is None:
        return {"found": False, "text": None, "status": "no plate visible - ask customer for a plate photo"}
    d, img, f = best
    path = out_dir / "plate.jpg"
    crop(img, d["bbox"], 0.10).save(path, quality=95)
    result = {"found": True, "frame": f["frame"], "time_s": f["time_s"],
              "confidence": d["confidence"], "crop_path": str(path), "text": None}
    try:
        import easyocr  # optional: pip install easyocr
        reader = easyocr.Reader(["en"], verbose=False)
        hits = reader.readtext(str(path))
        if hits:
            text = "".join(h[1] for h in sorted(hits, key=lambda h: h[0][0][0]))
            text = "".join(ch for ch in text.upper() if ch.isalnum())
            conf = float(np.mean([h[2] for h in hits]))
            result.update(text=text, ocr_confidence=round(conf, 3),
                          status="read by OCR - match against policy record" if conf >= 0.5
                          else "OCR unsure - human to confirm")
        else:
            result["status"] = "plate found but unreadable - human to confirm"
    except ImportError:
        result["status"] = "plate cropped; OCR not installed (pip install easyocr) - human to confirm"
    return result


# ----------------------------------------------------------------------------- #
# Multi-frame consensus
# ----------------------------------------------------------------------------- #
def aggregate(frame_results: List[Dict]) -> Dict:
    groups: Dict[Tuple[str, str], Dict] = {}
    part_visible: Dict[str, set] = {}
    for fr in frame_results:
        for p in fr["parts"]:
            part_visible.setdefault(p["label"], set()).add(fr["frame"])
        for m in fr["fused"]["matched"]:
            g = groups.setdefault((m["part"], m["damage"]), {
                "part": m["part"], "damage": m["damage"], "frames": [], "detections": []})
            if fr["frame"] not in g["frames"]:
                g["frames"].append(fr["frame"])
            g["detections"].append({"frame": fr["frame"], "time_s": fr["time_s"],
                                    "confidence": m["damage_confidence"],
                                    "part_uncertain": m["part_uncertain"],
                                    "severity_hint": m.get("severity_hint"),
                                    "severity": m.get("severity"),
                                    "severity_reasons": m.get("severity_reasons"),
                                    "part_bbox": m.get("part_bbox"),
                                    "mask_area_px": m.get("mask_area_px"),
                                    "part_coverage": m.get("part_coverage"),
                                    "crop_path": m.get("crop_path"),
                                    "frame_path": fr["path"], "bbox": m["damage_bbox"]})

    damages = []
    for g in groups.values():
        n_seen = len(g["frames"])
        n_visible = len(part_visible.get(g["part"], set()))
        best = max(g["detections"], key=lambda d: d["confidence"])
        if n_seen >= 2:
            status, why = "confirmed", f"seen in {n_seen} frames"
        elif n_visible >= VISIBLE_FOR_RECHECK:
            status, why = "inconsistent", (f"seen in 1 frame but the {g['part']} was visible in "
                                           f"{n_visible} frames - likely glare/reflection")
        elif best["confidence"] >= MIN_CONF_SINGLE:
            status, why = "single_view", (f"only 1 frame shows the {g['part']} clearly; "
                                          f"confidence {best['confidence']:.2f}")
        else:
            status, why = "weak", f"1 frame, low confidence {best['confidence']:.2f}"
        damages.append({
            "part": g["part"], "damage": g["damage"], "status": status, "reason": why,
            "frames_seen": sorted(g["frames"]), "part_visible_in_frames": n_visible,
            "best_confidence": best["confidence"],
            "mean_confidence": round(float(np.mean([d["confidence"] for d in g["detections"]])), 4),
            "part_uncertain": all(d["part_uncertain"] for d in g["detections"]),
            "best_evidence": best,
        })
    order = {"confirmed": 0, "single_view": 1, "inconsistent": 2, "weak": 3}
    damages.sort(key=lambda d: (order[d["status"]], -d["best_confidence"]))

    accepted = [d for d in damages if d["status"] in ("confirmed", "single_view")]
    claims: Dict[str, Dict] = {}
    for d in accepted:
        c = claims.setdefault(d["part"], {"part": d["part"], "damages": [], "part_uncertain": False})
        c["damages"].append(d["damage"])
        c["part_uncertain"] |= d["part_uncertain"]
    for c in claims.values():
        c["damages"] = sorted(set(c["damages"]), key=lambda x: -SEVERITY_RANK.get(x, 0))
        c["primary_damage"] = c["damages"][0]

    review = [d for d in damages if d["status"] in ("inconsistent", "weak")]
    vlm = [d for d in damages if d["status"] == "single_view" or d["part_uncertain"]]
    return {"damages": damages, "claims": list(claims.values()),
            "needs_vlm": [f"{d['part']}-{d['damage']}" for d in vlm],
            "needs_review": [f"{d['part']}-{d['damage']}" for d in review],
            "escalate": not accepted,
            "escalate_reason": None if accepted else "no damage confirmed across the video"}


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def analyze_video(video_path: str, parts_det, damage_det, out_root: Path, sample_fps: float,
                  max_frames: int, pad: float, plate: bool = True, vlm=None,
                  vehicle_type: str = "auto", expected_type: Optional[str] = None) -> Dict:
    t0 = time.perf_counter()
    out = out_root / Path(video_path).stem
    out.mkdir(parents=True, exist_ok=True)

    frames, info = extract_keyframes(video_path, out, sample_fps, max_frames)
    print(f"Video {info['duration_s']}s @ {info['fps']}fps -> {info['keyframes']} keyframes "
          f"(dropped {info['dropped_blurry']} blurry, {info['dropped_duplicates']} duplicate views)")

    # decide car / motorcycle / scooter ONCE (middle keyframe), so frames don't flip
    vt, type_info = norm_type(vehicle_type), None
    if frames and vt is None:
        mid = load_image(frames[len(frames) // 2]["path"])
        vt, type_info, _ = decide_vehicle_type(mid, parts_det, "auto", expected_type)
    vt = vt or "car"
    print(f"Vehicle type: {vt}" + (f" (auto, conf {type_info['confidence']})"
                                   if type_info and type_info.get("vehicle_type") else ""))

    frame_results = []
    rules = SeverityAssessor(None)   # cheap area rules on every frame
    for f in frames:
        row, fused, parts_res = run_one(Path(f["path"]), parts_det, damage_det, out, pad, details=True,
                                        assessor=rules, vehicle_type=vt)
        frame_results.append({**f, "fused": fused, "parts": parts_res["parts"]})
        print(f"  frame {f['frame']:>2} t={f['time_s']:>5.1f}s  {row['claims']}")

    agg = aggregate(frame_results)
    for i, d in enumerate(agg["damages"], 1):  # one best crop per damage, for the VLM
        if d["status"] in ("confirmed", "single_view", "inconsistent"):
            ev = d["best_evidence"]
            img = Image.open(ev["frame_path"]).convert("RGB")
            p = out / f"best_{i}_{d['part'].replace(' ', '_')}-{d['damage'].replace(' ', '_')}.jpg"
            crop(img, ev["bbox"], pad).save(p, quality=95)
            d["best_crop"] = str(p)
            if vlm is not None:   # VLM only on the best frame of each damage (saves time)
                sev = SeverityAssessor(vlm).assess(img, {
                    "part": d["part"], "damage": d["damage"], "damage_bbox": ev["bbox"],
                    "damage_confidence": ev.get("confidence"),
                    "part_bbox": ev.get("part_bbox"), "mask_area_px": ev.get("mask_area_px"),
                    "part_coverage": ev.get("part_coverage")})
                d.update(sev)

    plate_info = read_plate(parts_det, frames, out) if plate else {"status": "skipped"}

    report = {"video": str(video_path), "video_info": info, "keyframes": frames,
              "vehicle_type": vt, "vehicle_type_detection": type_info,
              "plate": plate_info, **agg, "total_s": round(time.perf_counter() - t0, 2)}
    (out / "video_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Auto-Settle AI - video claim analysis")
    ap.add_argument("input", help="video file (.mp4 / .mov / .avi ...)")
    ap.add_argument("--weights", help="CarDD .pt file (default: auto-download CarDD v2.0)")
    ap.add_argument("--dino", choices=["tiny", "base"], default="base")
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--fps", type=float, default=2.0, help="keyframe windows per second")
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--pad", type=float, default=0.30)
    ap.add_argument("--no-plate", action="store_true", help="skip licence plate detection")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--severity", choices=["rules", "vlm"], default="rules")
    ap.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    from run_pipeline import add_vehicle_args
    add_vehicle_args(ap)
    args = ap.parse_args(argv)

    if not Path(args.input).is_file():
        print(f"[ERROR] video not found: {args.input}", file=sys.stderr)
        return 1
    try:
        print("Loading models...")
        parts_det = PartDetector(model_size=args.dino)
        damage_det = DamageDetector(args.weights, args.yolo_conf)
        vlm = load_vlm("qwen" if args.severity == "vlm" else "none", args.vlm_model)
        pre, _ = pricing_rules.resolve_vehicle(   # policy known up front? use its type
            args.make, args.model, args.year, args.cc, args.zero_dep, args.plate,
            vehicle_type=None if args.vehicle_type == "auto" else args.vehicle_type)
        rep = analyze_video(args.input, parts_det, damage_det, Path(args.out_dir), args.fps,
                            args.max_frames, args.pad, plate=not args.no_plate, vlm=vlm,
                            vehicle_type=args.vehicle_type,
                            expected_type=(pre or {}).get("vehicle_type"))
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("\n========== VIDEO CLAIM REPORT ==========")
    for d in rep["damages"]:
        print(f"  [{d['status'].upper():<12}] {d['part']}-{d['damage']:<14} "
              f"frames {d['frames_seen']}  best conf {d['best_confidence']:.2f}  - {d['reason']}")
        sev = d.get("severity") or d["best_evidence"].get("severity")
        if sev or d.get("vlm_confirmed") is False:
            print(f"{'':<17}severity: {sev or 'VLM: not real damage'}"
                  f"{'  (review)' if d.get('severity_review') else ''}  "
                  f"{'; '.join(d.get('severity_reasons') or d['best_evidence'].get('severity_reasons') or [])}")
    print("\nClaims:")
    for c in rep["claims"]:
        print(f"  {c['part']:<14} {', '.join(c['damages'])}  (primary: {c['primary_damage']})")
    if rep["needs_vlm"]:
        print(f"VLM to confirm : {', '.join(rep['needs_vlm'])}")
    if rep["needs_review"]:
        print(f"Human review   : {', '.join(rep['needs_review'])}")
    print(f"Licence plate  : {rep['plate'].get('text') or '-'}  ({rep['plate']['status']})")
    if rep["escalate"]:
        print(f"ESCALATE       : {rep['escalate_reason']}")
    # ---- pricing: CLI vehicle details, else the plate read from the video
    plate_text = args.plate or rep["plate"].get("text")
    vehicle, suspicious = pricing_rules.resolve_vehicle(
        args.make, args.model, args.year, args.cc, args.zero_dep, plate_text,
        vehicle_type=None if args.vehicle_type == "auto" else args.vehicle_type)
    out = Path(args.out_dir) / Path(args.input).stem
    if vehicle is not None or suspicious:
        v = dict(vehicle or {})
        expected = v.get("vehicle_type")
        ti = rep.get("vehicle_type_detection") or {}
        if (expected and ti.get("vehicle_type") and (ti.get("confidence") or 0) >= MISMATCH_CONF
                and family(expected) != family(ti["vehicle_type"])):
            suspicious = list(suspicious) + [f"video shows a {ti['vehicle_type']} but the policy is "
                                             f"for a {norm_type(expected)}"]
        v["vehicle_type"] = expected or rep["vehicle_type"]
        q = pricing_rules.price_claim(pricing_rules.items_from_video(rep), v, suspicious)
        (out / "quote.json").write_text(json.dumps(q, indent=2))
        pricing_rules.print_quote(q)
    else:
        print("\n(No plate read and no --make/--plate given: damage only, no pricing.)")
    print(f"Done in {rep['total_s']}s -> {out / 'video_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
