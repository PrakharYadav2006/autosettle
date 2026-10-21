"""
Auto-Settle AI — Stage 5: Deterministic pricing, settlement range & guardrails
=============================================================================

Takes the damage found by the vision pipeline (photo: fused.json, video:
video_report.json) plus the policy/vehicle details, and returns ONE claim-level
decision:

    APPROVED         -> cash settlement range ready (opening offer .. maximum)
    NO_PAYOUT        -> damage is real but the payable amount is <= deductible
    REVIEW_REQUIRED  -> estimate shown, but a human must sign off
    ESCALATED        -> no automatic decision possible (reasons listed)

Python decides every number. The chatbot (Stage 6) may only *phrase* offers that
come out of this module and the negotiation state machine.

DEMO ASSUMPTIONS (state these in the pitch / assumptions doc)
------------------------------------------------------------
* All part / labour / paint prices are MOCK illustrative INR values.
* Depreciation on parts follows the pattern commonly used in Indian private-car
  own-damage policies (plastic/rubber 50 %, glass 0 %, metal by vehicle age),
  waived with a zero-depreciation add-on. Labour is never depreciated.
  Treat the exact percentages as configurable - verify before production.
* Compulsory deductible: cars Rs 1,000 (<= 1500 cc) / Rs 2,000 (> 1500 cc);
  two-wheelers Rs 100 (< 150 cc) / Rs 500 (>= 150 cc) - configurable.
* Two-wheelers (motorcycle / scooter) use their own price tier and part list.
* Cash-in-lieu range = 80 % .. 95 % of the insurer's payable amount. The customer
  takes less than a garage bill in exchange for instant money and no garage visit;
  the insurer saves surveyor and garage-handling cost. Ratios are configurable.
* Auto-approval only if the garage estimate is <= Rs 20,000 (demo threshold, not
  a statutory limit). Above it -> surveyor inspection.
* Metal panels with high severity, structural panels, glass with ADAS (windshield)
  and tyres always go to a human: hidden damage cannot be priced from a photo.

Usage
-----
    python pricing_rules.py results\\car\\fused.json --make maruti --model swift --year 2021
    python pricing_rules.py results\\claim\\video_report.json --plate DL3CAB1234
    python pricing_rules.py --demo                       # built-in test cases
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from datetime import date, datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from vehicles import TWO_WHEELERS, norm_type, type_from_model

# =============================================================================
# CONFIGURATION (all demo values - change here, nowhere else)
# =============================================================================
AUTO_APPROVAL_THRESHOLD_INR = 20_000
OPEN_OFFER_RATIO = Decimal("0.80")   # first offer = 80 % of payable amount
MAX_OFFER_RATIO = Decimal("0.95")    # never offer more than 95 % of payable amount
MIN_CONF_VLM = 0.60                  # VLM-confirmed damage
MIN_CONF_DETECTOR = 0.50             # detector-only damage (no VLM yet)
MIN_CONF_LAMP = 0.40                 # lamp/glass classes score lower in CarDD
REQUIRE_VLM_CONFIRMATION = False     # api_server turns this ON whenever the VLM is loaded (fail-closed)
DEDUCTIBLE_SMALL, DEDUCTIBLE_LARGE, CC_LIMIT = 1_000, 2_000, 1500
# Two-wheeler compulsory deductible.  Rs 100 is the commonly cited figure; the
# >=150 cc slab of Rs 500 is from our team spec and NOT yet verified against a
# policy wording - change here if the insurer's wording says otherwise.
TW_DEDUCTIBLE_SMALL, TW_DEDUCTIBLE_LARGE, TW_CC_LIMIT = 100, 500, 150

# Price tiers (mock). part = new part price; labour/paint = per panel.
TIERS: Dict[str, Dict[str, Any]] = {
    "economy": {"labour": 1200, "paint": 1800, "parts": {
        "front_bumper": 2800, "rear_bumper": 2600, "car_door": 4500, "fender": 2200,
        "car_hood": 4200, "quarter_panel": 5000, "side_skirt": 1800, "grille": 1200,
        "headlight": 3200, "taillight": 2800, "fog_light": 900, "side_mirror": 1400,
        "windshield": 6500}},
    "mid": {"labour": 1500, "paint": 2200, "parts": {
        "front_bumper": 4300, "rear_bumper": 4000, "car_door": 6600, "fender": 3200,
        "car_hood": 6200, "quarter_panel": 7500, "side_skirt": 2600, "grille": 2200,
        "headlight": 5600, "taillight": 4600, "fog_light": 1500, "side_mirror": 2300,
        "windshield": 9800}},
    "luxury": {"labour": 4500, "paint": 6500, "parts": {
        "front_bumper": 28000, "rear_bumper": 26000, "car_door": 42000, "fender": 18000,
        "car_hood": 38000, "quarter_panel": 45000, "side_skirt": 12000, "grille": 15000,
        "headlight": 48000, "taillight": 32000, "fog_light": 9000, "side_mirror": 16000,
        "windshield": 35000}},
    # Two-wheelers (mock demo rates, benchmark: Hero Splendor Plus / Honda Activa 6G)
    "two_wheeler": {"labour": 400, "paint": 800, "parts": {
        "fuel_tank": 2200, "front_mudguard": 450, "front_apron": 1750, "side_body_panel": 850,
        "floorboard": 600, "headlight": 650, "taillight": 450, "side_mirror": 180,
        "handlebar": 550, "exhaust_pipe": 1400}},
}
# two-wheeler make -> price multiplier inside the two_wheeler tier
# (separate table because "honda" / "suzuki" make both cars and scooters)
TW_MAKES = {"hero": 1.00, "honda": 1.05, "tvs": 1.00, "bajaj": 1.02, "suzuki": 1.05,
            "yamaha": 1.08, "royal enfield": 1.25}
# make -> (tier, price multiplier inside the tier)
MAKES = {
    "maruti": ("economy", 1.00), "tata": ("economy", 1.07), "renault": ("economy", 1.04),
    "nissan": ("economy", 1.05), "hyundai": ("mid", 0.98), "kia": ("mid", 1.00),
    "honda": ("mid", 1.05), "mahindra": ("mid", 1.08), "toyota": ("mid", 1.12),
    "skoda": ("mid", 1.10), "volkswagen": ("mid", 1.10), "mg": ("mid", 1.04),
    "bmw": ("luxury", 1.00), "mercedes": ("luxury", 1.04), "audi": ("luxury", 0.98),
}
MAKE_ALIASES = {"maruti suzuki": "maruti", "suzuki": "maruti", "mercedes benz": "mercedes",
                "mercedes-benz": "mercedes", "benz": "mercedes", "vw": "volkswagen",
                "morris garages": "mg", "hero motocorp": "hero", "honda moto": "honda",
                "honda_moto": "honda", "hmsi": "honda", "tvs motor": "tvs", "re": "royal enfield"}

# Part properties: material (for depreciation), paintable, metal body panel?
PARTS = {
    "front_bumper": ("plastic", True), "rear_bumper": ("plastic", True),
    "side_skirt": ("plastic", True), "grille": ("plastic", False),
    "car_door": ("metal", True), "fender": ("metal", True), "car_hood": ("metal", True),
    "quarter_panel": ("metal", True),
    "headlight": ("plastic", False), "taillight": ("plastic", False),  # lamp housings
    "fog_light": ("plastic", False), "side_mirror": ("plastic", False),
    "windshield": ("glass", False),
    # two-wheelers
    "fuel_tank": ("metal", True), "front_mudguard": ("plastic", True),
    "front_apron": ("plastic", True), "side_body_panel": ("plastic", True),
    "floorboard": ("plastic", False), "exhaust_pipe": ("metal", False),
    "handlebar": ("metal", False),
}
LAMP_GLASS = {"headlight", "taillight", "fog_light", "side_mirror", "windshield"}
ALWAYS_INSPECT = {
    "windshield": "windshield replacement needs ADAS-camera / safety checks",
    "quarter_panel": "quarter panel is welded structure - replacement needs inspection",
    "wheel": "tyre/wheel damage needs inspection (often not covered on its own)",
    "side_window": "side-window glass damage needs inspection (often a reflection in photos)",
    "trunk": "boot lid damage needs inspection",
    "handlebar": "handlebar / fork damage affects steering geometry - mandatory physical inspection",
    "exhaust_pipe": "bent exhaust may mean engine / manifold stress - surveyor inspection",
}
PART_ALIASES = {
    "front bumper": "front_bumper", "rear bumper": "rear_bumper", "car door": "car_door",
    "door": "car_door", "front door": "car_door", "rear door": "car_door",
    "fender": "fender", "car hood": "car_hood", "hood": "car_hood", "bonnet": "car_hood",
    "quarter panel": "quarter_panel", "side skirt": "side_skirt", "rocker panel": "side_skirt",
    "grille": "grille", "headlight": "headlight", "headlamp": "headlight",
    "taillight": "taillight", "tail light": "taillight", "tail lamp": "taillight",
    "fog light": "fog_light", "fog lamp": "fog_light", "side mirror": "side_mirror",
    "trunk": "trunk", "boot": "trunk", "boot lid": "trunk",
    "wing mirror": "side_mirror", "windshield": "windshield", "windscreen": "windshield",
    "wheel": "wheel", "tyre": "wheel", "tire": "wheel",
    "side window": "side_window", "rear windshield": "windshield", "rear window": "windshield",
    # two-wheelers (DINO labels -> price keys)
    "fuel tank": "fuel_tank", "motorcycle fuel tank": "fuel_tank", "petrol tank": "fuel_tank",
    "front mudguard": "front_mudguard", "mudguard": "front_mudguard", "front fender": "front_mudguard",
    "front apron": "front_apron", "apron": "front_apron",
    "side panel": "side_body_panel", "side body panel": "side_body_panel", "side cowl": "side_body_panel",
    "floorboard": "floorboard", "floor board": "floorboard",
    "rear view mirror": "side_mirror", "mirror": "side_mirror",
    "handlebar": "handlebar", "handle bar": "handlebar",
    "exhaust pipe": "exhaust_pipe", "silencer": "exhaust_pipe", "exhaust": "exhaust_pipe",
}
# CarDD / VLM damage labels -> our canonical set
DAMAGE_ALIASES = {
    "scratch": "scratch", "scratches": "scratch", "dent": "dent", "dents": "dent",
    "crack": "crack", "cracked": "crack", "lamp broken": "broken", "broken": "broken",
    "glass shatter": "shattered", "shattered": "shattered", "shatter": "shattered",
    "tire flat": "tyre_flat", "tyre flat": "tyre_flat", "flat tire": "tyre_flat",
    "deformation": "dent", "bent": "dent", "paint damage": "scratch",
}
SEVERITY_ALIASES = {"minor": "low", "low": "low", "moderate": "medium", "medium": "medium",
                    "severe": "high", "high": "high"}

# Metal-part depreciation by vehicle age (months). Plastic 50 %, glass 0 %.
METAL_DEP = [(6, 0), (12, 5), (24, 10), (36, 15), (48, 25), (60, 35), (120, 40), (10**6, 50)]
MATERIAL_DEP = {"plastic": 50, "glass": 0}

# Repair methods: labour multiplier, paint multiplier, uses new part?
METHODS = {
    "TOUCH_UP":         (0.5, 0.5, False),   # light scratch, polish + touch-up
    "REPAINT":          (0.5, 1.0, False),   # deep scratch, panel repaint
    "DENT_REPAIR":      (1.5, 1.0, False),   # denting + repaint
    "PDR":              (1.0, 0.0, False),   # small dent, paintless dent repair
    "PLASTIC_WELD":     (1.5, 1.0, False),   # small crack on a plastic panel
    "REPLACE":          (1.0, 1.0, True),    # new part + fitting (+ paint if paintable)
}


# =============================================================================
# REAL PRICE LAYER (from price_scraper.py -> external_prices.csv)
# =============================================================================
EXTERNAL_PRICES_CSV = Path(__file__).with_name("external_prices.csv")
STALE_AFTER_DAYS = 30
UNVERIFIED_FITMENT = re.compile(r"infer|verify|unverified|guess", re.I)
_EXTERNAL: Optional[Dict] = None


def load_external_prices(path: Path = EXTERNAL_PRICES_CSV) -> Dict:
    """(make, model, part) -> list of price rows. Missing/broken file -> {} (mock prices used)."""
    table: Dict = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    price = int(float(r["price_inr"]))
                except (KeyError, ValueError):
                    continue
                if price <= 0:
                    continue
                key = (norm(r.get("make")), norm(r.get("model")), norm(r.get("part")).replace(" ", "_"))
                table.setdefault(key, []).append({**r, "price_inr": price})
    except OSError:
        pass
    return table


def real_part_price(make: str, model: Optional[str], part: str) -> Optional[Dict]:
    """Median of the scraped prices for this exact make+model+part, with its sources."""
    global _EXTERNAL
    if _EXTERNAL is None:
        _EXTERNAL = load_external_prices()
    all_rows = _EXTERNAL.get((norm(make), norm(model), part))
    if not all_rows:
        return None
    # A row whose fitment the scraper could only infer ("... - verify") is not used
    # automatically: a wrong-model part price must not reach a quote unreviewed.
    rows = [r for r in all_rows if not UNVERIFIED_FITMENT.search(r.get("fitment") or "")]
    if not rows:
        return {"unverified_only": True, "urls": [r.get("source_url", "") for r in all_rows]}
    values = sorted(r["price_inr"] for r in rows)
    newest = max(r.get("scraped_at", "") for r in rows)
    stale = False
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(newest)
        stale = age.days > STALE_AFTER_DAYS
    except ValueError:
        stale = True
    return {"price": money(statistics.median(values)), "values": values, "n": len(values),
            "sites": sorted({r.get("source_site", "") for r in rows}),
            "urls": [r.get("source_url", "") for r in rows],
            "price_type": rows[0].get("price_type", ""), "scraped_at": newest, "stale": stale}


# =============================================================================
# Helpers
# =============================================================================
def money(x) -> int:
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def norm(text: Optional[str]) -> str:
    return " ".join(str(text or "").strip().lower().replace("_", " ").replace("-", " ").split())


def vehicle_age_months(year: Optional[int], month: int = 6) -> Optional[int]:
    if not isinstance(year, int) or isinstance(year, bool):
        return None
    today = date.today()
    return max(0, (today.year - year) * 12 + today.month - month)


def part_depreciation_pct(material: str, age_months: Optional[int], zero_dep: bool) -> int:
    if zero_dep:
        return 0
    if material == "metal":
        if age_months is None:
            return 50  # unknown age -> most conservative
        return next(p for limit, p in METAL_DEP if age_months <= limit)
    return MATERIAL_DEP.get(material, 50)


def choose_method(part: str, damage: str, severity: str) -> Dict[str, Any]:
    """Repair decision for ONE damage on ONE part. Returns method or an inspect reason."""
    material, _ = PARTS[part]
    if damage in ("broken", "shattered"):
        if part in LAMP_GLASS or material == "plastic":
            return {"method": "REPLACE"}
        return {"inspect": f"'{damage}' on a metal panel - structural damage needs inspection"}
    if damage == "crack":
        if part in LAMP_GLASS:
            return {"method": "REPLACE"}
        if material == "plastic":
            return {"method": "PLASTIC_WELD" if severity == "low" else "REPLACE"}
        return {"inspect": "crack in a metal panel can mean structural damage"}
    if damage == "dent":
        if part in LAMP_GLASS:
            return {"method": "REPLACE"}
        if severity == "high":
            if material == "plastic":
                return {"method": "REPLACE"}
            return {"inspect": "large dent in a metal panel - possible hidden damage"}
        return {"method": "PDR" if severity == "low" and material == "metal" else "DENT_REPAIR"}
    if damage == "scratch":
        if part in LAMP_GLASS:
            return {"method": "REPLACE"} if severity == "high" else {"method": "TOUCH_UP"}
        return {"method": "TOUCH_UP" if severity == "low" else "REPAINT"}
    if damage == "tyre_flat":
        return {"inspect": ALWAYS_INSPECT["wheel"]}
    return {"inspect": f"unknown damage type '{damage}'"}


def cost_line(part: str, method: str, prices: Dict[str, Any], mult: float,
              age_months: Optional[int], zero_dep: bool,
              make: str = "", model: Optional[str] = None) -> Dict[str, Any]:
    material, paintable = PARTS[part]
    lab_x, paint_x, new_part = METHODS[method]
    labour = money(prices["labour"] * mult * lab_x)
    paint = money(prices["paint"] * mult * paint_x) if paintable else 0  # lamps/glass aren't painted
    source = {"type": "none"}
    part_cost = 0
    if new_part:
        real = real_part_price(make, model, part)
        if real and not real.get("unverified_only"):  # real price for this exact model+part
            part_cost = real["price"]
            source = {"type": "scraped", **real}
        else:
            part_cost = money(prices["parts"][part] * mult)
            source = {"type": "demo_estimate", "tier_price": prices["parts"][part], "multiplier": mult,
                      **({"skipped_unverified": True, "unverified_urls": real["urls"]} if real else {})}
    dep_pct = part_depreciation_pct(material, age_months, zero_dep) if new_part else 0
    dep = money(part_cost * dep_pct / 100)
    return {"method": method, "material": material, "part_cost": part_cost, "labour": labour, "paint": paint,
            "garage_cost": part_cost + labour + paint,
            "depreciation_pct": dep_pct, "depreciation": dep, "part_price_source": source}


def deductible_for(vehicle: Dict) -> int:
    """Compulsory deductible from vehicle type + engine cc."""
    vtype = norm_type(vehicle.get("vehicle_type")) or type_from_model(vehicle.get("model")) or "car"
    cc = vehicle.get("engine_cc")
    if vtype in TWO_WHEELERS:
        return TW_DEDUCTIBLE_LARGE if isinstance(cc, (int, float)) and cc >= TW_CC_LIMIT else TW_DEDUCTIBLE_SMALL
    return DEDUCTIBLE_LARGE if isinstance(cc, (int, float)) and cc > CC_LIMIT else DEDUCTIBLE_SMALL


# =============================================================================
# Input adapters: our pipeline outputs -> line items
# =============================================================================
def _region(frame: Optional[str], bbox) -> Optional[Dict]:
    """Where a finding was seen: frame/photo id + box in that image's pixels."""
    if not bbox or len(bbox) != 4:
        return None
    return {"frame": frame or "photo", "bbox": [round(float(v), 1) for v in bbox]}


def _verification(d: Dict) -> str:
    """confirmed / rejected / inconclusive / failed / not_run (older results: derived)."""
    v = d.get("vlm_verification")
    if v:
        return v
    if d.get("vlm_confirmed") is True:
        return "confirmed"
    if d.get("vlm_confirmed") is False:
        return "rejected"
    return "not_run"


def items_from_fused(fused: Dict) -> List[Dict]:
    """Photo mode (fused.json). One item per detected damage."""
    out = []
    for m in fused.get("matched", []):
        confirmed = m.get("vlm_confirmed")
        out.append({"part": m["part"], "damage": m["damage"],
                    # once the VLM has voted, confidence = share of VALID views that agree
                    "confidence": m["vlm_agreement"] if confirmed and "vlm_agreement" in m
                    else m["damage_confidence"],
                    "detector_confidence": m["damage_confidence"],
                    "severity": m.get("vlm_severity") or m.get("severity") or m.get("severity_hint"),
                    "severity_source": m.get("severity_source") or "detector_heuristic",
                    "severity_reasons": m.get("severity_reasons"),
                    "severity_review": m.get("severity_review", False),
                    "vlm_confirmed": confirmed, "vlm_verification": _verification(m),
                    "vlm_valid_views": m.get("vlm_valid_views"),
                    "part_uncertain": m.get("part_uncertain", False),
                    "region": _region("photo", m.get("damage_bbox")),
                    "evidence": m.get("crop_path")})
    if fused.get("image_quality_warning"):
        out.append({"part": "photo", "damage": "quality", "quality_issue": fused["image_quality_warning"]})
    for ins in fused.get("inspect", []):   # lamp/glass the VLM thinks YOLO missed
        if ins.get("vlm_found_damage"):
            out.append({"part": ins["part"], "damage": ins.get("vlm_damage", "broken"),
                        "severity": ins.get("vlm_severity"), "confidence": None,
                        "vlm_only": True, "region": _region("photo", ins.get("bbox")),
                        "evidence": ins.get("crop_path")})
    return out


def items_from_video(report: Dict) -> List[Dict]:
    """Video mode (video_report.json).  Detections of the same part+damage across frames are
    already ONE group (video_pipeline.aggregate), so each group is one item here."""
    out = []
    for d in report.get("damages", []):
        ev = d.get("best_evidence", {})
        confirmed = d.get("vlm_confirmed")
        out.append({"part": d["part"], "damage": d["damage"],
                    "confidence": d["vlm_agreement"] if confirmed and "vlm_agreement" in d
                    else d["best_confidence"],
                    "detector_confidence": d["best_confidence"],
                    "severity": d.get("vlm_severity") or d.get("severity") or ev.get("severity")
                    or ev.get("severity_hint"),
                    "severity_source": d.get("severity_source") or ("rules_area" if ev.get("severity")
                                                                    else "detector_heuristic"),
                    "severity_reasons": d.get("severity_reasons") or ev.get("severity_reasons"),
                    "severity_review": d.get("severity_review", False),
                    "vlm_confirmed": confirmed, "vlm_verification": _verification(d),
                    "vlm_valid_views": d.get("vlm_valid_views"),
                    "part_uncertain": d.get("part_uncertain", False),
                    "video_status": d["status"], "frames_seen": d.get("frames_seen"),
                    "region": _region(ev.get("frame_path"), ev.get("bbox")),
                    "evidence": d.get("best_crop")})
    return out


# =============================================================================
# Overlap check: one damage region must never be charged on two panels
# =============================================================================
# Two damage boxes in the SAME image whose overlap is at least this share of the SMALLER box
# are treated as the same physical damage.  Boxes that merely touch (small overlap) stay
# separate damages.  Same-part overlaps need no grouping: one repair per part already covers them.
OVERLAP_MIN_SHARE = 0.50


def _overlap_share(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def _ruled_out(it: Dict) -> bool:
    """Already rejected by the evidence - cannot take part in an overlap group."""
    return (it.get("video_status") == "inconsistent" or _verification(it) == "rejected"
            or bool(it.get("quality_issue")))


def overlap_groups(items: List[Dict]) -> List[List[int]]:
    """Indices of items that are one damage region on DIFFERENT canonical parts."""
    n, parent = len(items), list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if _ruled_out(items[i]) or _ruled_out(items[j]):
                continue
            a, b = items[i].get("region"), items[j].get("region")
            if not a or not b or a["frame"] != b["frame"]:
                continue
            pa, pb = PART_ALIASES.get(norm(items[i]["part"])), PART_ALIASES.get(norm(items[j]["part"]))
            if pa == pb:
                continue
            if _overlap_share(a["bbox"], b["bbox"]) >= OVERLAP_MIN_SHARE:
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


# =============================================================================
# The engine
# =============================================================================
# Per-finding status (the same names are used by the API and the frontend):
#   confirmed_priceable  priced in the estimate
#   uncertain_review     real damage possible, evidence not strong enough -> human, not priced
#   unsupported_review   cannot be priced automatically (location unclear, inspection part,
#                        structural) -> surveyor, not priced
#   rejected             the evidence says it is not damage (VLM majority / video) -> not priced
#   duplicate_or_overlap one damage region mapped to several panels -> human, not priced
FINDING_STATUSES = ("confirmed_priceable", "uncertain_review", "unsupported_review", "rejected",
                    "duplicate_or_overlap")


def price_claim(items: List[Dict], vehicle: Dict, suspicious: Optional[List[str]] = None) -> Dict:
    """vehicle: {make, model, year, engine_cc, zero_dep, policy_active}"""
    reasons_escalate: List[str] = []
    reasons_review: List[str] = []
    blockers: List[Dict] = []        # structured: {code, text, level, finding?}
    notes: List[str] = []
    findings: List[Dict] = []

    def block(level: str, code: str, text: str, finding: Optional[int] = None) -> None:
        (reasons_escalate if level == "escalate" else reasons_review).append(text)
        blockers.append({"code": code, "level": level, "text": text,
                         **({"finding": finding} if finding is not None else {})})

    # ---- vehicle / policy -------------------------------------------------
    make = MAKE_ALIASES.get(norm(vehicle.get("make")), norm(vehicle.get("make")))
    if not make:
        why = [f"suspicious: {s}" for s in suspicious or []] or ["vehicle make unknown - cannot choose a price list"]
        return _decision("ESCALATED", why, [], {}, vehicle, notes,
                         blockers=[{"code": "SUSPICIOUS" if suspicious else "MAKE_UNKNOWN",
                                    "level": "escalate", "text": w} for w in why])
    if vehicle.get("policy_active") is False:
        return _decision("ESCALATED", ["policy not active"], [], {}, vehicle, notes,
                         blockers=[{"code": "POLICY_INACTIVE", "level": "escalate", "text": "policy not active"}])
    vtype = norm_type(vehicle.get("vehicle_type")) or type_from_model(vehicle.get("model")) or "car"
    two_wheeler = vtype in TWO_WHEELERS
    if two_wheeler:
        tier, mult = "two_wheeler", TW_MAKES.get(make, 1.0)
        if make not in TW_MAKES:
            block("review", "MAKE_NOT_IN_PRICE_LIST",
                  f"two-wheeler make '{make}' not in price list - base two-wheeler prices used")
    elif make in MAKES:
        tier, mult = MAKES[make]
    else:
        tier, mult = "mid", 1.0
        block("review", "MAKE_NOT_IN_PRICE_LIST", f"make '{make}' not in price list - generic mid-tier prices used")
    prices = TIERS[tier]
    year = vehicle.get("year")
    age = vehicle_age_months(year)
    if not vehicle.get("model") or age is None:
        block("review", "VEHICLE_UNVERIFIED", "model / year missing - exact vehicle not verified")
    zero_dep = bool(vehicle.get("zero_dep"))
    for s in suspicious or []:
        block("escalate", "SUSPICIOUS", f"suspicious: {s}")

    # ---- overlap groups: decided BEFORE pricing so nothing is charged twice --------
    in_group: Dict[int, int] = {}
    groups = overlap_groups(items)
    for gi, g in enumerate(groups):
        for i in g:
            in_group[i] = gi

    # ---- per item: validate, decide status + method --------------------------
    per_part: Dict[str, List[Dict]] = {}

    def finding(i, it, status, code=None, reason=None, **extra):
        findings.append({"id": i, "part": it.get("part"), "price_key": PART_ALIASES.get(norm(it.get("part"))),
                         "damage": it.get("damage"), "status": status, "reason_code": code, "reason": reason,
                         "severity": SEVERITY_ALIASES.get(norm(it.get("severity"))) or None,
                         "verification": it.get("vlm_verification") or ("vlm_only" if it.get("vlm_only") else None),
                         "vlm_valid_views": it.get("vlm_valid_views"),
                         "video_status": it.get("video_status"), "frames_seen": it.get("frames_seen"),
                         "region": it.get("region"), "evidence": it.get("evidence"), **extra})

    for i, it in enumerate(items):
        raw = f"{it['part']}-{it['damage']}"
        part = PART_ALIASES.get(norm(it["part"]), "")
        damage = DAMAGE_ALIASES.get(norm(it["damage"]), "")
        sev = SEVERITY_ALIASES.get(norm(it.get("severity")), "")
        status = it.get("video_status")
        ver = it.get("vlm_verification") or _verification(it)

        def review(code, text):
            block("review", code, f"{raw}: {text}", i)
            finding(i, it, "uncertain_review", code, text)

        def unsupported(code, text):
            block("escalate", code, f"{raw}: {text}", i)
            finding(i, it, "unsupported_review", code, text)

        # -- evidence says: not damage --------------------------------------
        # "inconsistent" = seen in 1 frame while other frames show the same part clearly
        # with NO damage. The video itself contradicts it: noted, not priced, not blocking.
        if status == "inconsistent":
            notes.append(f"{raw}: seen in only 1 video frame, other frames show the part undamaged - not priced")
            finding(i, it, "rejected", "VIDEO_CONTRADICTS", "seen in only 1 video frame; other frames show the part undamaged")
            continue
        if ver == "rejected":
            notes.append(f"{raw}: rejected by VLM - not priced")
            finding(i, it, "rejected", "VLM_REJECTED", "most valid VLM views see no real damage")
            continue
        if it.get("quality_issue"):
            block("review", "IMAGE_QUALITY", it["quality_issue"])
            finding(i, it, "uncertain_review", "IMAGE_QUALITY", it["quality_issue"])
            continue
        if i in in_group:
            others = sorted({items[j]["part"] for j in groups[in_group[i]] if j != i})
            text = f"same damage region also mapped to {', '.join(others)} - one repair, human to assign the panel"
            block("review", "OVERLAP", f"{raw}: {text}", i)
            finding(i, it, "duplicate_or_overlap", "OVERLAP", text, group=in_group[i])
            continue
        # -- evidence too weak -> human, not priced ---------------------------
        if status == "weak":
            review("VIDEO_WEAK", "seen once in video with low confidence - not priced, human to check")
            continue
        if it.get("vlm_only"):
            review("VLM_ONLY", "VLM sees damage the detector missed - not priced, human to confirm")
            continue
        if ver == "inconclusive":
            review("VLM_INCONCLUSIVE", "VLM views split on whether the damage is real - not priced, human to check")
            continue
        if ver == "failed":
            review("VLM_FAILED", "VLM could not give a valid answer - not priced, human to check")
            continue
        # -- cannot be priced automatically -----------------------------------
        if it.get("part_uncertain"):
            unsupported("PART_LOCATION_UNCLEAR", "damaged panel could not be identified")
            continue
        if not part:
            unsupported("PART_NOT_IN_PRICE_LIST", f"part '{it['part']}' not in price list")
            continue
        if part not in prices["parts"] and part not in ALWAYS_INSPECT:
            unsupported("PART_NOT_IN_PRICE_LIST", f"no {tier.replace('_', '-')} price for '{part}'")
            continue
        if part in ALWAYS_INSPECT:
            unsupported("ALWAYS_INSPECT", ALWAYS_INSPECT[part])
            continue
        if not damage:
            unsupported("UNKNOWN_DAMAGE", f"unknown damage type '{it['damage']}'")
            continue
        if not sev:
            unsupported("SEVERITY_UNKNOWN", "severity unknown")
            continue
        conf = it.get("confidence")
        if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            unsupported("CONFIDENCE_INVALID", "confidence missing/invalid")
            continue
        floor = (MIN_CONF_VLM if ver == "confirmed" else
                 MIN_CONF_LAMP if damage in ("broken", "shattered") else MIN_CONF_DETECTOR)
        if REQUIRE_VLM_CONFIRMATION and ver != "confirmed":
            review("VLM_NOT_CONFIRMED", "not confirmed by the VLM - not priced, human to check")
            continue
        if it.get("severity_review"):
            why = "; ".join(it.get("severity_reasons") or [])
            review("SEVERITY_REVIEW", f"severity needs human check ({why}) - not priced")
            continue
        if conf < floor and status != "confirmed":
            review("LOW_CONFIDENCE", f"low confidence {conf:.2f} (< {floor}) - not priced, human to confirm")
            continue
        decision = choose_method(part, damage, sev)
        if "inspect" in decision:
            unsupported("STRUCTURAL_INSPECT", decision["inspect"])
            continue
        line = cost_line(part, decision["method"], prices, mult, age, zero_dep,
                         make=make, model=vehicle.get("model"))
        src = line["part_price_source"]
        if src.get("type") == "scraped" and src.get("stale"):
            notes.append(f"{raw}: real price older than {STALE_AFTER_DAYS} days - re-run price_scraper.py")
        if src.get("skipped_unverified"):
            notes.append(f"{raw}: scraped price with unverified fitment ignored - demo estimate used")
        line.update(part=part, damage=damage, severity=sev, severity_source=it.get("severity_source"),
                    severity_reasons=it.get("severity_reasons"), finding=i,
                    confidence=round(float(conf), 3), evidence=it.get("evidence"))
        per_part.setdefault(part, []).append(line)
        finding(i, it, "confirmed_priceable")

    # ---- one repair per part: the most expensive method covers the rest ----
    lines = []
    for part, ls in per_part.items():
        best = max(ls, key=lambda l: l["garage_cost"])
        covered = sorted({l["damage"] for l in ls if l is not best})
        if covered:
            best["also_covers"] = covered
        lines.append(best)
    lines.sort(key=lambda l: -l["garage_cost"])

    if not lines and not reasons_escalate and not reasons_review:
        return _decision("ESCALATED", ["no priceable damage found"], [], {}, vehicle, notes, findings,
                         [{"code": "NO_PRICEABLE_DAMAGE", "level": "escalate", "text": "no priceable damage found"}],
                         groups)

    pending = [f for f in findings if f["status"] in ("uncertain_review", "unsupported_review",
                                                        "duplicate_or_overlap")]
    garage = sum(l["garage_cost"] for l in lines)
    depreciation = sum(l["depreciation"] for l in lines)
    deductible = deductible_for(vehicle)
    payable = max(0, garage - depreciation - deductible)
    # complete = every damage found is either priced or ruled out; partial = something is pending
    scope = "none" if not lines else "partial" if pending else "complete"
    money_block = {
        "garage_estimate_inr": garage,
        "depreciation_inr": depreciation,
        "deductible_inr": deductible,
        "insurer_payable_inr": payable,
        "estimate_scope": scope,
        "items_included": len(lines),
        "items_pending": len(pending),
        "zero_depreciation_cover": zero_dep,
        "price_tier": tier, "vehicle_type": vtype,
        "make_used": make if make in (TW_MAKES if two_wheeler else MAKES) else "generic",
        "parts_priced_from_real_data": sum(1 for l in lines if l["part_price_source"].get("type") == "scraped"),
        "parts_priced_from_demo_estimates": sum(1 for l in lines if l["part_price_source"].get("type") == "demo_estimate"),
    } if lines else {}

    if garage > AUTO_APPROVAL_THRESHOLD_INR:
        block("escalate", "ABOVE_AUTO_LIMIT", f"garage estimate Rs {garage:,} > auto-approval limit "
                                              f"Rs {AUTO_APPROVAL_THRESHOLD_INR:,} - surveyor inspection")
    if reasons_escalate:
        return _decision("ESCALATED", reasons_escalate + reasons_review, lines, money_block, vehicle, notes,
                         findings, blockers, groups)
    if reasons_review:
        return _decision("REVIEW_REQUIRED", reasons_review, lines, money_block, vehicle, notes,
                         findings, blockers, groups)
    if pending:   # defensive: a pending item must ALWAYS have produced a blocker above
        return _decision("REVIEW_REQUIRED", ["unresolved findings"], lines, money_block, vehicle, notes,
                         findings, blockers, groups)
    if payable == 0:
        return _decision("NO_PAYOUT", [f"payable amount after depreciation and deductible is Rs 0 "
                                       f"(garage estimate Rs {garage:,})"], lines, money_block, vehicle, notes,
                         findings, [], groups)

    money_block["settlement_range"] = {
        "opening_offer_inr": money(payable * OPEN_OFFER_RATIO),
        "maximum_offer_inr": money(payable * MAX_OFFER_RATIO),
        "open_ratio": str(OPEN_OFFER_RATIO), "max_ratio": str(MAX_OFFER_RATIO),
    }
    return _decision("APPROVED", ["all checks passed - cash settlement can be offered"],
                     lines, money_block, vehicle, notes, findings, [], groups)


def _decision(status, reasons, lines, money_block, vehicle, notes, findings=None, blockers=None,
              groups=None) -> Dict:
    # An offer range may only exist on an APPROVED, COMPLETE estimate.  Belt and braces:
    if status != "APPROVED" or (money_block or {}).get("estimate_scope") != "complete":
        (money_block or {}).pop("settlement_range", None)
    return {
        "status": status,
        "reasons": reasons,
        "blockers": blockers or [],
        "findings": findings or [],
        "overlap_groups": groups or [],
        "line_items": lines,
        "totals": money_block,
        "vehicle": vehicle,
        "notes": notes,
        "authorized": status == "APPROVED",
        "requires_human": status in ("ESCALATED", "REVIEW_REQUIRED"),
        "currency": "INR",
        "pricing_data": ("labour/paint: MOCK demo rates; part prices: real scraped MRP where "
                         "'part_price_source.type' == 'scraped', else MOCK demo estimates"),
    }


# =============================================================================
# Chatbot guard (kept from the original design)
# =============================================================================
def validate_chatbot_action(action: Dict[str, Any], quote: Dict[str, Any]) -> Dict[str, Any]:
    """The chatbot may request actions but may never supply or change an amount.
    Amounts come only from this module / the negotiation state machine."""
    if not isinstance(action, dict):
        return {"accepted": False, "reason": "action must be a JSON object"}
    if any(k in action for k in ("amount", "amount_inr", "payout", "offer")):
        return {"accepted": False, "reason": "chatbot cannot supply or modify amounts"}
    kind = action.get("type")
    if kind == "request_human_review":
        return {"accepted": True, "action": kind}
    rng = quote.get("totals", {}).get("settlement_range")
    if kind in ("present_offer", "record_acceptance"):
        if not quote.get("authorized") or not rng:
            return {"accepted": False, "reason": "no authorized settlement for this claim"}
        return {"accepted": True, "action": kind, "range": rng}
    return {"accepted": False, "reason": f"unsupported action '{kind}'"}


# =============================================================================
# Mock policy lookup (licence plate -> vehicle). Replace with a real policy DB.
# =============================================================================
DEMO_POLICIES = {
    "DL3CAB1234": {"policy_no": "AS-MTR-2026-000184", "policy_start": "2026-04-01", "policy_end": "2027-03-31",
                   "make": "maruti", "model": "Swift", "vehicle_type": "car", "year": 2021, "engine_cc": 1197,
                   "zero_dep": False, "policy_active": True, "holder": "Demo Customer A"},
    "UP16BT4521": {"policy_no": "AS-MTR-2026-000391", "policy_start": "2026-04-01", "policy_end": "2027-03-31",
                   "make": "hyundai", "model": "Creta", "vehicle_type": "car", "year": 2023, "engine_cc": 1497,
                   "zero_dep": True, "policy_active": True, "holder": "Demo Customer B"},
    "HR26DK0001": {"policy_no": "AS-MTR-2026-000522", "policy_start": "2026-04-01", "policy_end": "2027-03-31",
                   "make": "bmw", "model": "320d", "vehicle_type": "car", "year": 2022, "engine_cc": 1995,
                   "zero_dep": False, "policy_active": True, "holder": "Demo Customer C"},
    # two-wheelers
    "DL5SAB5678": {"policy_no": "AS-2W-2026-000077", "policy_start": "2026-04-01", "policy_end": "2027-03-31",
                   "make": "hero", "model": "Splendor Plus", "year": 2022, "engine_cc": 97,
                   "vehicle_type": "motorcycle", "zero_dep": False, "policy_active": True,
                   "holder": "Demo Rider A"},
    "HR26BK9999": {"policy_no": "AS-2W-2026-000129", "policy_start": "2026-04-01", "policy_end": "2027-03-31",
                   "make": "honda", "model": "Activa 6G", "year": 2023, "engine_cc": 109,
                   "vehicle_type": "scooter", "zero_dep": True, "policy_active": True,
                   "holder": "Demo Rider B"},
}


def lookup_policy(plate: Optional[str]) -> Optional[Dict]:
    key = "".join(ch for ch in str(plate or "").upper() if ch.isalnum())
    return DEMO_POLICIES.get(key)


def resolve_vehicle(make=None, model=None, year=None, cc=None, zero_dep=False,
                    plate: Optional[str] = None, vehicle_type: Optional[str] = None):
    """CLI details win; otherwise look the plate up in the (mock) policy database.
    Returns (vehicle or None, list of suspicious-flag strings)."""
    if make:
        return {"make": make, "model": model, "year": year, "engine_cc": cc,
                "vehicle_type": norm_type(vehicle_type) or type_from_model(model),
                "zero_dep": zero_dep, "policy_active": True}, []
    if plate:
        v = lookup_policy(plate)
        if v:
            return v, []
        return None, [f"no active policy found for plate '{plate}'"]
    return None, []


def quote_from_file(path: str, vehicle: Optional[Dict] = None, plate: Optional[str] = None,
                    suspicious: Optional[List[str]] = None) -> Dict:
    data = json.loads(Path(path).read_text())
    items = items_from_video(data) if "keyframes" in data else items_from_fused(data)
    if vehicle is None:
        plate = plate or (data.get("plate") or {}).get("text")
        vehicle = lookup_policy(plate)
        if vehicle is None:
            vehicle = {}
            suspicious = (suspicious or []) + [f"no active policy found for plate '{plate or 'unknown'}'"]
    return price_claim(items, vehicle, suspicious)


def print_quote(q: Dict) -> None:
    print(f"\n=========== PRICING: {q['status']} ===========")
    for l in q["line_items"]:
        extra = f" (+ covers {', '.join(l['also_covers'])})" if l.get("also_covers") else ""
        dep = f", dep -{l['depreciation']} ({l['depreciation_pct']}%)" if l["depreciation"] else ""
        print(f"  {l['part']:<13} {l['damage']:<9} {l['severity']:<6} {l['method']:<13} "
              f"Rs {l['garage_cost']:>7,}  [part {l['part_cost']}, labour {l['labour']}, "
              f"paint {l['paint']}{dep}]{extra}")
        src = l.get("part_price_source", {})
        if src.get("type") == "scraped":
            vals = ", ".join(f"{v:,}" for v in src["values"])
            print(f"      part price: REAL {src['price_type']} from {', '.join(src['sites'])} "
                  f"(median of {src['n']}: {vals}; scraped {src['scraped_at'][:10]})")
        elif src.get("type") == "demo_estimate":
            print(f"      part price: demo estimate (no real price for this model/part)")
    t = q["totals"]
    if t:
        print(f"  Garage estimate   Rs {t['garage_estimate_inr']:>8,}")
        print(f"  - depreciation    Rs {t['depreciation_inr']:>8,}"
              + ("  (zero-dep cover)" if t["zero_depreciation_cover"] else ""))
        print(f"  - deductible      Rs {t['deductible_inr']:>8,}")
        print(f"  = insurer payable Rs {t['insurer_payable_inr']:>8,}")
        if "settlement_range" in t:
            r = t["settlement_range"]
            print(f"  Cash offer range  Rs {r['opening_offer_inr']:,} (opening) .. "
                  f"Rs {r['maximum_offer_inr']:,} (maximum)")
        if t.get("estimate_scope") == "partial":
            print(f"  (PROVISIONAL PARTIAL estimate: {t['items_pending']} finding(s) pending review, not priced)")
    for f in q.get("findings", []):
        if f["status"] != "confirmed_priceable":
            print(f"  [{f['status']}] {f['part']}-{f['damage']}: {f.get('reason') or ''}")
    for r in q["reasons"]:
        print(f"  * {r}")
    for n in q["notes"]:
        print(f"  - {n}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Auto-Settle AI - pricing & settlement range")
    ap.add_argument("report", nargs="?", help="fused.json (photo) or video_report.json (video)")
    ap.add_argument("--plate", help="licence plate -> demo policy lookup")
    ap.add_argument("--make"); ap.add_argument("--model"); ap.add_argument("--year", type=int)
    ap.add_argument("--cc", type=int, help="engine cc (deductible slab)")
    ap.add_argument("--zero-dep", action="store_true")
    ap.add_argument("--vehicle-type", choices=["car", "motorcycle", "scooter"])
    ap.add_argument("--out", help="save quote JSON here (default: next to the report)")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(argv)
    if a.demo:
        return run_demo()
    if not a.report:
        ap.error("give a fused.json / video_report.json, or --demo")
    vehicle = None
    if a.make:
        vehicle, _ = resolve_vehicle(a.make, a.model, a.year, a.cc, a.zero_dep,
                                     vehicle_type=a.vehicle_type)
    try:
        q = quote_from_file(a.report, vehicle, a.plate)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print_quote(q)
    out = Path(a.out) if a.out else Path(a.report).with_name("quote.json")
    out.write_text(json.dumps(q, indent=2))
    print(f"Saved -> {out}")
    return 0


def run_demo() -> int:
    swift = DEMO_POLICIES["DL3CAB1234"]
    cases = {
        "Your front photo (Swift 2021)": [
            {"part": "headlight", "damage": "lamp broken", "confidence": 0.90, "severity": "severe"},
            {"part": "front bumper", "damage": "crack", "confidence": 0.40, "severity": "minor"},
            {"part": "front bumper", "damage": "scratch", "confidence": 0.39, "severity": "minor"},
        ],
        "Clean small claim: door dent + bumper scratch": [
            {"part": "car door", "damage": "dent", "confidence": 0.85, "severity": "minor"},
            {"part": "front bumper", "damage": "scratch", "confidence": 0.70, "severity": "minor"},
        ],
    }
    for title, items in cases.items():
        print(f"\n##### {title}")
        print_quote(price_claim(items, swift))
    print("\n##### Same headlight on a BMW (over threshold)")
    print_quote(price_claim([{"part": "headlight", "damage": "lamp broken", "confidence": 0.9,
                              "severity": "low"}], DEMO_POLICIES["HR26DK0001"]))
    print("\n##### Light scratch on a headlight lens (below deductible)")
    print_quote(price_claim([{"part": "headlight", "damage": "scratch", "confidence": 0.8,
                              "severity": "minor"}], swift))
    print("\n##### Splendor Plus 2022: fuel tank dent + broken headlight")
    print_quote(price_claim([
        {"part": "fuel tank", "damage": "dent", "confidence": 0.7, "severity": "medium"},
        {"part": "headlight", "damage": "lamp broken", "confidence": 0.8, "severity": "high"},
    ], DEMO_POLICIES["DL5SAB5678"]))
    print("\n##### Activa 6G 2023 (zero-dep): cracked front apron + scratched floorboard + bent handlebar")
    print_quote(price_claim([
        {"part": "front apron", "damage": "crack", "confidence": 0.75, "severity": "high"},
        {"part": "floorboard", "damage": "scratch", "confidence": 0.6, "severity": "low"},
        {"part": "handlebar", "damage": "dent", "confidence": 0.6, "severity": "medium"},
    ], DEMO_POLICIES["HR26BK9999"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
