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
import os
import re
import statistics
import sys
from datetime import date, datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# Catalogue part names -> the canonical price key a detected panel uses.  Only whole panels
# that a claim can actually be priced for are mapped; sub-components (hinges, brackets,
# weatherstrips, floor pans, screws ...) stay in the table under their own name, unused
# by pricing.  A left/right suffix is dropped: the detector does not tell sides apart.
CATALOGUE_PART_MAP = {
    **{p: p for p in PARTS},                     # canonical names map to themselves (incl. two-wheelers)
    "front_bumper": "front_bumper", "rear_bumper": "rear_bumper",
    "grille": "grille", "upper_radiator_grille_assembly": "grille",
    "quarter_panel": "quarter_panel",
    "front_door_window_glass": "side_window", "rear_door_window_glass": "side_window",
    "headlight": "headlight", "head_lamp": "headlight", "taillight": "taillight", "tail_lamp": "taillight",
    "side_mirror": "side_mirror", "car_door": "car_door", "fender": "fender", "car_hood": "car_hood",
}
SECONDARY_SOURCE = re.compile(r"secondary|needs_official_recheck", re.I)


def model_key(model: Optional[str]) -> str:
    """'WagonR', 'Wagon R', 'wagon_r' -> 'wagonr'; 'Activa 6G' / 'activa_6g' -> 'activa6g'."""
    return norm(model).replace(" ", "")


def make_key(make: Optional[str]) -> str:
    m = norm(make)
    return MAKE_ALIASES.get(m, m)


def catalogue_part(part: Optional[str]) -> Tuple[str, Optional[str]]:
    """(raw catalogue key, canonical price key or None)."""
    raw = norm(part).replace(" ", "_")
    base = re.sub(r"_(left|right)(_(rear|front))?$", "", raw)
    return raw, CATALOGUE_PART_MAP.get(base)


def load_external_prices(path: Path = EXTERNAL_PRICES_CSV) -> Dict:
    """(make, model, canonical part) -> list of price rows.  Missing/broken file -> {} (mock prices).
    Rows for catalogue sub-components are stored under (make, model, '#'+raw name): visible to
    staff tooling, never used by pricing."""
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
                raw, canon = catalogue_part(r.get("part"))
                key = (make_key(r.get("make")), model_key(r.get("model")), canon or "#" + raw)
                table.setdefault(key, []).append({**r, "price_inr": price, "catalogue_part": raw,
                                                  "secondary": bool(SECONDARY_SOURCE.search(
                                                      f"{r.get('price_type', '')} {r.get('fitment', '')}"))})
    except OSError:
        pass
    return table


def priced_parts_for(make: Optional[str], model: Optional[str]) -> List[str]:
    """Canonical parts with a usable real price for this exact vehicle (for the policy view)."""
    global _EXTERNAL
    if _EXTERNAL is None:
        _EXTERNAL = load_external_prices()
    mk, md = make_key(make), model_key(model)
    return sorted(p for (a, b, p) in _EXTERNAL if a == mk and b == md and not p.startswith("#")
                  and (real_part_price(make, model, p) or {}).get("price"))


def real_part_price(make: str, model: Optional[str], part: str) -> Optional[Dict]:
    """Median of the scraped prices for this exact make+model+part, with its sources."""
    global _EXTERNAL
    if _EXTERNAL is None:
        _EXTERNAL = load_external_prices()
    all_rows = _EXTERNAL.get((make_key(make), model_key(model), part))
    if not all_rows:
        return None
    # A row whose fitment the scraper could only infer ("... - verify") is not used
    # automatically: a wrong-model part price must not reach a quote unreviewed.
    rows = [r for r in all_rows if not UNVERIFIED_FITMENT.search(r.get("fitment") or "")]
    if not rows:
        return {"unverified_only": True, "urls": [r.get("source_url", "") for r in all_rows]}
    # one official (manufacturer) price beats secondary catalogues for the same part
    official = [r for r in rows if not r["secondary"]]
    rows = official or rows
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
            "part_numbers": sorted({r.get("part_number", "") for r in rows if r.get("part_number")}),
            "catalogue_parts": sorted({r["catalogue_part"] for r in rows}),
            "secondary_source": not official,
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


LESSER_OF_METHODS = {"DENT_REPAIR", "PLASTIC_WELD", "REPAINT"}


def provenance(line: Dict, demo: bool = False) -> Dict[str, str]:
    """Where each money input of a line comes from.  OBSERVED = read from a real source for this
    exact vehicle; IMPUTED = an assumption / rate card; RULE = a fixed schedule applied to facts."""
    src = line["part_price_source"]
    if line["part_cost"] == 0:
        part = "N/A - no part replaced"
    elif src.get("type") == "scraped" and not src.get("secondary_source"):
        part = "OBSERVED - official genuine-part MRP (" + ", ".join(src.get("sites") or []) + ")"
    elif src.get("type") == "scraped":
        part = "IMPUTED - secondary genuine-parts catalogue (" + ", ".join(src.get("sites") or []) + "), official recheck pending"
    else:
        part = "IMPUTED - demo rate card for the price tier"
    dep = ("N/A - nothing replaced" if line["part_cost"] == 0 else
           "IMPUTED - vehicle age unknown in Demo Mode, most conservative 50% for metal" if demo else
           f"RULE - IMT-style schedule: {line['material']} part, vehicle age from the policy record")
    return {"part_price": part, "labour": "IMPUTED - demo rate card", "paint": "IMPUTED - demo rate card"
            if line["paint"] else "N/A - no paint", "depreciation": dep}


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


def _vote_summary(d: Dict) -> Dict:
    """From the VLM's per-view votes: the severities the damage-seeing views gave, whether a
    MAJORITY of them raised a quality / reflection / wrong-part flag, and - for the stricter
    Demo Mode rule - whether ANY valid view raised a flag or said 'no damage'."""
    valid = [v for v in d.get("severity_votes") or [] if "error" not in v]
    seen = [v for v in valid if v.get("visible")]
    return {"severity_levels": sorted({v["severity"] for v in seen if v.get("severity")}),
            "severity_flagged": sum(1 for v in seen if v.get("review")) * 2 > len(seen) if seen else False,
            "severity_any_flag": any(v.get("review") for v in seen),
            "vlm_all_views_visible": bool(valid) and len(seen) == len(valid)}


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
                    "part_uncertain_reason": m.get("part_uncertain_reason"),
                    "part_candidates": m.get("part_candidates"), **_vote_summary(m),
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
                    "part_uncertain_reason": d.get("part_uncertain_reason"), **_vote_summary(d),
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


# =============================================================================
# DEMO MODE (judges upload their own photos; there is no real policy behind them)
# =============================================================================
# The pipeline detects the vehicle TYPE (car / motorcycle / scooter) but has NO make/model or
# price-class classifier, so the price tier in Demo Mode is an operator-chosen assumption.
# "auto" = unknown -> the generic mid-range tier.  Never Maruti, never a model-specific
# scraped price: the demo vehicle's identity is not established.
DEMO_TIERS = ("auto", "economy", "mid", "luxury", "two_wheeler")
DEMO_TIER_LABELS = {"auto": "Auto / unknown (generic mid-range)", "economy": "Economy", "mid": "Mid-range",
                    "luxury": "Luxury", "two_wheeler": "Two-wheeler"}
# Blockers that stop ANY demo offer, even a scoped one: claim-level problems and suspected
# safety-critical / structural damage anywhere on the vehicle.
DEMO_OFFER_STOPS = {"SUSPICIOUS", "PHOTO_EDITED", "MAKE_UNKNOWN", "POLICY_INACTIVE", "IMAGE_QUALITY", "ABOVE_AUTO_LIMIT",
                    "ALWAYS_INSPECT", "STRUCTURAL_INSPECT", "NO_PRICEABLE_DAMAGE"}
COSMETIC_DAMAGE = {"dent", "scratch"}


def demo_vehicle(tier: str, detected_type: Optional[str]) -> Dict:
    """The 'vehicle' of a Demo Mode claim: no make, model, plate or policy - only the
    detected type and the operator's tier assumption."""
    return {"demo": True, "demo_tier": tier if tier in DEMO_TIERS else "auto", "make": None, "model": None,
            "year": None, "engine_cc": None, "vehicle_type": norm_type(detected_type) or "car",
            "zero_dep": False, "policy_active": True}


def demo_tier_for(vehicle: Dict) -> Tuple[str, str]:
    """(price tier, how it was chosen).  A detected two-wheeler always uses the two-wheeler tier."""
    req = vehicle.get("demo_tier") or "auto"
    vtype = norm_type(vehicle.get("vehicle_type")) or "car"
    if vtype in TWO_WHEELERS:
        return "two_wheeler", ("operator-selected" if req == "two_wheeler" else "detected vehicle type (two-wheeler)")
    if req in ("economy", "mid", "luxury"):
        return req, "operator-selected demo assumption"
    return "mid", "generic fallback (tier unknown)" if req != "two_wheeler" else \
        "generic fallback (two-wheeler tier selected but a car was detected)"


def finding_signature(it: Dict) -> Tuple[str, str]:
    """Identity of a damage across claims: canonical part + canonical damage."""
    return (PART_ALIASES.get(norm(it.get("part")), norm(it.get("part"))),
            DAMAGE_ALIASES.get(norm(it.get("damage")), norm(it.get("damage"))))


def price_claim(items: List[Dict], vehicle: Dict, suspicious: Optional[List[str]] = None,
                settled: Optional[set] = None, repeat_parts: Optional[set] = None,
                garage_overrides: Optional[Dict] = None) -> Dict:
    """vehicle: {make, model, year, engine_cc, zero_dep, policy_active} (policy record), or
    demo_vehicle(...) in Demo Mode.
    settled      = item signatures already paid for THIS SAME IMAGE (same file or a near-identical
                   copy, any session, any mode) -> excluded, never paid twice.
    repeat_parts = signatures already paid on THIS POLICY from a different photo -> a person
                   checks it is new damage before it can be priced again."""
    demo = bool(vehicle.get("demo"))
    settled, repeat_parts = settled or set(), repeat_parts or set()
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
    make = "demo" if demo else MAKE_ALIASES.get(norm(vehicle.get("make")), norm(vehicle.get("make")))
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
    tier_source = "policy make"
    if demo:
        tier, tier_source = demo_tier_for(vehicle)
        mult = 1.0
        notes.append(f"Demo Mode: illustrative '{tier}' rate card ({tier_source}); vehicle make/model not "
                     "identified, so no model-specific prices are used")
    elif two_wheeler:
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
    if demo:
        notes.append("Demo Mode: vehicle age unknown - the most conservative metal depreciation (50%) applies")
    elif not vehicle.get("model") or age is None:
        block("review", "VEHICLE_UNVERIFIED", "model / year missing - exact vehicle not verified")
    zero_dep = bool(vehicle.get("zero_dep"))
    for s in suspicious or []:
        block("escalate", "PHOTO_EDITED" if s.startswith("photo metadata") else "SUSPICIOUS", f"suspicious: {s}")

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
                         "region": it.get("region"), "evidence": it.get("evidence"),
                         "demo_rule": it.get("demo_rule"), "source": it.get("evidence_source"),
                         "image_ref": it.get("image_ref"), **extra})

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
        if finding_signature(it) in settled:
            notes.append(f"{raw}: already paid for this same image in an earlier claim - not counted again")
            finding(i, it, "duplicate_or_overlap", "ALREADY_SETTLED",
                    "already paid for this same image in an earlier claim - not counted again")
            continue
        if finding_signature(it) in repeat_parts:
            review("PREVIOUSLY_SETTLED",
                   "this part and damage were already paid on this policy - a person checks it is new damage")
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
            unsupported("PART_LOCATION_UNCLEAR", it.get("part_uncertain_reason") or "damaged panel could not be identified")
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
        levels = set(it.get("severity_levels") or [])
        if (demo and it.get("severity_review") and ver == "confirmed" and damage in COSMETIC_DAMAGE
                and part not in LAMP_GLASS and not it.get("part_uncertain")
                and (it.get("vlm_valid_views") or 0) >= 2 and it.get("vlm_all_views_visible")
                and not it.get("severity_flagged") and not it.get("severity_any_flag")
                and levels == {"low", "medium"}):
            # Demo review profile, deliberately narrow: a clearly located cosmetic dent/scratch,
            # EVERY valid VLM view saw real damage, NO view raised a quality / reflection /
            # wrong-part flag, and the views only split between low and medium.
            # Deterministic rule: 'medium' (the fuller cosmetic repair, never a replacement).
            # Any 'high', any flag, any 'no damage' view or an unclear panel -> normal review.
            sev = "medium"
            notes.append(f"{raw}: VLM views split low/medium - Demo Mode cosmetic rule used 'medium'")
            it = {**it, "demo_rule": "cosmetic severity split -> medium"}
        elif it.get("severity_review"):
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
        if decision["method"] in LESSER_OF_METHODS and part in prices["parts"] and part not in LAMP_GLASS:
            # Lesser-of rule: a repairable part is never paid more than replacing it would cost after
            # depreciation.  It also guarantees that a LOWER severity can never raise the amount.
            alt = cost_line(part, "REPLACE", prices, mult, age, zero_dep, make=make, model=vehicle.get("model"))
            net = lambda l: l["garage_cost"] - l["depreciation"]  # noqa: E731
            if net(alt) < net(line):
                notes.append(f"{raw}: replacing (net Rs {net(alt):,}) is cheaper than repairing "
                             f"(Rs {net(line):,}) - lesser-of rule: replacement priced")
                alt["lesser_of"] = {"instead_of": line["method"], "repair_net": net(line), "replace_net": net(alt)}
                line = alt
        line["provenance"] = provenance(line, demo)
        src = line["part_price_source"]
        if src.get("type") == "scraped" and src.get("stale"):
            notes.append(f"{raw}: real price older than {STALE_AFTER_DAYS} days - re-run price_scraper.py")
        if src.get("skipped_unverified"):
            notes.append(f"{raw}: scraped price with unverified fitment ignored - demo estimate used")
        if src.get("type") == "scraped" and src.get("secondary_source"):
            notes.append(f"{raw}: part price from a secondary genuine-parts catalogue "
                         f"({', '.join(src.get('sites') or [])}) - official MRP recheck pending")
        if demo:
            src["demo_tier"] = tier
        ov = (garage_overrides or {}).get(part)
        if ov:
            apply_garage_override(line, ov)
        line.update(part=part, damage=damage, severity=sev, severity_source=it.get("severity_source"),
                    demo_rule=it.get("demo_rule"),
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
                                                        "duplicate_or_overlap")
               and f.get("reason_code") != "ALREADY_SETTLED"]
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
    if demo and money_block:
        money_block.update(demo_mode=True, tier_requested=vehicle.get("demo_tier") or "auto",
                           tier_source=tier_source, make_used="demo (not identified)")

    if garage > AUTO_APPROVAL_THRESHOLD_INR:
        block("escalate", "ABOVE_AUTO_LIMIT", f"garage estimate Rs {garage:,} > auto-approval limit "
                                              f"Rs {AUTO_APPROVAL_THRESHOLD_INR:,} - surveyor inspection")
    if reasons_escalate or reasons_review:
        status = "ESCALATED" if reasons_escalate else "REVIEW_REQUIRED"
        q = _decision(status, reasons_escalate + reasons_review, lines, money_block, vehicle, notes,
                      findings, blockers, groups)
        if demo:
            _scoped_demo_offer(q, payable)
        return q
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


# =============================================================================
# Evidence: merging a re-assessed proof photo into a claim (deterministic, no language model)
# =============================================================================
SEV_RANK = {"low": 1, "medium": 2, "high": 3}


def _usable(it: Dict) -> bool:
    """A proof item counts only if the evidence itself is solid: a located part, not ruled out,
    and VLM-confirmed (or, with the VLM off, a confident detection)."""
    if it.get("quality_issue") or it.get("vlm_only") or it.get("part_uncertain"):
        return False
    if it.get("video_status") in ("inconsistent", "weak"):
        return False
    ver = _verification(it)
    conf = it.get("confidence") if isinstance(it.get("confidence"), (int, float)) else 0
    return ver == "confirmed" or (ver == "not_run" and conf >= MIN_CONF_DETECTOR)


def _rank(it: Dict) -> Tuple:
    return (_usable(it), SEV_RANK.get(SEVERITY_ALIASES.get(norm(it.get("severity")), ""), 0),
            it.get("confidence") or 0)


def merge_evidence(orig_items: List[Dict], orig_damages: List[Dict], ev_items: List[Dict],
                   ev_damages: List[Dict], n: int, image_ref: Optional[Dict] = None) -> Dict:
    """Combine the claim's findings with those of proof photo #n.
    - same part + damage, and the proof shows it better (now confirmed / clearly located /
      more severe)  -> the proof version replaces it ("upgraded" or "resolved")
    - new part + damage that the proof shows solidly                          -> added
    - anything weak, unconfirmed or ruled out in the proof                     -> ignored
    Items and their damage summaries stay aligned (quality items have no summary), and the
    original ones keep their positions so boxes on the original photo still match."""
    def pairs(items, damages):
        out, k = [], 0
        for it in items:
            if it.get("quality_issue"):
                out.append([it, None])
            else:
                out.append([it, damages[k] if k < len(damages) else {}])
                k += 1
        return out
    merged = pairs(orig_items, orig_damages)
    index = {finding_signature(it): i for i, (it, _) in enumerate(merged) if not it.get("quality_issue")}
    added, upgraded, resolved, ignored = [], [], [], []
    ev_quality = any(it.get("quality_issue") for it in ev_items)
    for it, dmg in pairs(ev_items, ev_damages):
        if it.get("quality_issue"):
            continue
        sig = finding_signature(it)
        label = f"{str(it['part']).replace('_', ' ')} ({str(it['damage']).replace('_', ' ')})"
        if ev_quality or not _usable(it):
            ignored.append(label)
            continue
        tagged = {**it, "evidence_source": n, "image_ref": image_ref,
                  "region": {**(it.get("region") or {}), "frame": f"proof{n}"} if it.get("region") else None}
        dtag = {**(dmg or {}), "source": f"proof photo {n}"}
        if sig in index:
            i = index[sig]
            old = merged[i][0]
            if _rank(tagged) > _rank(old):
                (upgraded if _usable(old) else resolved).append(label)
                merged[i] = [tagged, {**(merged[i][1] or {}), **dtag}]
            else:
                ignored.append(label)
        else:
            index[sig] = len(merged)
            merged.append([tagged, dtag])
            added.append(label)
    quality_resolved = False
    if not ev_quality and (added or upgraded or resolved):
        before = len(merged)
        merged = [p for p in merged if not p[0].get("quality_issue")]
        quality_resolved = len(merged) < before
    return {"items": [p[0] for p in merged], "damages": [p[1] for p in merged if p[1] is not None],
            "added": added, "upgraded": upgraded, "resolved": resolved, "ignored": ignored,
            "evidence_quality_issue": ev_quality, "quality_resolved": quality_resolved,
            "material": bool(added or upgraded or resolved or quality_resolved)}


# =============================================================================
# Garage estimate / invoice: deterministic line checks (no language model decides anything)
# =============================================================================
# DEMO ASSUMPTION, not a market fact: a garage line is accepted automatically only up to this
# multiple of OUR rate for the same part + component.  Above it, our figure is kept (the excess
# is never paid automatically).  Configurable: AUTOSETTLE_QUOTE_CAP=1.20
GARAGE_LINE_CAP = Decimal(os.environ.get("AUTOSETTLE_QUOTE_CAP", "1.25"))
GARAGE_MAX_LINES = 40
_COMPONENT = {"part": "part_cost", "labour": "labour", "paint": "paint"}
_LINE_TYPES = [   # (type, words) - checked in order on the extracted type, then the description
    ("paint", ("paint", "painting", "refinish", "polish", "colour", "color", "clear coat")),
    ("labour", ("labour", "labor", "r/r", "r&r", "fitting", "denting", "tinkering", "repair", "remove",
                "welding", "service charge")),
    ("part", ("part", "parts", "spare", "spares", "replace", "replacement", "new", "assy", "assembly")),
    ("other", ("gst", "tax", "cgst", "sgst", "igst", "consumable", "discount", "total", "round")),
]
_PENDING_STATUSES = ("uncertain_review", "unsupported_review", "duplicate_or_overlap")
_AMOUNT = re.compile(r"^(?:rs\.?|inr|₹)?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(?:/-)?$", re.I)


def parse_amount(x) -> Optional[Decimal]:
    """A number from extracted text.  Anything that is not plainly a number -> None."""
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return Decimal(str(x)) if x == x and abs(x) < 1e9 else None
    m = _AMOUNT.match(str(x).strip())
    if not m:
        return None
    try:
        return Decimal(m.group(1).replace(",", ""))
    except ArithmeticError:
        return None


def match_part(text: Optional[str]) -> Optional[str]:
    """Canonical price key named in a line ('Front Bumper Painting' -> front_bumper), or None.
    The longest alias wins, so 'rear bumper' is never read as just 'bumper'."""
    t = f" {norm(text)} "
    if not t.strip():
        return None
    direct = PART_ALIASES.get(t.strip())
    if direct:
        return direct
    for alias in sorted(PART_ALIASES, key=len, reverse=True):
        if f" {alias} " in t:
            return PART_ALIASES[alias]
    key = t.strip().replace(" ", "_")
    return key if key in PARTS else None


def line_type(extracted: Optional[str], description: str) -> str:
    for source in (norm(extracted), norm(description)):
        for kind, words in _LINE_TYPES:
            if source in (kind, kind + "s") or any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", source) for w in words):
                return kind
    return "other"


def _close(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= max(Decimal(2), abs(b) * Decimal("0.005"))


def check_garage_estimate(doc: Dict, quote: Dict, cap: Decimal = None) -> Dict:
    """Check an extracted garage estimate against the claim.  `doc` is UNTRUSTED candidate data
    (from a vision model reading the document).  Python re-does every sum; each line must match a
    part this claim's photos already priced, the same repair method, and stay within `cap` x our
    rate for that component.  The document's grand total is never used as an amount.

    Returns {"ok", "problems", "lines", "overrides", "accepted", "cap"}:
      ok=False  -> unreadable / malformed / figures don't add up -> no automatic change at all
      overrides -> {part: {"method", "part_cost"|"labour"|"paint": garage amount}} for price_claim"""
    cap = cap or GARAGE_LINE_CAP
    out = {"ok": False, "problems": [], "lines": [], "overrides": {}, "accepted": 0, "cap": str(cap),
           "authenticity": "not verified - contents checked against the claim evidence and pricing rules only"}
    problems = out["problems"]
    if not isinstance(doc, dict) or doc.get("readable") is False:
        problems.append("the document could not be read as an itemised estimate")
        return out
    raw_lines = doc.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        problems.append("no itemised lines could be read")
        return out
    if len(raw_lines) > GARAGE_MAX_LINES:
        problems.append(f"more than {GARAGE_MAX_LINES} lines - too long to check automatically")
        return out

    lines = []
    for n, raw in enumerate(raw_lines, 1):
        if not isinstance(raw, dict):
            problems.append(f"line {n} is malformed")
            continue
        desc = " ".join(str(raw.get("description") or "").split())[:120] or f"line {n}"
        amount = parse_amount(raw.get("amount"))
        qty = Decimal(1) if raw.get("qty") in (None, "") else parse_amount(raw.get("qty"))
        rate = None if raw.get("rate") in (None, "") else parse_amount(raw.get("rate"))
        if amount is None or amount <= 0:
            problems.append(f"'{desc}': amount unreadable")
            continue
        if qty is None or qty <= 0 or qty != qty.to_integral_value():
            problems.append(f"'{desc}': quantity unreadable")
            continue
        if raw.get("rate") not in (None, "") and rate is None:
            problems.append(f"'{desc}': rate unreadable")
            continue
        if rate is not None and not _close(rate * qty, amount):
            problems.append(f"'{desc}': {qty} x {rate} is not {amount}")
            continue
        lines.append({"n": n, "description": desc, "type": line_type(raw.get("type"), desc),
                      "part": match_part(raw.get("part")) or match_part(desc),
                      "qty": int(qty), "amount": money(amount)})

    line_sum = sum((Decimal(l["amount"]) for l in lines), Decimal(0))
    fields = {k: parse_amount(doc.get(k)) for k in ("subtotal", "tax_total", "grand_total")}
    for k, v in fields.items():
        if doc.get(k) not in (None, "") and v is None:
            problems.append(f"the {k.replace('_', ' ')} is unreadable")
    sub, tax, grand = fields["subtotal"], fields["tax_total"], fields["grand_total"]
    # tax / total rows that were read as lines are not repair lines: compare without them
    repair_sum = sum((Decimal(l["amount"]) for l in lines if l["type"] != "other"), Decimal(0))
    if sub is not None and not (_close(line_sum, sub) or _close(repair_sum, sub)):
        problems.append(f"the lines add up to Rs {money(line_sum):,} but the estimate's subtotal says Rs {money(sub):,}")
    if grand is not None:
        base = sub if sub is not None else repair_sum
        if not (_close(base + (tax or 0), grand) or _close(line_sum, grand)):
            problems.append(f"subtotal + tax (Rs {money(base + (tax or 0)):,}) does not match the "
                            f"grand total (Rs {money(grand):,})")
    if problems:                      # contradictory figures: nothing from this document is used
        out["lines"] = [dict(l, status="not_checked", reason="the document's figures did not check out") for l in lines]
        return out

    priced = {l["part"]: l for l in quote.get("line_items", [])}
    pending = {f.get("price_key") for f in quote.get("findings", []) if f["status"] in _PENDING_STATUSES}
    seen, groups = set(), {}
    for l in lines:
        key = (norm(l["description"]), l["amount"], l["type"])
        ql = priced.get(l["part"])
        nice = (l["part"] or "").replace("_", " ")
        if key in seen:
            l.update(status="duplicate", reason="same line appears twice - counted once")
        elif l["type"] == "other":
            l.update(status="not_compared", reason="not a part, labour or paint charge - not used (we apply our own tax and policy rules)")
        elif not l["part"]:
            l.update(status="unsupported", reason="does not name a panel we can match to your claim")
        elif not ql:
            l.update(status="unsupported", reason=(f"the {nice} is still under review - send a clear photo of it first"
                                                   if l["part"] in pending else
                                                   f"no damage to the {nice} was found in your photos"))
        elif l["type"] == "part" and ql["part_cost"] == 0:
            l.update(status="method_mismatch", reason=f"the estimate replaces the {nice}, but the photos support "
                                                      f"{ql['method'].replace('_', ' ').lower()} - our repair method stays")
        elif l["type"] == "part" and l["qty"] > 1:
            l.update(status="unsupported", reason=f"quantity {l['qty']} of one {nice}")
        elif l["type"] == "paint" and ql["paint"] == 0:
            l.update(status="method_mismatch", reason=f"painting is not part of the {nice} repair")
        elif l["type"] == "labour" and ql["labour"] == 0:
            l.update(status="method_mismatch", reason=f"labour is not part of the {nice} repair")
        else:
            groups.setdefault((l["part"], _COMPONENT[l["type"]]), []).append(l)
            l["status"] = "compared"
        seen.add(key)

    for (part, comp), ls in groups.items():
        ql = priced[part]
        ours = (ql.get("base_components") or ql)[comp]
        garage = sum(x["amount"] for x in ls)
        limit = money(ours * cap)
        if garage <= ours:
            st, why = "within_our_estimate", f"our rate (Rs {ours:,}) already covers this"
        elif garage <= limit:
            st, why = "accepted", f"accepted: Rs {garage:,} is within {cap * 100:.0f}% of our rate (Rs {ours:,})"
            out["overrides"].setdefault(part, {"method": ql["method"]})[comp] = garage
        else:
            st, why = "above_cap", (f"Rs {garage:,} is more than {cap * 100:.0f}% of our rate (Rs {ours:,}) - "
                                    f"our figure is kept")
        for x in ls:
            x.update(status=st, reason=why, ours=ours, cap_amount=limit, garage_component_total=garage)
    for l in lines:   # left out of the offer, shown to staff as lines a specialist may look at
        l["for_review"] = l["status"] in ("unsupported", "method_mismatch", "above_cap")
    out["lines"] = lines
    out["accepted"] = sum(1 for l in lines if l["status"] == "accepted")
    out["ok"] = True
    return out


def merge_overrides(old: Optional[Dict], new: Dict) -> Dict:
    """Overrides from several estimates: per part + component the higher CHECKED figure (each is
    already within the cap of our rate; price_claim enforces the cap again)."""
    merged = {p: dict(v) for p, v in (old or {}).items()}
    for part, ov in new.items():
        cur = merged.get(part)
        if not cur or cur.get("method") != ov.get("method"):
            merged[part] = dict(ov)
            continue
        for k, v in ov.items():
            if k != "method":
                cur[k] = max(cur.get(k, 0), v)
    return merged


def apply_garage_override(line: Dict, ov: Dict) -> None:
    """Use checked garage figures for a priced line - only for the SAME repair method, only
    upward, never above the cap, and depreciation is recomputed on the part price."""
    if ov.get("method") != line["method"]:
        return
    base = {k: line[k] for k in ("part_cost", "labour", "paint")}
    applied = {}
    for k, ours in base.items():
        v = ov.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and ours > 0 and v > ours:
            v = min(v, money(ours * GARAGE_LINE_CAP))
            line[k] = v
            applied[k] = {"ours": ours, "garage": v}
    if applied:
        line["base_components"] = base
        line["garage_cost"] = line["part_cost"] + line["labour"] + line["paint"]
        line["depreciation"] = money(line["part_cost"] * line["depreciation_pct"] / 100)
        line["garage_estimate_applied"] = applied


def _scoped_demo_offer(q: Dict, payable: int) -> None:
    """Demo Mode only: an offer limited to the findings that passed every check, while the
    rest stay pending.  Not made when any claim-level or safety stop is present, and never
    for a payable of 0.  The quote keeps its REVIEW/ESCALATED status; the offer lists exactly
    which finding ids it covers."""
    codes = {b["code"] for b in q["blockers"]}
    if not q["line_items"] or payable <= 0 or codes & DEMO_OFFER_STOPS:
        q["scoped_offer"] = None
        return
    q["totals"]["settlement_range"] = {
        "opening_offer_inr": money(payable * OPEN_OFFER_RATIO),
        "maximum_offer_inr": money(payable * MAX_OFFER_RATIO),
        "open_ratio": str(OPEN_OFFER_RATIO), "max_ratio": str(MAX_OFFER_RATIO),
    }
    included = [l["finding"] for l in q["line_items"]]
    covered = {f["id"] for f in q["findings"] if f["status"] == "confirmed_priceable"}
    q["scoped_offer"] = {
        "demo": True,
        "finding_ids": sorted(covered),
        "line_finding_ids": included,
        "excluded_finding_ids": [f["id"] for f in q["findings"] if f["id"] not in covered],
        "signatures": sorted({(f["price_key"] or f["part"], DAMAGE_ALIASES.get(norm(f["damage"]), norm(f["damage"])))
                              for f in q["findings"] if f["id"] in covered}),
    }
    q["authorized"] = True
    q["notes"].append("Demo Mode: scoped partial offer on the listed findings only; the rest stay under review")


def _decision(status, reasons, lines, money_block, vehicle, notes, findings=None, blockers=None,
              groups=None) -> Dict:
    # An offer range may only exist on an APPROVED, COMPLETE estimate.  Belt and braces:
    if status != "APPROVED" or (money_block or {}).get("estimate_scope") != "complete":
        (money_block or {}).pop("settlement_range", None)   # a scoped demo offer re-adds it explicitly
    return {
        "status": status,
        "reasons": reasons,
        "blockers": blockers or [],
        "findings": findings or [],
        "overlap_groups": groups or [],
        "scoped_offer": None,
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
    "UP408060": {"policy_no": "AS-MTR-2026-000612", "policy_start": "2026-04-01", "policy_end": "2027-03-31",
                 "make": "maruti", "model": "WagonR", "vehicle_type": "car", "year": 2023, "engine_cc": 998,
                 "zero_dep": False, "policy_active": True, "holder": "Demo Customer D"},
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
