"""
Auto-Settle AI - vehicle types (car / motorcycle / scooter)
============================================================

One place that knows which parts each vehicle type has.  Used by:
  grounding_dino_parts.py  -> which part names to prompt DINO with
  fuse.py / yolo_damage.py -> which parts the VLM must double-check
  pricing_rules.py         -> two-wheeler price list and deductible

No model training is needed for two-wheelers: Grounding DINO is zero-shot, so we
only change the TEXT prompt.  YOLO (CarDD) was trained on cars only, so on a
two-wheeler every part it is unsure about is sent to the VLM for a fact check.
"""

from __future__ import annotations

from typing import Dict, List, Optional

CAR_PARTS: List[str] = [
    "car door", "fender", "quarter panel", "side skirt", "door handle", "side mirror", "wheel",
    "front bumper", "rear bumper", "headlight", "taillight", "fog light", "car hood", "grille",
    "windshield", "side window", "trunk",
]
MOTORCYCLE_PARTS: List[str] = [
    "fuel tank", "front mudguard", "headlight", "taillight", "side panel", "handlebar",
    "exhaust pipe", "rear view mirror", "wheel", "seat",
]
SCOOTER_PARTS: List[str] = [
    "front apron", "front fender", "headlight", "taillight", "side body panel", "floorboard",
    "handlebar", "rear view mirror", "wheel", "seat",
]
VEHICLE_PARTS: Dict[str, List[str]] = {
    "car": CAR_PARTS, "motorcycle": MOTORCYCLE_PARTS, "scooter": SCOOTER_PARTS,
}
TWO_WHEELERS = {"motorcycle", "scooter"}

# Small, expensive, easily-missed parts: YOLO zooms in on them, VLM checks them.
LAMP_GLASS_PARTS = {"headlight", "taillight", "fog light", "side mirror", "rear view mirror",
                    "windshield"}

TYPE_ALIASES = {
    "car": "car", "4w": "car", "four wheeler": "car", "hatchback": "car", "sedan": "car", "suv": "car",
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "bike": "motorcycle", "2w": "motorcycle",
    "scooter": "scooter", "scooty": "scooter", "moped": "scooter",
}

# Known demo models -> type (so a policy that only says "Activa 6G" still works)
MODEL_TYPES = {
    "splendor": "motorcycle", "splendor plus": "motorcycle", "hf deluxe": "motorcycle",
    "shine": "motorcycle", "pulsar": "motorcycle", "platina": "motorcycle", "apache": "motorcycle",
    "classic 350": "motorcycle", "fz": "motorcycle", "passion": "motorcycle",
    "activa": "scooter", "activa 6g": "scooter", "jupiter": "scooter", "access": "scooter",
    "access 125": "scooter", "ntorq": "scooter", "dio": "scooter", "pleasure": "scooter",
}


def norm_type(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    return TYPE_ALIASES.get(" ".join(str(t).lower().replace("_", " ").split()))


def type_from_model(model: Optional[str]) -> Optional[str]:
    m = " ".join(str(model or "").lower().split())
    if m in MODEL_TYPES:
        return MODEL_TYPES[m]
    hits = [k for k in MODEL_TYPES if k in m]
    return MODEL_TYPES[max(hits, key=len)] if hits else None


def is_two_wheeler(t: Optional[str]) -> bool:
    return norm_type(t) in TWO_WHEELERS


def family(t: Optional[str]) -> Optional[str]:
    """car vs two_wheeler (DINO can confuse motorcycle and scooter, not car and bike)."""
    t = norm_type(t)
    return None if t is None else "two_wheeler" if t in TWO_WHEELERS else "car"
