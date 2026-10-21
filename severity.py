"""
Auto-Settle AI - Stage 4: damage severity (uniform, explainable)
=================================================================

Mentor's question: "how do you make the VLM give UNIFORM severity?"
Our answer, in three layers:

1. The VLM never says "minor / moderate / severe".  It only answers simple,
   checkable FACT questions in a fixed JSON schema (is the paint broken? is the
   dent deep? is the lamp lens broken? ...).  Yes/no facts are far more stable
   than an opinion word.
2. Python RULES turn those facts (+ the YOLO mask area, a real measurement)
   into low / medium / high, and record WHY ("high: deep dent + sharp crease").
   Same facts -> same severity, every time.
3. VOTING: every damage is checked on 3 crops (tight / normal / wide).  The
   majority wins; if the views disagree the item is sent to a human instead of
   guessing.  Greedy decoding (temperature 0) + JSON validation + 1 retry.

A labelled sample set + severity_eval.py measures accuracy and consistency.

Without a VLM (--severity rules) the same module gives a mask-area based
severity, which is better than the old box-size guess in fuse.py.

Usage
-----
    # re-assess an existing result folder (no need to rerun DINO / YOLO):
    python severity.py results\\car\\fused.json car.jpg --vlm
    python severity.py results\\car\\fused.json car.jpg          # rules only
    # or as part of the full pipeline:
    python run_pipeline.py car.jpg --plate DL3CAB1234 --severity vlm

VLM needs:  pip install "transformers>=4.49" accelerate bitsandbytes
Model: Qwen/Qwen2.5-VL-3B-Instruct in 4-bit (~3 GB VRAM, fits an 8 GB RTX 4060
next to DINO + YOLO).  Downloads once (~7 GB) from Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

# =============================================================================
# 1. The fact schema - the ONLY thing the VLM is allowed to answer
# =============================================================================
BOOL = "bool"
FACT_SCHEMA: Dict[str, object] = {
    "damage_visible": BOOL,          # is there real damage (not a reflection / dirt)?
    "part_matches": BOOL,            # does the crop show the part we named?
    "paint_broken": BOOL,            # paint chipped / scraped through to primer or metal
    "material_torn": BOOL,           # torn, split, hole, or a piece missing
    "dent_depth": ("none", "shallow", "deep"),
    "sharp_crease": BOOL,            # a sharp fold line in the metal
    "lamp_lens_broken": BOOL,        # lamp lens cracked / shattered / missing
    "lamp_housing_broken": BOOL,     # lamp body / mounting broken
    "glass_shattered": BOOL,         # spider-web cracks or broken through
    "extent": ("small", "medium", "large"),   # share of the part that is damaged
    "image_quality": ("good", "poor"),
    "reflection_or_dirt_suspected": BOOL,
}

SYSTEM_PROMPT = (
    "You are a careful motor-insurance damage inspector. You only report what is "
    "clearly visible in the photo. You never guess costs or severity words. "
    "You answer with one JSON object and nothing else."
)

FIELD_HELP = """\
Answer every field:
- damage_visible: true only if real physical damage is visible (not a reflection, shadow, water or dirt).
- part_matches: true if the photo shows a {part} (or part of one).
- paint_broken: true if paint is chipped or scraped through so primer, metal or plastic shows.
- material_torn: true if the panel/plastic is torn, split, has a hole, or a piece is missing.
- dent_depth: "none" (no dent), "shallow" (only visible in reflections, panel shape intact), "deep" (panel shape clearly pushed in).
- sharp_crease: true if there is a sharp fold line in the metal (not a smooth dent).
- lamp_lens_broken: true if a lamp lens is cracked, shattered or missing. false if no lamp.
- lamp_housing_broken: true if a lamp body or its mounting is broken. false if no lamp.
- glass_shattered: true if glass shows spider-web cracks or is broken through. false if no glass.
- extent: how much of the {part} is damaged: "small" (under 10%), "medium" (10-40%), "large" (over 40%).
- image_quality: "poor" if too blurry, dark or far to judge, else "good".
- reflection_or_dirt_suspected: true if what looks like damage might be a reflection or dirt.

The detector thinks there is a "{damage}" on the {part}. The detector can be wrong - check yourself.

{focus}
Look at the photo and decide each value yourself. Replace every <...> with your answer.
Reply with ONLY this JSON:
{example}"""

EXAMPLE_JSON = json.dumps({
    "damage_visible": True, "part_matches": True, "paint_broken": False,
    "material_torn": False, "dent_depth": "none", "sharp_crease": False,
    "lamp_lens_broken": False, "lamp_housing_broken": False, "glass_shattered": False,
    "extent": "small", "image_quality": "good", "reflection_or_dirt_suspected": False,
})


# Blank template (NOT a filled example): small VLMs copy a filled example verbatim,
# which made every damage come back as "no dent, small, paint intact".
TEMPLATE_JSON = """{
  "damage_visible": <true|false>,
  "part_matches": <true|false>,
  "paint_broken": <true|false>,
  "material_torn": <true|false>,
  "dent_depth": "<none|shallow|deep>",
  "sharp_crease": <true|false>,
  "lamp_lens_broken": <true|false>,
  "lamp_housing_broken": <true|false>,
  "glass_shattered": <true|false>,
  "extent": "<small|medium|large>",
  "image_quality": "<good|poor>",
  "reflection_or_dirt_suspected": <true|false>
}"""

FOCUS = {
    "lamp broken": "Most important: look closely at the lamp. Is the lens cracked, shattered, or missing pieces? Is the lamp body broken?",
    "glass shatter": "Most important: is the glass cracked in a spider-web pattern or broken through, or is it just a reflection?",
    "dent": "Most important: is the panel pushed in? Shallow = only visible in reflections; deep = the panel shape is clearly changed.",
    "crack": "Most important: is the material split or torn, and how much of the part is affected?",
    "scratch": "Most important: is the paint scraped through to primer, metal or plastic, or only on the surface?",
}


def build_prompt(part: str, damage: str) -> str:
    return FIELD_HELP.format(part=part, damage=damage, example=TEMPLATE_JSON,
                             focus=FOCUS.get(damage.lower(), ""))


def looks_like_default(facts: Dict) -> bool:
    """The exact all-default answer is what a model copying a template produces.
    It is not trusted as evidence for a lamp/glass break claim."""
    return facts == json.loads(EXAMPLE_JSON)


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "1"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "no", "0"):
        return False
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    raise ValueError(f"not a boolean: {v!r}")


def parse_facts(text: str) -> Dict:
    """Pull the JSON object out of the VLM reply and validate it against FACT_SCHEMA.
    Raises ValueError if anything is missing or out of range - never silently guesses."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    raw = json.loads(m.group(0))
    facts = {}
    for key, kind in FACT_SCHEMA.items():
        if key not in raw:
            raise ValueError(f"missing field {key}")
        v = raw[key]
        if kind == BOOL:
            facts[key] = _to_bool(v)
        else:
            v = str(v).strip().lower()
            if v not in kind:
                raise ValueError(f"{key}={v!r} not in {kind}")
            facts[key] = v
    bad = contradictions(facts)
    if bad:
        raise ValueError("contradictory answer: " + "; ".join(bad))
    return facts


def contradictions(f: Dict) -> List[str]:
    """Answers that cannot all be true at once.  Such a reply is not evidence of anything,
    so it is treated as invalid (retry once, then 'failed') - never as damage or no damage."""
    out = []
    if not f["damage_visible"]:
        positive = [k for k in ("material_torn", "glass_shattered", "lamp_lens_broken",
                                "lamp_housing_broken", "sharp_crease") if f[k]]
        if f["dent_depth"] == "deep":
            positive.append("dent_depth=deep")
        if positive:
            out.append(f"damage_visible=false but {', '.join(positive)}")
    if f["sharp_crease"] and f["dent_depth"] == "none":
        out.append("sharp_crease=true but dent_depth=none")
    return out


def extent_from_area(area: float) -> str:
    """Same cut-offs the prompt gives the VLM (under 10% / 10-40% / over 40%)."""
    return "small" if area < 0.10 else "medium" if area < 0.40 else "large"


# =============================================================================
# 2. Rules: facts (+ mask area) -> severity.  Deterministic and explainable.
# =============================================================================
LEVELS = ["low", "medium", "high"]
CANON = {"dent": "dent", "scratch": "scratch", "crack": "crack", "lamp broken": "lamp",
         "glass shatter": "glass", "tire flat": "tyre"}


def damage_area_ratio(m: Dict) -> Optional[float]:
    """Damaged pixels / part box pixels.  Uses the YOLO segmentation mask when
    present (real area), else the damage box (over-estimates)."""
    pb = m.get("part_bbox")
    if not pb:
        return None
    part_area = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
    if m.get("mask_area_px"):
        return min(1.0, m["mask_area_px"] / part_area)
    return m.get("part_coverage")


def severity_from_facts(damage: str, f: Dict, area: Optional[float]):
    """Returns (severity or None, reasons, needs_review)."""
    d = CANON.get(damage.lower(), damage.lower())
    if not f["damage_visible"]:
        return None, ["VLM: no real damage visible"], False
    reasons, review = [], False
    if f["image_quality"] == "poor":
        reasons.append("photo quality poor")
        review = True
    if f["reflection_or_dirt_suspected"]:
        reasons.append("may be reflection / dirt")
        review = True

    if d == "dent":
        if f["material_torn"] or f["dent_depth"] == "deep" or f["sharp_crease"]:
            sev = "high"
            reasons.append("deep dent / sharp crease / torn metal")
        elif f["dent_depth"] == "shallow" and f["extent"] == "small" and not f["paint_broken"]:
            sev = "low"
            reasons.append("shallow small dent, paint intact (PDR candidate)")
        else:
            sev = "medium"
            reasons.append(f"dent {f['dent_depth']}, extent {f['extent']}"
                           + (", paint broken" if f["paint_broken"] else ""))
    elif d == "scratch":
        if f["material_torn"]:
            sev = "high"
            reasons.append("scratch has torn the material")
        elif f["paint_broken"] or f["extent"] == "large":
            sev = "medium"
            reasons.append("paint broken through" if f["paint_broken"] else "large scratched area")
        else:
            sev = "low"
            reasons.append("surface scratch, paint layer intact")
    elif d == "crack":
        if f["material_torn"] or f["extent"] == "large":
            sev = "high"
            reasons.append("crack split the part / large crack")
        elif f["extent"] == "medium":
            sev = "medium"
            reasons.append("medium crack")
        else:
            sev = "low"
            reasons.append("small crack")
    elif d == "lamp":
        if f["lamp_housing_broken"] or f["lamp_lens_broken"]:
            sev = "high"
            reasons.append("lamp " + ("housing" if f["lamp_housing_broken"] else "lens") + " broken")
        else:
            sev = "medium"
            reasons.append("lamp scuffed but lens/housing not broken")
            review = True
    elif d == "glass":
        sev = "high" if f["glass_shattered"] else "medium"
        reasons.append("glass shattered" if f["glass_shattered"] else "glass chipped, not shattered")
    elif d == "tyre":
        sev = "high"
        reasons.append("flat tyre")
    else:
        sev = "medium"
        reasons.append(f"unknown damage type {damage}")
        review = True

    # Second, independent measurement: real damaged area.  It can only RAISE
    # low -> medium (a big area is never 'low'), never invent a 'high'.
    if area is not None and area >= 0.40 and sev == "low":
        sev = "medium"
        reasons.append(f"raised to medium: damage covers {area:.0%} of the part")
    if not f["part_matches"]:
        reasons.append("VLM says crop does not show the named part")
        review = True
    return sev, reasons, review


def severity_from_area(damage: str, area: Optional[float]):
    """No-VLM fallback: damage type + measured area only."""
    d = CANON.get(damage.lower(), damage.lower())
    if d in ("lamp", "glass", "tyre"):
        return "high", [f"{damage}: always high"]
    if area is None:
        return "medium", ["area unknown - default medium"]
    cuts = {"dent": (0.10, 0.30), "crack": (0.10, 0.30), "scratch": (0.15, 1.01)}.get(d, (0.15, 0.40))
    sev = "low" if area < cuts[0] else "medium" if area < cuts[1] else "high"
    return sev, [f"{damage} covers {area:.0%} of the part"]


# =============================================================================
# 3. VLM backends
# =============================================================================
class QwenVL:
    """Qwen2.5-VL, greedy decoding (deterministic), 4-bit by default."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", four_bit: bool = True):
        import torch
        from transformers import AutoProcessor
        try:   # transformers >= 4.57: one class for Qwen2.5-VL and Qwen3-VL
            from transformers import AutoModelForImageTextToText as VLMClass
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLMClass

        kw = {"torch_dtype": torch.float16, "device_map": "auto"}
        if four_bit:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            except ImportError:
                print("[warn] bitsandbytes not installed - loading in fp16 (needs ~8 GB VRAM). "
                      "pip install bitsandbytes", file=sys.stderr)
        self.model = VLMClass.from_pretrained(model_id, **kw).eval()
        # cap image tokens: enough detail for a crop, keeps VRAM + time low
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28,
                                                       max_pixels=768 * 28 * 28)
        self.name = model_id
        self.torch = torch

    def ask(self, image: Image.Image, prompt: str) -> str:
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image.convert("RGB")],
                                return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0]


class MockVLM:
    """Offline stand-in for testing the plumbing (no GPU).  Returns fixed facts."""
    name = "mock"

    def __init__(self, facts: Optional[Dict] = None, replies: Optional[List[str]] = None):
        self.facts = facts or json.loads(EXAMPLE_JSON)
        self.replies = list(replies or [])

    def ask(self, image, prompt):
        return self.replies.pop(0) if self.replies else json.dumps(self.facts)


# =============================================================================
# 4. Assessor: 3 views, vote, fall back safely
# =============================================================================
VIEW_PADS = (0.10, 0.30, 0.60)   # tight, normal, wide context
MIN_VIEW_SIDE = 224              # px: a smaller crop of a small dent looks "poor quality" to the VLM
BREAK_DETECTOR_CONF = 0.75       # see SeverityAssessor.assess: lamp / glass break rule


def views_for(image: Image.Image, bbox) -> List[Image.Image]:
    W, H = image.size
    x1, y1, x2, y2 = bbox
    if (x2 - x1) * (y2 - y1) >= 0.9 * W * H:   # input is already a crop -> zoom in instead
        out = []
        for keep in (0.75, 0.90, 1.0):
            dx, dy = W * (1 - keep) / 2, H * (1 - keep) / 2
            out.append(image.crop((int(dx), int(dy), int(W - dx), int(H - dy))))
        return out
    out = []
    for pad in VIEW_PADS:
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        a, b, c, d = x1 - px, y1 - py, x2 + px, y2 + py
        # Small damage: grow the crop around its centre to at least MIN_VIEW_SIDE, so the
        # tight view is not a blurry thumbnail.  The damage stays in the middle of every view.
        if c - a < MIN_VIEW_SIDE:
            g = (MIN_VIEW_SIDE - (c - a)) / 2
            a, c = a - g, c + g
        if d - b < MIN_VIEW_SIDE:
            g = (MIN_VIEW_SIDE - (d - b)) / 2
            b, d = b - g, d + g
        out.append(image.crop((max(0, int(a)), max(0, int(b)), min(W, int(c)), min(H, int(d)))))
    return out


class SeverityAssessor:
    def __init__(self, vlm=None):
        self.vlm = vlm   # None -> rules on measured area only

    def _facts(self, img, prompt):
        """One VLM call + strict validation; at most one retry with a stricter prompt.
        Returns (facts or None, error or None, raw replies).  A copied template counts as
        invalid: it is what a model produces when it did not look at the photo."""
        raws = []
        ask = prompt
        err = None
        for attempt in range(2):
            try:
                reply = self.vlm.ask(img, ask)
            except Exception as e:  # noqa: BLE001 - model crash / OOM / timeout = no evidence
                return None, f"VLM call failed: {str(e)[:120]}", raws
            raws.append(str(reply)[:600])
            try:
                # A copied template ("<true|false>") fails parse_facts: placeholders are not
                # valid values.  An all-plain answer is NOT rejected - "small scratch, paint
                # intact" legitimately looks like that.
                return parse_facts(reply), None, raws
            except (ValueError, json.JSONDecodeError) as e:
                err = str(e)
                ask = (prompt + f"\n\nYour last answer was invalid ({err}). Look at the photo again. "
                       "Reply with ONLY the JSON object, every field filled with your own answer.")
        return None, err, raws

    def assess(self, image: Image.Image, m: Dict) -> Dict:
        """m: one fused 'matched' entry (part, damage, damage_bbox, part_bbox, mask_area_px...).

        Result field vlm_verification:
          confirmed    - a majority of VALID views see real damage -> severity from rules
          rejected     - a majority of VALID views see no damage   -> not priced
          inconclusive - valid views split evenly                 -> human review
          failed       - fewer than 2 valid views                 -> human review
        Invalid / failed views never count as a vote either way."""
        area = damage_area_ratio(m)
        base_sev, base_why = severity_from_area(m["damage"], area)
        if self.vlm is None:
            return {"severity": base_sev, "severity_source": "rules_area",
                    "severity_reasons": base_why, "damage_area_ratio": area,
                    "vlm_verification": "not_run"}

        prompt = build_prompt(m["part"], m["damage"])
        votes = []
        for i, view in enumerate(views_for(image, m["damage_bbox"])):
            facts, err, raws = self._facts(view, prompt)
            if facts is None:
                votes.append({"view": i, "error": err, "raw": raws})
                continue
            votes.append({"view": i, "visible": facts["damage_visible"], "facts": facts, "raw": raws})

        good = [v for v in votes if "error" not in v]
        res = {"severity_votes": votes, "damage_area_ratio": area, "vlm_model": self.vlm.name,
               "vlm_valid_views": len(good), "vlm_views": len(votes)}
        if len(good) < 2:   # not enough valid evidence: never confirmed, never rejected
            res.update(severity=base_sev, severity_source="rules_area", vlm_confirmed=None,
                       vlm_verification="failed",
                       severity_reasons=base_why + [f"VLM gave only {len(good)} valid answer(s) out of "
                                                    f"{len(votes)} - area rules used, human to check"],
                       severity_review=True)
            return res

        visible = Counter(v["visible"] for v in good)
        res["vlm_agreement"] = round(max(visible.values()) / len(good), 2)   # share of VALID views
        if visible[True] == visible[False]:
            res.update(severity=base_sev, severity_source="rules_area", vlm_confirmed=None,
                       vlm_verification="inconclusive", severity_review=True,
                       severity_reasons=base_why + ["VLM views split on whether the damage is real"])
            return res
        if visible[True] < visible[False]:
            res.update(vlm_confirmed=False, vlm_verification="rejected", severity=None,
                       severity_source="vlm_rules", severity_review=False,
                       severity_reasons=[f"{visible[False]} of {len(good)} valid views: no real damage "
                                         "(reflection / dirt / misdetection)"])
            return res

        # Confirmed.  'extent' depends on how tightly a crop is cut (a tight crop makes any
        # damage look large), which made views disagree.  The measured mask area is the same
        # for every view, so it replaces the VLM's guess when we have it; otherwise every view
        # uses the extent seen in the WIDEST valid view (the one that shows the whole part).
        seen = [v for v in good if v["visible"]]
        if m.get("mask_area_px") and area is not None:
            ext, ext_src = extent_from_area(area), f"measured mask area {area:.0%} of the part"
        else:
            ext, ext_src = seen[-1]["facts"]["extent"], "widest valid view"
        for v in seen:
            f = {**v["facts"], "extent": ext}
            v["severity"], v["reasons"], v["review"] = severity_from_facts(m["damage"], f, area)
        sevs = Counter(v["severity"] for v in seen)
        top, n = sevs.most_common(1)[0]
        if n < 2:   # no two views agree -> middle value, human decides
            top = sorted((v["severity"] for v in seen), key=LEVELS.index)[len(seen) // 2]
        # Quality / reflection / wrong-part flags follow the same majority rule as the vote:
        # one view out of three saying "poor photo" (usually the tightest crop) is not enough.
        flagged = sum(1 for v in seen if v["review"])
        winner = next((v for v in seen if v["severity"] == top and not v["review"]),
                      next(v for v in seen if v["severity"] == top))
        review = n < 2 or flagged * 2 > len(seen)
        res.update(vlm_confirmed=True, vlm_verification="confirmed", vlm_severity=top, severity=top,
                   severity_source="vlm_rules", severity_reasons=winner["reasons"] + [f"extent from {ext_src}"],
                   severity_facts={**winner["facts"], "extent": ext}, severity_review=review,
                   vlm_severity_agreement=round(n / len(seen), 2))
        if n < 2:
            res["severity_reasons"] = res["severity_reasons"] + ["views disagree on severity"]

        # A break is local: a tight crop can cut the cracked part of a lens off. For lamp /
        # glass damage that the detector is already sure about, one valid view that clearly
        # SEES the break is enough; without any such view the item stays as voted.
        kind = CANON.get(m["damage"].lower())
        if kind in ("lamp", "glass") and (m.get("damage_confidence") or 0) >= BREAK_DETECTOR_CONF:
            key = ("lamp_lens_broken", "lamp_housing_broken") if kind == "lamp" else ("glass_shattered",)
            br = [v for v in seen if any(v["facts"][k] for k in key)
                  and not v["facts"]["reflection_or_dirt_suspected"]]
            if br:
                res.update(vlm_severity="high", severity="high", severity_review=False,
                           severity_facts=br[0]["facts"],
                           severity_reasons=[f"break seen in {len(br)} of {len(good)} valid views and detector "
                                             f"confidence {m['damage_confidence']:.2f}"])
        return res

    def check_intact(self, image: Image.Image, ins: Dict) -> Dict:
        """fused['inspect'] items: lamps / glass where YOLO found nothing.  The VLM
        only FLAGS a missed break for a human - VLM-only finds are never priced."""
        if self.vlm is None:
            return {}
        part = ins["part"]
        lamp = part in ("headlight", "taillight", "fog light")
        glass = part in ("windshield", "side mirror", "rear view mirror")
        label = "lamp broken" if lamp else "glass shatter" if glass else "dent or scratch"
        facts, err, raws = self._facts(views_for(image, ins["bbox"])[1], build_prompt(part, label))
        if facts is None:
            return {"vlm_error": err, "vlm_raw": raws}
        if not facts["damage_visible"] or facts["reflection_or_dirt_suspected"]:
            return {"vlm_found_damage": False, "vlm_facts": facts}
        # what kind of damage do the facts describe? (two-wheeler panels: any damage counts)
        if facts["lamp_lens_broken"] or facts["lamp_housing_broken"]:
            dmg = "lamp broken"
        elif facts["glass_shattered"]:
            dmg = "glass shatter"
        elif lamp or glass:
            return {"vlm_found_damage": False, "vlm_facts": facts}   # lamp/glass: only breaks matter
        elif facts["material_torn"]:
            dmg = "crack"
        elif facts["dent_depth"] != "none" or facts["sharp_crease"]:
            dmg = "dent"
        elif facts["paint_broken"]:
            dmg = "scratch"
        else:
            return {"vlm_found_damage": False, "vlm_facts": facts}
        sev, why, _ = severity_from_facts(dmg, facts, None)
        return {"vlm_found_damage": True, "vlm_damage": dmg, "vlm_severity": sev,
                "vlm_reasons": why, "vlm_facts": facts, "vlm_raw": raws}


def assess_fused(fused: Dict, image: Image.Image, assessor: SeverityAssessor) -> Dict:
    """Adds severity fields to every matched damage (in place) and returns fused."""
    for m in fused.get("matched", []):
        m.update(assessor.assess(image, m))
    for ins in fused.get("inspect", []):
        ins.update(assessor.check_intact(image, ins))
    fused["vlm_missed_damage"] = [i["part"] for i in fused.get("inspect", []) if i.get("vlm_found_damage")]
    return fused


DEFAULT_VLM = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_vlm(kind: str, model_id: Optional[str] = None):
    """model: argument > AUTOSETTLE_VLM_MODEL env var > Qwen2.5-VL-3B.
    Choices: Qwen/Qwen2.5-VL-3B-Instruct (default), Qwen/Qwen3-VL-4B-Instruct (needs transformers>=4.57)."""
    if kind in ("none", "rules", None):
        return None
    if kind == "mock":
        return MockVLM()
    import os
    mid = model_id if model_id and model_id != DEFAULT_VLM else os.environ.get("AUTOSETTLE_VLM_MODEL") or model_id or DEFAULT_VLM
    print(f"VLM: {mid}")
    return QwenVL(mid)


def print_table(fused: Dict) -> None:
    print(f"\n{'part':<14}{'damage':<14}{'severity':<9}{'src':<11}{'agree':<7}{'review':<7}why")
    for m in fused.get("matched", []):
        sev = m.get("severity") or ("NOT DAMAGE" if m.get("vlm_confirmed") is False else "-")
        print(f"{m['part']:<14}{m['damage']:<14}{sev:<9}{m.get('severity_source', '-'):<11}"
              f"{str(m.get('vlm_agreement', '-')):<7}{'YES' if m.get('severity_review') else '':<7}"
              f"{'; '.join(m.get('severity_reasons', []))}")
    for i in fused.get("inspect", []):
        if i.get("vlm_found_damage"):
            print(f"{i['part']:<14}{i.get('vlm_damage', '?'):<14}{i.get('vlm_severity') or '-':<9}"
                  f"{'VLM only':<11}{'':<7}{'YES':<7}detector missed it -> human check")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 4 severity on an existing fused.json")
    ap.add_argument("fused_json")
    ap.add_argument("image")
    ap.add_argument("--vlm", action="store_true", help="use Qwen2.5-VL (else area rules only)")
    ap.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--mock", action="store_true", help="fake VLM for plumbing tests")
    a = ap.parse_args(argv)

    from yolo_damage import load_image   # same EXIF handling as the detectors
    fused = json.loads(Path(a.fused_json).read_text())
    vlm = load_vlm("mock" if a.mock else "qwen" if a.vlm else "none", a.vlm_model)
    assess_fused(fused, load_image(a.image), SeverityAssessor(vlm))
    Path(a.fused_json).write_text(json.dumps(fused, indent=2))
    print_table(fused)
    print(f"\nupdated {a.fused_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
