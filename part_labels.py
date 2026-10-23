"""
Auto-Settle AI - part-label decisions for Grounding DINO (numpy only, testable without a GPU)
=============================================================================================

WHY.  The original post-processing let `post_process_grounded_object_detection` pick each
box's label from the prompt tokens above `text_threshold`, then mapped the phrase to a part
with substring matching.  Several part names share tokens:

    car door  / car hood          share "car"
    headlight / taillight         share "##light"
    front bumper / rear bumper    share "bumper"
    car door  / door handle       share "door"

When only the shared token clears the threshold, the phrase is "car" or "light", and the
substring match breaks the tie by LIST ORDER - a hood can silently become a door with no
record that it was a guess.

WHAT THIS DOES (deterministic, documented, no new model):
  1. score_labels(): every candidate label gets a score from the box's token probabilities,
     using the tokens that DISTINGUISH it ("hood", "door", "head", "tail" ...).  The best
     label wins on evidence, not list order, and the runner-up is kept.
  2. Confusable pairs whose scores are within LABEL_MARGIN are marked label_ambiguous,
     with the alternative and both scores.  Nothing is silently relabelled.
  3. cross_label_nms(): the old cross-label NMS dropped the losing label of an overlapping
     box without trace; now a confusable loser is kept as an alternative on the winner.
  4. context_flags(): simple scene geometry that can CONTRADICT a lamp / hood-door label
     (rear-only parts around a "headlight", a door handle inside a "car hood" ...) adds an
     ambiguity flag - it never relabels on its own.
  5. recheck_decision(): an ambiguous box may be re-run by the caller on a crop with only the
     two candidate names; the label is changed only if one wins by RECHECK_MARGIN, and the
     change is recorded (label_resolution / refined_from).

An ambiguous part that a damage lands on is reported by fuse.py as part_uncertain, so
pricing sends it to a person (PART_LOCATION_UNCLEAR) instead of pricing a guessed panel.

Accuracy is NOT claimed here: see tools/eval_dino_parts.py for the human-labelled comparison.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LABEL_MARGIN = 0.05      # confusable labels closer than this (token-probability units) -> ambiguous
RECHECK_MARGIN = 0.10    # a crop re-check must win by this much to settle the label
MAX_RECHECKS = 3         # per photo: extra DINO passes are sequential and cost ~0.1-0.3 s each
CUE_MIN_CONF = 0.35      # = the DINO box threshold: every kept part counts as scene context

CONFUSABLE = [
    frozenset({"car hood", "car door"}),
    frozenset({"headlight", "taillight"}),
    frozenset({"front bumper", "rear bumper"}),
    frozenset({"fender", "car door"}),
    frozenset({"quarter panel", "car door"}),
    frozenset({"car hood", "trunk"}),
    frozenset({"car hood", "fender"}),
    frozenset({"windshield", "side window"}),
]
# Strong orientation cues (a car's rear has no grille / hood / fog lamps; its front has no boot).
# Bumpers and lamps are NOT used as cues - they are exactly the labels being checked.
FRONT_STRONG = {"grille": 2, "car hood": 1, "fog light": 1, "front bumper": 1}
REAR_STRONG = {"trunk": 2, "rear bumper": 1}
SINGULAR = {"car hood": "fender", "trunk": "quarter panel", "windshield": "side window"}
FRONT_CUES = {"front bumper", "grille", "car hood", "fog light"}
REAR_CUES = {"rear bumper", "trunk"}


def confusable(a: str, b: str) -> bool:
    return frozenset({a, b}) in CONFUSABLE


# --------------------------------------------------------------------------- tokens
def label_char_spans(prompt: str, labels: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    """Character span of each label inside 'a . b . c .' (build_prompt format)."""
    spans, pos = {}, 0
    for l in labels:
        i = prompt.find(l, pos)
        if i < 0:
            raise ValueError(f"label '{l}' not found in prompt")
        spans[l] = (i, i + len(l))
        pos = i + len(l)
    return spans


def label_token_ids(offsets: Sequence[Tuple[int, int]], input_ids: Sequence[int], prompt: str,
                    labels: Sequence[str]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """(token positions of each label, its DISTINCTIVE positions).  A token is distinctive for a
    label if no other label contains the same token id ('door' is not distinctive for 'car door'
    because 'door handle' has it too; then the label falls back to all of its tokens)."""
    spans = label_char_spans(prompt, labels)
    pos: Dict[str, List[int]] = {l: [] for l in labels}
    for t, (a, b) in enumerate(offsets):
        if b <= a:
            continue                                    # special tokens / padding
        for l, (s, e) in spans.items():
            if a >= s and b <= e:
                pos[l].append(t)
                break
    ids_of = {l: {int(input_ids[t]) for t in ts} for l, ts in pos.items()}
    distinct = {}
    for l, ts in pos.items():
        others = set().union(*(ids_of[o] for o in labels if o != l)) if len(labels) > 1 else set()
        distinct[l] = [t for t in ts if int(input_ids[t]) not in others]
    return pos, distinct


def score_labels(token_probs: np.ndarray, pos: Dict[str, List[int]],
                 distinct: Dict[str, List[int]]) -> List[Tuple[str, float]]:
    """[(label, score)] best first.  score = mean probability over the label's distinctive tokens
    (all its tokens if it has none).  A label with no tokens scores 0."""
    out = []
    for l, ts in pos.items():
        use = distinct.get(l) or ts
        out.append((l, float(np.mean(token_probs[use])) if use else 0.0))
    out.sort(key=lambda x: -x[1])
    return out


def decide(ranked: List[Tuple[str, float]], margin: float = LABEL_MARGIN) -> Dict:
    """Label + honest ambiguity for one box from score_labels() output."""
    best, s1 = ranked[0]
    d = {"label": best, "label_score": round(s1, 4),
         "label_scores": [{"label": l, "score": round(s, 4)} for l, s in ranked[:3]],
         "label_ambiguous": False, "label_alternatives": [], "ambiguity_reason": None}
    for l, s in ranked[1:]:
        if confusable(best, l) and s1 - s < margin:
            d["label_ambiguous"] = True
            d["label_alternatives"].append({"label": l, "score": round(s, 4)})
            d["ambiguity_reason"] = f"'{best}' and '{l}' scored almost the same ({s1:.2f} vs {s:.2f})"
    return d


# --------------------------------------------------------------------------- geometry
def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(_area(a) + _area(b) - inter, 1e-6)


def _inside(pt, b) -> bool:
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def cross_label_nms(dets: List[Dict], iou_thr: float) -> List[Dict]:
    """Same region with two labels: keep the higher-confidence box.  A CONFUSABLE loser is kept
    as an alternative on the winner (and makes it ambiguous if the scores are close) instead of
    disappearing."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["confidence"])
    kept: List[Dict] = []
    for i in order:
        d = dets[i]
        winner = next((k for k in kept if iou(k["bbox"], d["bbox"]) >= iou_thr), None)
        if winner is None:
            kept.append(d)
            continue
        if winner["label"] != d["label"] and confusable(winner["label"], d["label"]):
            alts = winner.setdefault("label_alternatives", [])
            if all(a["label"] != d["label"] for a in alts):
                alts.append({"label": d["label"], "score": round(float(d.get("label_score", d["confidence"])), 4),
                             "source": "overlapping box"})
            gap = float(winner["confidence"]) - float(d["confidence"])
            if gap < LABEL_MARGIN:
                winner["label_ambiguous"] = True
                winner["ambiguity_reason"] = (f"an overlapping box was labelled '{d['label']}' with almost the same "
                                              f"confidence ({winner['confidence']:.2f} vs {d['confidence']:.2f})")
    return kept


def scene_orientation(dets: List[Dict]) -> Tuple[Optional[str], List[str]]:
    """'front' / 'rear' / None from confident, non-whole-frame parts.  Needs 2+ points of cues on
    one side and NONE on the other, so a mixed or unclear photo stays None (no fixes applied)."""
    cues = {d["label"] for d in dets if d["confidence"] >= CUE_MIN_CONF and not d.get("whole_frame")}
    f = sum(w for l, w in FRONT_STRONG.items() if l in cues)
    r = sum(w for l, w in REAR_STRONG.items() if l in cues)
    if f >= 2 and r == 0:
        return "front", sorted(cues & set(FRONT_STRONG))
    if r >= 2 and f == 0:
        return "rear", sorted(cues & set(REAR_STRONG))
    return None, []


def _flag(d: Dict, alt: str, why: str, notes: List[str], forbid: Optional[str] = None) -> None:
    d["label_ambiguous"] = True
    d.setdefault("label_alternatives", [])
    if all(a["label"] != alt for a in d["label_alternatives"]):
        d["label_alternatives"].append({"label": alt, "score": None, "source": "scene context"})
    d["ambiguity_reason"] = why
    if forbid:
        d.setdefault("context_forbids", [])
        if forbid not in d["context_forbids"]:
            d["context_forbids"].append(forbid)
    notes.append(f"'{d['label']}' {[int(v) for v in d['bbox']]} marked ambiguous: {why}")


def _relabel(d: Dict, new: str, why: str, notes: List[str]) -> None:
    notes.append(f"relabelled '{d['label']}' {[int(v) for v in d['bbox']]} -> '{new}': {why}")
    d["refined_from"] = d["label"]
    d["label"] = new
    d["label_resolution"] = "scene orientation"
    d["label_resolution_reason"] = why
    d["label_ambiguous"] = False
    d["label_alternatives"] = [a for a in d.get("label_alternatives", []) if a["label"] != new]
    d["context_forbids"] = [d["refined_from"]]


def context_flags(dets: List[Dict]) -> List[str]:
    """Scene geometry.  Returns notes.  Two kinds of action, both recorded on the detection:
      * RELABEL - only front/rear pairs, and only when the photo's orientation is unambiguous
        (e.g. a 'taillight' in a photo that shows the grille and hood and no boot).  A lamp or
        bumper cannot be on the far end of the car from the one in view.
      * FLAG ambiguous (never relabel) - a 'car door' with no handle in a front/rear view or with
        the windshield on its top edge; a 'car hood' with a door handle inside or sitting on a
        wheel; a second, separate box of a part a car has only one of."""
    notes: List[str] = []
    view, cues = scene_orientation(dets)
    if view:
        notes.append(f"scene orientation: {view} of the vehicle ({', '.join(cues)} visible)")
    swap = {"front": {"taillight": "headlight", "rear bumper": "front bumper"},
            "rear": {"headlight": "taillight", "front bumper": "rear bumper"}}.get(view, {})
    conf_parts = [d for d in dets if d["confidence"] >= CUE_MIN_CONF and not d.get("whole_frame")]
    handles = [d for d in conf_parts if d["label"] == "door handle"]
    shields = [d for d in conf_parts if d["label"] == "windshield"]
    wheels = [d for d in conf_parts if d["label"] == "wheel"]

    for d in dets:
        lab = d["label"]
        if lab in swap:
            _relabel(d, swap[lab], f"only the {view} of the car is visible ({', '.join(cues)})", notes)
            continue
        b = d["bbox"]
        h = b[3] - b[1]
        has_handle = any(_inside(_center(x["bbox"]), b) for x in handles)
        if lab == "car hood" and has_handle:
            _flag(d, "car door", "a door handle lies inside the 'car hood' box", notes)
        elif lab == "car hood" and any(b[0] <= _center(w["bbox"])[0] <= b[2] and w["bbox"][1] >= b[1] + 0.5 * h
                                       and w["bbox"][1] <= b[3] + 0.1 * h for w in wheels):
            _flag(d, "fender", "it sits directly on a wheel, where the fender is", notes)
        elif lab == "car door" and not has_handle:
            windshield_on_top = False
            for w in shields:
                wb = w["bbox"]
                overlap_x = max(0.0, min(b[2], wb[2]) - max(b[0], wb[0])) / max(b[2] - b[0], 1.0)
                windshield_on_top |= overlap_x >= 0.5 and abs(wb[3] - b[1]) <= 0.15 * h
            alt = "car hood" if (b[2] - b[0]) > 1.6 * h else "fender"
            if windshield_on_top:
                _flag(d, alt, "the windshield sits on its top edge and no door handle is inside", notes,
                      forbid="car door")
            elif view:
                _flag(d, alt, f"a {view} view of the car with no door handle inside this box", notes,
                      forbid="car door")
    for lab, alt in SINGULAR.items():   # a car has one hood / boot / windshield
        boxes = sorted((d for d in dets if d["label"] == lab), key=lambda d: -d["confidence"])
        for d in boxes[1:]:
            if all(iou(d["bbox"], k["bbox"]) < 0.3 for k in boxes[:1]) and not d.get("label_ambiguous"):
                _flag(d, alt, f"a second, separate '{lab}' box (a car has only one) with lower confidence", notes)
    return notes


def recheck_decision(scores: Dict[str, float], margin: float = RECHECK_MARGIN) -> Optional[str]:
    """Winner of a two-name crop re-check, or None if it is still too close to call."""
    if len(scores) < 2:
        return None
    (a, sa), (b, sb) = sorted(scores.items(), key=lambda x: -x[1])[:2]
    return a if sa - sb >= margin else None


def apply_recheck(d: Dict, scores: Dict[str, float], margin: float = RECHECK_MARGIN) -> Optional[str]:
    """Record a re-check on the detection; resolve the label only on a clear win, and never to a
    label the scene geometry ruled out (a crop of an orange front corner lamp can look like a
    taillight to the model - the photo's orientation says it is not)."""
    win = recheck_decision(scores, margin)
    blocked = win is not None and win in d.get("context_forbids", [])
    d["recheck"] = {"scores": {k: round(v, 4) for k, v in scores.items()}, "winner": win, "margin": margin,
                    **({"overruled_by_scene": True} if blocked else {})}
    if win is None or blocked:
        return None
    if win != d["label"]:
        d["refined_from"] = d["label"]
        d["label"] = win
    d["label_ambiguous"] = False
    d["label_resolution"] = "targeted recheck"
    return win


def ambiguous_pairs(dets: Iterable[Dict]) -> List[Tuple[Dict, List[str]]]:
    """Detections still ambiguous and the candidate names to re-check them with."""
    out = []
    for d in dets:
        if d.get("label_ambiguous"):
            names = [d["label"]] + [a["label"] for a in d.get("label_alternatives", []) if a["label"] != d["label"]]
            out.append((d, list(dict.fromkeys(names))[:2]))
    return out
