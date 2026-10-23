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
CUE_MIN_CONF = 0.40      # context parts must be at least this confident to count

CONFUSABLE = [
    frozenset({"car hood", "car door"}),
    frozenset({"headlight", "taillight"}),
    frozenset({"front bumper", "rear bumper"}),
    frozenset({"fender", "car door"}),
    frozenset({"quarter panel", "car door"}),
    frozenset({"car hood", "trunk"}),
]
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


def context_flags(dets: List[Dict]) -> List[str]:
    """Scene geometry that CONTRADICTS a label marks it ambiguous (never relabels).  Returns notes."""
    notes = []
    cues = [d for d in dets if d["confidence"] >= CUE_MIN_CONF and not d.get("whole_frame")]
    front = [d for d in cues if d["label"] in FRONT_CUES]
    rear = [d for d in cues if d["label"] in REAR_CUES]
    handles = [d for d in cues if d["label"] == "door handle"]
    shields = [d for d in cues if d["label"] == "windshield"]

    def flag(d, alt, why):
        if d.get("label_ambiguous") and any(a["label"] == alt for a in d.get("label_alternatives", [])):
            return
        d["label_ambiguous"] = True
        d.setdefault("label_alternatives", [])
        if all(a["label"] != alt for a in d["label_alternatives"]):
            d["label_alternatives"].append({"label": alt, "score": None, "source": "scene context"})
        d["ambiguity_reason"] = why
        notes.append(f"'{d['label']}' {[int(v) for v in d['bbox']]} marked ambiguous: {why}")

    for d in dets:
        lab = d["label"]
        if lab == "headlight" and rear and not front:
            flag(d, "taillight", "only rear parts (" + ", ".join(sorted({r['label'] for r in rear})) + ") are visible")
        elif lab == "taillight" and front and not rear:
            flag(d, "headlight", "only front parts (" + ", ".join(sorted({f['label'] for f in front})) + ") are visible")
        elif lab == "car hood" and any(_inside(_center(h["bbox"]), d["bbox"]) for h in handles):
            flag(d, "car door", "a door handle lies inside the 'car hood' box")
        elif lab == "car door" and not any(_inside(_center(h["bbox"]), d["bbox"]) for h in handles):
            b = d["bbox"]
            h = b[3] - b[1]
            for w in shields:   # windshield sitting on the box's top edge = the hood, not a door
                wb = w["bbox"]
                overlap_x = max(0.0, min(b[2], wb[2]) - max(b[0], wb[0])) / max(b[2] - b[0], 1.0)
                if overlap_x >= 0.5 and abs(wb[3] - b[1]) <= 0.15 * h and (b[2] - b[0]) > 1.6 * h:
                    flag(d, "car hood", "a windshield sits on its top edge and it is wide and flat, like a hood")
                    break
    return notes


def recheck_decision(scores: Dict[str, float], margin: float = RECHECK_MARGIN) -> Optional[str]:
    """Winner of a two-name crop re-check, or None if it is still too close to call."""
    if len(scores) < 2:
        return None
    (a, sa), (b, sb) = sorted(scores.items(), key=lambda x: -x[1])[:2]
    return a if sa - sb >= margin else None


def apply_recheck(d: Dict, scores: Dict[str, float], margin: float = RECHECK_MARGIN) -> Optional[str]:
    """Record a re-check on the detection; resolve the label only on a clear win."""
    win = recheck_decision(scores, margin)
    d["recheck"] = {"scores": {k: round(v, 4) for k, v in scores.items()}, "winner": win, "margin": margin}
    if win is None:
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
