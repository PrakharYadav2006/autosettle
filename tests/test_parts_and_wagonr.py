"""Part-label post-processing (part_labels.py), the DINO review scorer (tools/eval_dino_parts.py)
and the WagonR price / policy data.  No GPU: DINO's token probabilities are hand-made arrays, so
these tests check the DECISION RULES, not detector accuracy."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

import part_labels as pl
import pricing_rules as pr
from test_autosettle import _defaults, api, item  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_dino_parts as ev  # noqa: E402

PROMPT = "car door . car hood . door handle . headlight . taillight ."
# a BERT-style tokenisation of PROMPT: (text, char offsets); same word -> same id
TOKS = [("[CLS]", (0, 0)), ("car", (0, 3)), ("door", (4, 8)), (".", (9, 10)), ("car", (11, 14)), ("hood", (15, 19)),
        (".", (20, 21)), ("door", (22, 26)), ("handle", (27, 33)), (".", (34, 35)), ("head", (36, 40)),
        ("##light", (40, 45)), (".", (46, 47)), ("tail", (48, 52)), ("##light", (52, 57)), (".", (58, 59)),
        ("[SEP]", (0, 0))]
VOCAB = {}
IDS = [VOCAB.setdefault(t, len(VOCAB) + 100) for t, _ in TOKS]
OFFS = [o for _, o in TOKS]
LABELS = ["car door", "car hood", "door handle", "headlight", "taillight"]


def probs(**tok):   # token text -> probability (applies to every position with that text)
    return np.array([tok.get(t, 0.02) for t, _ in TOKS], dtype=float)


class TestTokenLabels:
    def test_distinctive_tokens(self):
        pos, dist = pl.label_token_ids(OFFS, IDS, PROMPT, LABELS)
        name = lambda ts: [TOKS[t][0] for t in ts]  # noqa: E731
        assert name(pos["car door"]) == ["car", "door"] and name(dist["car door"]) == []   # both shared
        assert name(dist["car hood"]) == ["hood"] and name(dist["headlight"]) == ["head"]
        assert name(dist["taillight"]) == ["tail"] and name(dist["door handle"]) == ["handle"]

    def test_hood_box_with_strong_shared_car_token_is_a_hood(self):
        """The old phrase path could return just 'car' and resolve it by list order to 'car door'.
        Token evidence: 'hood' is high, 'door' is low -> car hood, not ambiguous."""
        pos, dist = pl.label_token_ids(OFFS, IDS, PROMPT, LABELS)
        d = pl.decide(pl.score_labels(probs(car=0.6, hood=0.5, door=0.05), pos, dist))
        assert d["label"] == "car hood" and not d["label_ambiguous"]

    def test_lamp_decided_by_head_vs_tail(self):
        pos, dist = pl.label_token_ids(OFFS, IDS, PROMPT, LABELS)
        assert pl.decide(pl.score_labels(probs(**{"##light": 0.7, "tail": 0.5, "head": 0.1}), pos, dist))["label"] == "taillight"
        tie = pl.decide(pl.score_labels(probs(**{"##light": 0.7, "tail": 0.31, "head": 0.30}), pos, dist))
        assert tie["label_ambiguous"] and {tie["label"], tie["label_alternatives"][0]["label"]} == {"headlight", "taillight"}

    def test_non_confusable_runner_up_is_not_ambiguity(self):
        d = pl.decide([("car door", 0.50), ("door handle", 0.49)])
        assert not d["label_ambiguous"]

    def test_cross_label_nms_keeps_the_alternative(self):
        a = {"label": "car door", "confidence": 0.50, "bbox": [0, 0, 100, 80]}
        b = {"label": "car hood", "confidence": 0.48, "bbox": [1, 0, 100, 80]}
        kept = pl.cross_label_nms([a, b], 0.8)
        assert kept == [a] and a["label_ambiguous"] and a["label_alternatives"][0]["label"] == "car hood"
        far = {"label": "car hood", "confidence": 0.30, "bbox": [0, 0, 100, 80]}
        a2 = {"label": "car door", "confidence": 0.50, "bbox": [0, 0, 100, 80]}
        pl.cross_label_nms([a2, far], 0.8)
        assert not a2.get("label_ambiguous") and a2["label_alternatives"]    # recorded, but a clear winner

    def test_context_hood_vs_door(self):
        door = {"label": "car door", "confidence": 0.6, "bbox": [50, 100, 450, 220]}
        shield = {"label": "windshield", "confidence": 0.7, "bbox": [80, 20, 420, 105]}
        pl.context_flags([door, shield])
        assert door["label_ambiguous"] and "windshield" in door["ambiguity_reason"]
        hood = {"label": "car hood", "confidence": 0.6, "bbox": [0, 0, 300, 200]}
        handle = {"label": "door handle", "confidence": 0.6, "bbox": [100, 80, 140, 100]}
        pl.context_flags([hood, handle])
        assert hood["label_ambiguous"] and hood["label_alternatives"][0]["label"] == "car door"
        real_door = {"label": "car door", "confidence": 0.6, "bbox": [0, 0, 200, 300]}
        pl.context_flags([real_door, {**handle, "bbox": [150, 100, 180, 120]}])
        assert not real_door.get("label_ambiguous")

    def test_ambiguous_pairs_for_recheck(self):
        d = {"label": "headlight", "label_ambiguous": True, "label_alternatives": [{"label": "taillight"}]}
        assert pl.ambiguous_pairs([d, {"label": "fender"}]) == [(d, ["headlight", "taillight"])]


# --------------------------------------------------------------------------- review scorer
def _review(tmp_path, rows):
    p = tmp_path / "review.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ev.FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ev.FIELDS})
    return p


class TestEvalScorer:
    def test_unfinished_review_reports_nothing(self, tmp_path):
        p = _review(tmp_path, [{"photo": "a.jpg", "box_id": "__parts_present__", "true_label": "car hood"},
                               {"photo": "a.jpg", "box_id": 1, "label_phrase": "car door", "label_tokens": "car hood"}])
        with pytest.raises(SystemExit):
            ev.score(p)

    def test_counts_per_method_and_disagreements(self, tmp_path):
        p = _review(tmp_path, [
            {"photo": "a.jpg", "box_id": "__parts_present__", "true_label": "car hood;headlight;grille"},
            {"photo": "a.jpg", "box_id": 1, "label_phrase": "car door", "label_tokens": "car hood", "true_label": "car hood", "box_ok": "y"},
            {"photo": "a.jpg", "box_id": 2, "label_phrase": "headlight", "label_tokens": "taillight", "true_label": "headlight",
             "tokens_ambiguous": "y", "tokens_candidates": "taillight;headlight", "box_ok": "y"},
            {"photo": "a.jpg", "box_id": 3, "label_phrase": "fender", "label_tokens": "fender", "true_label": "none"},
        ])
        rep = ev.score(p)
        old, new = rep["phrase"]["metrics"], rep["tokens"]["metrics"]
        assert old["correct_label"] == 1 and old["wrong_label"] == 1 and old["false_positive"] == 1
        assert new["correct_label"] == 1 and new["ambiguous_truth_in_candidates"] == 1
        assert old["car hood <-> car door confusions"] == 1 and new["headlight <-> taillight confusions"] == 1
        assert old["missed_parts"] == 2 and new["missed_parts"] == 2      # grille + one of hood/headlight
        assert sorted(d["verdict"] for d in rep["disagreements"]) == ["improvement", "regression"]

    def test_merge_matches_boxes_by_iou(self):
        rows = ev.merge_boxes([{"label": "car door", "confidence": 0.5, "bbox": [0, 0, 100, 100]}],
                              [{"label": "car hood", "confidence": 0.5, "bbox": [2, 1, 100, 99], "label_ambiguous": True,
                                "label_alternatives": [{"label": "car door"}]},
                               {"label": "grille", "confidence": 0.4, "bbox": [300, 300, 350, 330]}])
        assert len(rows) == 2 and rows[0]["label_phrase"] == "car door" and rows[0]["label_tokens"] == "car hood"
        assert rows[0]["tokens_candidates"] == "car hood;car door" and rows[1]["label_phrase"] == ""


# --------------------------------------------------------------------------- WagonR
WAGONR = pr.DEMO_POLICIES["UP408060"]


class TestWagonR:
    def test_policy_record(self):
        assert WAGONR["policy_no"] == "AS-MTR-2026-000612" and WAGONR["model"] == "WagonR"
        assert pr.lookup_policy("up 40 8060") is WAGONR

    def test_sign_in_and_policy_view(self, api):
        client, _, _ = api
        r = client.post("/api/session", json={"policy_no": "AS-MTR-2026-000612", "plate": "UP408060"})
        assert r.status_code == 200
        pol = r.json()["policy"]
        assert pol["model"] == "WagonR" and "front bumper" in pol["genuine_part_prices"]

    def test_catalogue_prices_are_used_and_labelled_secondary(self):
        q = pr.price_claim([item("front bumper", "crack", "high")], WAGONR)
        l = q["line_items"][0]
        assert l["method"] == "REPLACE" and l["part_cost"] == 1950
        src = l["part_price_source"]
        assert src["type"] == "scraped" and src["secondary_source"] and "71711M69R00-5PK" in src["part_numbers"]
        assert any("secondary genuine-parts catalogue" in n for n in q["notes"])

    def test_grille_maps_from_the_catalogue_name_and_sub_parts_are_not_priced(self):
        r = pr.real_part_price("maruti", "Wagon R", "grille")
        assert r["price"] == 1250 and r["catalogue_parts"] == ["upper_radiator_grille_assembly"]
        assert pr.real_part_price("maruti", "WagonR", "screw") is None
        assert "fender" not in pr.priced_parts_for("maruti", "WagonR")     # only sub-brackets in the file

    def test_swift_official_prices_unchanged(self):
        assert pr.real_part_price("maruti", "Swift", "front_bumper")["secondary_source"] is False
        assert pr.real_part_price("maruti", "Swift", "headlight")["unverified_only"]   # 'verify' fitment still skipped


# --------------------------------------------------------------------------- replays of the two reported photos
# Part / damage boxes are taken from the attached fused.json (blue car) and read off the Swift
# screenshot; the wheel / grille / bumper boxes there were not in the files and are approximate.
# These check the RULES on those scenes - they are not a detector accuracy measurement.
from fuse import fuse  # noqa: E402


def P(label, conf, bbox, **kw):
    return {"label": label, "confidence": conf, "bbox": list(bbox), "whole_frame": False, **kw}


class TestReportedScenes:
    def blue_car(self):
        return [P("car hood", 0.3927, (1.2, 0.5, 429.3, 211.0)), P("car hood", 0.3731, (377.6, 1.5, 798.4, 329.1)),
                P("headlight", 0.5096, (181.1, 163.2, 439.2, 323.6)), P("taillight", 0.4479, (389.8, 201.0, 588.0, 328.6)),
                P("front bumper", 0.55, (60, 300, 798, 530)), P("wheel", 0.6, (600, 280, 798, 530))]

    def test_blue_car_lamp_and_fender(self):
        parts = self.blue_car()
        notes = pl.context_flags(parts)
        lamp = parts[3]
        assert lamp["label"] == "headlight" and lamp["refined_from"] == "taillight"          # front view
        right_hood = parts[1]
        assert right_hood["label_ambiguous"] and right_hood["label_alternatives"][0]["label"] == "fender"
        assert any("front of the vehicle" in n for n in notes)
        damages = [{"label": "dent", "confidence": 0.5818, "bbox": [382.5, 30.2, 799.5, 315.2]},
                   {"label": "lamp broken", "confidence": 0.58, "bbox": [71.6, 143.4, 584.8, 381.6]},
                   {"label": "dent", "confidence": 0.5191, "bbox": [22.0, 34.1, 271.1, 98.7]},
                   {"label": "lamp broken", "confidence": 0.4549, "bbox": [343.7, 204.4, 577.0, 356.0]}]
        f = fuse(damages, parts, (800, 531))
        got = sorted((m["part"], m["damage"], m["part_uncertain"]) for m in f["matched"])
        assert ("headlight", "lamp broken", False) in got and not any(p == "taillight" for p, _, _ in got)
        assert sum(1 for p, d, _ in got if d == "lamp broken") == 1                         # duplicate merged
        assert ("car hood", "dent", True) in got and ("car hood", "dent", False) in got     # right one: hood/fender?
        assert any("duplicate" in r["reason"] for r in f["rejected"])

    def test_swift_front_view_door_is_not_trusted(self):
        parts = [P("windshield", 0.6, (265, 25, 780, 135)), P("car door", 0.42, (540, 140, 785, 462)),
                 P("grille", 0.55, (90, 350, 520, 520)), P("car hood", 0.5, (90, 120, 600, 340)),
                 P("side mirror", 0.5, (830, 60, 910, 130))]
        pl.context_flags(parts)
        door = parts[1]
        assert door["label_ambiguous"] and door["label_alternatives"][0]["label"] == "fender"
        assert pl.apply_recheck(door, {"car door": 0.7, "fender": 0.3}) is None            # scene forbids 'door'
        assert pl.apply_recheck(door, {"car door": 0.2, "fender": 0.6}) == "fender"
        damages = [{"label": "crack", "confidence": 0.6, "bbox": [545, 140, 785, 462]},
                   {"label": "crack", "confidence": 0.5, "bbox": [655, 140, 780, 455]}]
        f = fuse(damages, parts, (993, 640))
        assert len(f["matched"]) == 1 and f["matched"][0]["part"] == "fender"

    def test_plate_shape_filter(self):
        import plate_reader
        assert plate_reader.plate_shaped([150, 495, 375, 600], 993, 721)          # real Swift plate
        assert not plate_reader.plate_shaped([180, 160, 440, 330], 800, 531)      # lamp-sized square box
        assert not plate_reader.plate_shaped([0, 0, 700, 300], 800, 531)          # whole corner

    def test_overlapping_ocr_boxes_join_without_repeat(self):
        import plate_reader
        hits = [([[11, 20], [87, 17], [88, 48], [12, 51]], "DL3(", 0.61),
                ([[73, 17], [246, 8], [248, 41], [74, 50]], "3CAB1234", 0.77)]
        assert plate_reader._join(hits)[0] == "DL3CAB1234"
        two_line = [([[29, 33], [245, 26], [247, 90], [31, 96]], "HR26", 0.74),
                    ([[32, 123], [288, 114], [290, 178], [34, 186]], "BK9999", 0.76)]
        assert plate_reader._join(two_line)[0] == "HR26BK9999"

    def test_partial_plate_is_not_identified(self, monkeypatch, tmp_path):
        import plate_reader
        from PIL import Image
        monkeypatch.setattr(plate_reader, "_ENGINES", [("fake", lambda img: [([[0, 0], [9, 0], [9, 9], [0, 9]], "BRO7AD", 0.68)])])

        class Det:
            def find(self, img, labels):
                return [{"bbox": [150, 200, 300, 250], "confidence": 0.6}]
        r = plate_reader.read_from_images(Det(), [(Image.new("RGB", (800, 600)), {})], tmp_path)
        assert r["status"] == "not_identified" and "part of the plate" in r["reason"] and r["crop_path"]
