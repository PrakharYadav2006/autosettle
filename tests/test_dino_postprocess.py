"""PartDetector's NEW label path with simulated DINO outputs (needs torch + torchvision + transformers;
skipped otherwise).  No model weights: a local WordPiece tokenizer and hand-made logits / boxes.
This checks the plumbing (token alignment, thresholds, NMS, ambiguity, re-check) - not accuracy."""
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
transformers = pytest.importorskip("transformers")
from PIL import Image  # noqa: E402

import grounding_dino_parts as gdp  # noqa: E402

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", ".", "car", "door", "hood", "handle", "head", "##light", "tail",
         "front", "rear", "bump", "##er", "license", "plate", "fender", "grille", "wind", "##shield"]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    p = tmp_path_factory.mktemp("tok") / "vocab.txt"
    p.write_text("\n".join(VOCAB))
    from tokenizers import BertWordPieceTokenizer
    bw = BertWordPieceTokenizer(str(p), lowercase=True)
    return transformers.PreTrainedTokenizerFast(tokenizer_object=bw._tokenizer, unk_token="[UNK]", cls_token="[CLS]",
                                                sep_token="[SEP]", pad_token="[PAD]")


def make_detector(tokenizer, parts, label_mode="tokens"):
    d = object.__new__(gdp.PartDetector)
    d.parts, d.prompt = list(parts), gdp.build_prompt(parts)
    d.box_threshold, d.text_threshold, d.nms_iou, d.cross_label_iou = 0.35, 0.25, 0.5, 0.8
    d.label_mode, d._tok_cache, d.refine, d.vehicle_type, d.custom_parts = label_mode, {}, False, "car", True
    d.device = torch.device("cpu")
    d.model_id, d.load_time_s = "stub", 0.0
    d.processor = types.SimpleNamespace(tokenizer=tokenizer)
    return d


def fake_outputs(tokenizer, prompt, queries):
    """queries: [(box cxcywh normalised, {token text: prob})]."""
    enc = tokenizer(prompt)
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    logits = torch.full((1, len(queries), 256), -10.0)
    for q, (_, tp) in enumerate(queries):
        for t, name in enumerate(toks):
            if name in tp:
                logits[0, q, t] = float(np.log(tp[name] / (1 - tp[name])))
    boxes = torch.tensor([[b for b, _ in queries]], dtype=torch.float32)
    inputs = types.SimpleNamespace(input_ids=torch.tensor([enc["input_ids"]]))
    return inputs, types.SimpleNamespace(logits=logits, pred_boxes=boxes)


IMG = Image.new("RGB", (400, 300))
PARTS = ["car door", "car hood", "door handle", "headlight", "taillight"]


def test_hood_is_not_turned_into_a_door(tokenizer):
    det = make_detector(tokenizer, PARTS)
    inputs, out = fake_outputs(tokenizer, det.prompt, [((0.5, 0.3, 0.6, 0.3), {"car": 0.62, "hood": 0.55, "door": 0.08})])
    d = det._postprocess(inputs, out, IMG)
    assert [x["label"] for x in d] == ["car hood"] and not d[0]["label_ambiguous"]
    assert d[0]["confidence"] == pytest.approx(0.62, abs=1e-3)               # box score = max token prob, as before
    assert d[0]["bbox"] == [80.0, 45.0, 320.0, 135.0]


def test_near_tie_lamp_is_ambiguous(tokenizer):
    det = make_detector(tokenizer, PARTS)
    inputs, out = fake_outputs(tokenizer, det.prompt,
                               [((0.2, 0.5, 0.1, 0.1), {"##light": 0.6, "head": 0.33, "tail": 0.35})])
    d = det._postprocess(inputs, out, IMG)[0]
    assert d["label"] == "taillight" and d["label_ambiguous"] and d["label_alternatives"][0]["label"] == "headlight"


def test_thresholds_and_nms_still_apply(tokenizer):
    det = make_detector(tokenizer, PARTS)
    inputs, out = fake_outputs(tokenizer, det.prompt, [
        ((0.5, 0.5, 0.4, 0.4), {"car": 0.7, "door": 0.6}),
        ((0.51, 0.5, 0.4, 0.4), {"car": 0.65, "door": 0.55}),          # duplicate of the first -> NMS
        ((0.1, 0.1, 0.1, 0.1), {"car": 0.30, "door": 0.30}),           # below the box threshold
        ((0.8, 0.8, 0.1, 0.1), {"car": 0.40, ".": 0.4}),                # a box but no part name grounded
    ])
    d = det._postprocess(inputs, out, IMG)
    assert [x["label"] for x in d] == ["car door"]


def test_phrase_mode_is_the_original_path(tokenizer, monkeypatch):
    det = make_detector(tokenizer, PARTS, label_mode="phrase")
    called = {}
    monkeypatch.setattr(det, "_postprocess_phrase", lambda *a: called.setdefault("phrase", []) or [])
    det._postprocess(None, None, IMG)
    assert "phrase" in called


def test_recheck_resolves_only_a_clear_winner(tokenizer, monkeypatch):
    det = make_detector(tokenizer, PARTS)
    box = {"label": "car door", "bbox": [100.0, 100.0, 300.0, 200.0], "label_ambiguous": True}

    def forward(crop):
        # crop covers the box with 25 % padding -> the box is the centre ~2/3 of the crop
        return fake_outputs(tokenizer, det.prompt, [((0.5, 0.5, 0.67, 0.67), {"car": 0.6, "hood": 0.58, "door": 0.2})]) + (0.0,)
    monkeypatch.setattr(det, "_forward", forward)
    scores = det.recheck(IMG, box, ["car door", "car hood"])
    assert scores["car hood"] > scores["car door"] and det.prompt == gdp.build_prompt(PARTS)   # prompt restored
    import part_labels
    assert part_labels.apply_recheck(box, scores) == "car hood" and box["refined_from"] == "car door"


def test_documented_baseline_tie_breaks():
    """The ORIGINAL phrase path (kept as --labels phrase) resolves shared-token phrases by list order /
    name length - with no record that it guessed.  These are the cases the token path replaces."""
    from vehicles import CAR_PARTS
    assert gdp.canonicalize("car", CAR_PARTS) == "car door"
    assert gdp.canonicalize("car hood car door", CAR_PARTS) == "car door"
    assert gdp.canonicalize("bumper", CAR_PARTS) == "rear bumper"
