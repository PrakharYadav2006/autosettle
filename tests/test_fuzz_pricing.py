"""5,000 random claims through the real pricing engine (seeded, deterministic, ~2 s).  Invariants that
must hold for EVERY claim, whatever the vision models say:

  1. payable = max(0, garage - depreciation - deductible); every money field is a non-negative int
  2. an offer range exists only on an APPROVED, complete estimate, inside 0 < open <= max <= payable
  3. a pending / rejected finding never contributes a priced line
  4. lowering one finding's severity never raises the payable amount (lesser-of rule makes this hold)
  5. garage overrides, however wild, never push a component above 125 % of our own rate
"""
import random

import pytest

import pricing_rules as pr

PARTS = ["front bumper", "rear bumper", "car door", "fender", "car hood", "headlight", "taillight", "fog light",
         "side mirror", "grille", "side skirt", "windshield", "quarter panel", "wheel", "engine"]
DAMAGES = ["dent", "scratch", "crack", "broken", "shattered", "tire flat", "rust"]
SEVS = ["low", "medium", "high", None, "weird"]
VERS = ["confirmed", "rejected", "inconclusive", "failed", "not_run", None]
POLICIES = list(pr.DEMO_POLICIES.values()) + [pr.demo_vehicle(t, "car") for t in ("economy", "mid", "luxury")]


def rand_item(rng):
    it = {"part": rng.choice(PARTS), "damage": rng.choice(DAMAGES), "severity": rng.choice(SEVS),
          "confidence": rng.choice([rng.random(), 0.9, 0.2, 1.5, -0.1, None]),
          "vlm_verification": rng.choice(VERS)}
    if rng.random() < 0.1:
        it["part_uncertain"] = True
    if rng.random() < 0.05:
        it["quality_issue"] = "blurry"
    return it


def check(q):
    t = q["totals"] or {}
    for k, v in t.items():
        if k.endswith("_inr"):
            assert isinstance(v, int) and v >= 0, (k, v)
    if t:
        assert t["insurer_payable_inr"] == max(0, t["garage_estimate_inr"] - t["depreciation_inr"] - t["deductible_inr"])
    rng_ = t.get("settlement_range")
    if rng_ and not q.get("scoped_offer"):
        assert q["status"] == "APPROVED" and t["estimate_scope"] == "complete"
    if rng_:
        assert 0 < rng_["opening_offer_inr"] <= rng_["maximum_offer_inr"] <= t["insurer_payable_inr"]
    priced = {l["finding"] for l in q["line_items"]}
    for f in q["findings"]:
        if f["status"] != "confirmed_priceable":
            assert f["id"] not in priced
    for l in q["line_items"]:
        assert l["garage_cost"] == l["part_cost"] + l["labour"] + l["paint"] and l["depreciation"] <= l["part_cost"]


@pytest.fixture(autouse=True)
def _gate(monkeypatch):
    monkeypatch.setattr(pr, "REQUIRE_VLM_CONFIRMATION", False)


def test_invariants_on_5000_random_claims():
    rng = random.Random(20261024)
    for _ in range(5000):
        items = [rand_item(rng) for _ in range(rng.randint(0, 5))]
        check(pr.price_claim(items, rng.choice(POLICIES)))


def test_lower_severity_never_pays_more():
    rng = random.Random(7)
    rank = {"low": 0, "medium": 1, "high": 2}
    for _ in range(3000):
        pol = rng.choice(POLICIES)
        it = {**rand_item(rng), "confidence": 0.9, "vlm_verification": "confirmed", "severity": "high"}
        it.pop("part_uncertain", None)
        it.pop("quality_issue", None)
        paid = {}
        for sev in ("low", "medium", "high"):
            q = pr.price_claim([{**it, "severity": sev}], pol)
            paid[sev] = (q["totals"] or {}).get("insurer_payable_inr") if q["line_items"] else None
        got = [paid[s] for s in sorted(paid, key=rank.get) if paid[s] is not None]
        assert got == sorted(got), (it, pol.get("model"), paid)


def test_wild_overrides_stay_within_the_cap():
    rng = random.Random(11)
    for _ in range(2000):
        pol = rng.choice(POLICIES)
        it = {**rand_item(rng), "confidence": 0.9, "vlm_verification": "confirmed"}
        base = pr.price_claim([it], pol)
        if not base["line_items"]:
            continue
        b = base["line_items"][0]
        ov = {pr.PART_ALIASES.get(pr.norm(it["part"])): {"method": rng.choice([b["method"], "REPLACE", "PDR"]),
              **{k: rng.choice([10 ** 9, -5, 0, True, "x", b[k] * 2]) for k in ("part_cost", "labour", "paint")}}}
        q = pr.price_claim([it], pol, garage_overrides=ov)
        check(q)
        l = q["line_items"][0]
        for k in ("part_cost", "labour", "paint"):
            assert b[k] <= l[k] <= pr.money(b[k] * pr.GARAGE_LINE_CAP)
