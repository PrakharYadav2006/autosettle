# What is measured, and what is not

Every number here was produced by the command shown, on the date shown. Anything without a measurement is listed as such. Please quote nothing else.

## Measured

| What | Value | Command / source | Caveat |
|---|---|---|---|
| Automated tests | 214 passed | `python -m pytest -q tests` | No GPU; vision and language models are stubbed in the tests. The pricing engine, state machine, API and checks run for real. |
| Red-team attacks | 70 of 70 held | `python -m pytest -q tests/test_redteam.py`, or **Run all attacks** in the staff console | Shows that these attacks fail, not that no attack exists. The tests are written against the same code. |
| Pricing invariants | 5,000 random claims, 0 violations | `tests/test_fuzz_pricing.py` (seeded) | Invariants: payable formula, offer only on approved complete claims, offer ≤ payable, lower severity never pays more, overrides capped at 125%. |
| Bug found by the red-team suite | 1 (fixed) | `test_path_style_claim_ids_are_rejected` | A 300-character claim ID returned HTTP 500; it now returns 404. |
| Hood→door mislabel cause | Confirmed in code | `tests/test_dino_postprocess.py::test_documented_baseline_tie_breaks` | The original phrase mapping turned `car`, `car hood car door` into "car door", and `bumper` into "rear bumper". This shows the code path, not how often it happens. |
| YOLOv8 damage detector | mAP figures of the public CarDD checkpoint | Model card of the checkpoint used | Not re-validated by this team. Attribute it to the checkpoint. |
| Plate OCR | RapidOCR 4/4 vs EasyOCR 3/4 on synthetic plates; both partial on 1 real, partly hidden plate | Ad-hoc check, 24 Oct | 5 images. An anecdote, not an accuracy figure. |

## Part detection (Grounding DINO), 15 photos, AI-reviewed

- **Labels:** the review sheet (`eval_out/review.csv`) was filled by an AI reviewer (Codex) looking at the numbered images, not by a person.
- **Dev set, not held-out:** the same 15 photos were used to design the whole-vehicle rule below, so these figures are optimistic.
- **Commands:** `python tools/eval_dino_parts.py run …`, then `score eval_out/review.csv`. The new rule was applied to the same boxes and labels.

| Metric | Old labels (phrase) | New (token labels + scene rules + whole-vehicle rule) |
|---|---|---|
| Boxes | 100 | 91 |
| Correct part label | 47 | 46 |
| Wrong part label | 21 | 22 |
| False boxes (not a part) | 32 | **23** |
| Badly localised | 30 | 30 |
| Missed visible parts | 68 | 69 |
| Hood↔door confusions | 0 | 0 |
| Headlight↔taillight confusions | 1 | 1 |
| Front↔rear bumper confusions | 1 | 2 |

What the numbers say:
- **Main gain:** 9 false "car door" boxes removed. These were DINO boxing the *whole car* and labelling it "car door", because the label contains the word "car". All 9 removals were false boxes; no correct part was lost.
- **One regression:** `images (1).jpeg`, a front bumper read as "rear bumper". The photo had no strong front cue (no grille or hood detected), so the scene rule could not act.
- **Hood/door and headlight/taillight:** no difference between the methods on these photos. The token-label change matters only on shared-token ties, which did not occur here.
- **Overall:** DINO's raw accuracy is low on this set; 47 of 100 boxes are correct. That is why every priced finding also needs VLM confirmation and anything uncertain goes to a person. A model trained on car parts is the real fix (not done).

## Not measured (say so if asked)

- **A human-verified or held-out part-label evaluation.** The figures above are AI-reviewed, on the dev photos.
- **Severity accuracy** of the VLM fact rules. There is no labelled severity data.
- **End-to-end accuracy on real Indian claim photos.**
- **Garage-estimate reading accuracy** on real or handwritten estimates. The line checks are tested; the transcription is not measured.
- **Labour and paint rates.** They are demo assumptions (IMPUTED). Part prices are OBSERVED only where `external_prices.csv` has an official row; the WagonR rows come from a secondary catalogue.
- **The 125% estimate cap and the 80% cash-offer share.** They are demo policy choices, not industry figures.
- **Cost per claim, hosting, latency at scale.**
