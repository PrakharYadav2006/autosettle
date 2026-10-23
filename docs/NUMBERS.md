# What is measured, and what is not

Every number here was produced by the command shown, on the date shown. Anything without a measurement is listed as such. Please quote nothing else.

## Measured

| What | Value | Command / source | Caveat |
|---|---|---|---|
| Automated tests | 213 passed | `python -m pytest -q tests` | No GPU; vision and language models are stubbed in the tests. The pricing engine, state machine, API and checks run for real. |
| Red-team attacks | 70 of 70 held | `python -m pytest -q tests/test_redteam.py`, or **Run all attacks** in the staff console | Shows that these attacks fail, not that no attack exists. The tests are written against the same code. |
| Pricing invariants | 5,000 random claims, 0 violations | `tests/test_fuzz_pricing.py` (seeded) | Invariants: payable formula, offer only on approved complete claims, offer ≤ payable, lower severity never pays more, overrides capped at 125%. |
| Bug found by the red-team suite | 1 (fixed) | `test_path_style_claim_ids_are_rejected` | A 300-character claim ID returned HTTP 500; it now returns 404. |
| Hood→door mislabel cause | Confirmed in code | `tests/test_dino_postprocess.py::test_documented_baseline_tie_breaks` | The original phrase mapping turned `car`, `car hood car door` into "car door", and `bumper` into "rear bumper". This shows the code path, not how often it happens. |
| YOLOv8 damage detector | mAP figures of the public CarDD checkpoint | Model card of the checkpoint used | Not re-validated by this team. Attribute it to the checkpoint. |
| Plate OCR | RapidOCR 4/4 vs EasyOCR 3/4 on synthetic plates; both partial on 1 real, partly hidden plate | Ad-hoc check, 24 Oct | 5 images. An anecdote, not an accuracy figure. |

## Not measured (say so if asked)

- **Part-label accuracy of Grounding DINO**, old vs new labelling. The tool is ready (`tools/eval_dino_parts.py run` and `score`), but the 15-photo review sheet has not been filled in by a person.
- **Severity accuracy** of the VLM fact rules. There is no labelled severity data.
- **End-to-end accuracy on real Indian claim photos.**
- **Garage-estimate reading accuracy** on real or handwritten estimates. The line checks are tested; the transcription is not measured.
- **Labour and paint rates.** They are demo assumptions (IMPUTED). Part prices are OBSERVED only where `external_prices.csv` has an official row; the WagonR rows come from a secondary catalogue.
- **The 125% estimate cap and the 80% cash-offer share.** They are demo policy choices, not industry figures.
- **Cost per claim, hosting, latency at scale.**
