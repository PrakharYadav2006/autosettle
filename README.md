# Auto-Settle AI

Auto-Settle AI is a hackathon prototype for motor insurance claim assessment. It combines photo and video damage analysis, transparent repair estimates, and a server-controlled settlement conversation. It includes a customer portal and a staff review console backed by synthetic demo policies. No payment is initiated.

> **Prototype only:** This project is for demonstration and evaluation. Computer-vision findings, policy rules, repair rates, and sample prices are not production-grade insurance decisions. Do not use this software to settle real claims or upload real customer data.

## Features

- Accepts vehicle damage photos and videos through the customer portal.
- Detects vehicle parts and damage using Grounding DINO and YOLO-based models.
- Samples video keyframes and aggregates evidence across frames.
- Uses area-based severity rules by default. An optional Qwen2.5-VL model can verify physical damage facts and severity; the VLM does not set repair costs or settlement amounts.
- Prices findings with deterministic Python rules for repair method, parts, labour, paint, depreciation, and the demo policy deductible.
- Tracks each finding separately, including items that are priceable, uncertain, unsupported, rejected, or overlapping.
- Can show a provisional partial estimate for priceable items while other findings await review. A partial estimate is not an offer and cannot authorize settlement.
- Checks overlapping detections to reduce duplicate pricing. Ambiguous overlaps remain for review rather than being silently charged twice.
- Answers supported claim questions such as estimate explanations and vehicle/policy details. Unclear questions receive a clarification prompt; an optional text LLM can assist with intent classification and wording.
- Keeps the settlement state and offer amounts on the backend. The language model cannot create, change, or authorize an offer.
- Provides a staff console with claim details, evidence, pricing provenance, negotiation history, and audit records.

## Architecture

```text
frontend/          React + TypeScript + Vite customer and staff portals
api_server.py      FastAPI API, model startup, claims, authentication and audit endpoints
run_pipeline.py    Single-image analysis pipeline
video_pipeline.py  Video keyframe analysis and aggregation
pricing_rules.py   Finding review, overlap handling, pricing and settlement approval
negotiation.py     Message intent, claim Q&A and server-side settlement state machine
severity.py        Area rules and optional Qwen2.5-VL physical-fact assessment
fuse.py            Evidence fusion between detected parts and damage
refine_parts.py    Geometry-based part refinements
grounding_dino_parts.py  Vehicle-part detection
yolo_damage.py     Damage detection
external_prices.csv     Example price records with provenance notes
seed_urls.csv           Optional price-scraper seed URLs
tests/             Automated rules, state-machine and API tests
```

## Requirements

- Python 3.10 or later.
- Node.js and npm for the frontend.
- Python dependencies from `requirements.txt` and a PyTorch build compatible with your machine.
- Internet access may be needed on first startup to download model weights.
- GPU acceleration is recommended for practical model inference. Actual memory use and speed depend on the selected models, image/video sizes, precision, and hardware.

The default vision model is **Qwen2.5-VL-3B-Instruct in 4-bit mode**, when VLM assessment is enabled. VLM assessment is off by default. The optional Qwen3-VL-4B model is not the default and may exceed available memory once detector models and runtime overhead are included; validate the complete stack on the target GPU before using it.

## Run locally

### 1. Install dependencies

Install the PyTorch build appropriate for your CUDA/runtime first (or the CPU build if you do not have a supported GPU). Then, from the project root:

```bash
python -m pip install -r requirements.txt
```

### 2. Start the API backend

From the project root:

```bash
python api_server.py
```

The API listens on `http://localhost:8000`; API documentation is at `http://localhost:8000/docs`. Unless model loading is skipped for tests, startup loads the configured part detector, damage detector, and severity mode. First startup can take several minutes while models initialize or download.

### 3. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal. The customer portal is at `/`; the staff console is at `/admin`. Locally, the frontend uses `http://localhost:8000` by default. Set `VITE_API_URL` at build time to point it at a different backend.

## Configuration

Set these variables in the backend environment before starting `api_server.py`, unless marked as frontend build-time configuration.

| Variable | Purpose | Default |
|---|---|---|
| `AUTOSETTLE_SEVERITY` | `rules` for measured-area severity; `vlm` to load Qwen VLM assessment | `rules` |
| `AUTOSETTLE_VLM_MODEL` | Hugging Face vision model ID when VLM is enabled | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `AUTOSETTLE_REQUIRE_VLM` | When VLM is loaded, require VLM confirmation for automated pricing; `0` disables this gate | `1` with VLM; otherwise unused |
| `AUTOSETTLE_DINO` | Part detector size: `tiny` or `base` | `base` |
| `AUTOSETTLE_WEIGHTS` | Optional damage detector weights path | Detector default behavior |
| `AUTOSETTLE_LLM` | Set to `1` to enable optional local Ollama chat classification/wording | `0` (rules and templates) |
| `OLLAMA_URL` | Ollama server URL for the optional chat LLM | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama text model for the optional chat LLM | `llama3.2:1b` |
| `AUTOSETTLE_MAX_VIDEO_S` | Maximum accepted video length in seconds | `90` |
| `AUTOSETTLE_CORS` | Comma-separated browser origins allowed by the API | `*` |
| `AUTOSETTLE_ADMIN_USER` | Staff console username | `admin` |
| `AUTOSETTLE_ADMIN_PASSWORD` | Staff console password; set this explicitly for a hosted demo | Random value printed at startup if unset |
| `AUTOSETTLE_SECRET` | Persistent secret for signing customer session tokens | Generated if unset; configure explicitly for a hosted deployment |
| `AUTOSETTLE_CLAIMS_DIR` | Directory for claim uploads and stored claim data | `claims/` |
| `AUTOSETTLE_AUDIT_DIR` | Directory for negotiation audit logs | `negotiations/` |

`VITE_API_URL` is a **frontend build-time** variable, not a backend setting. It must be set in the frontend deployment environment before building the Vite app.

### Optional VLM assessment

The VLM runs only when `AUTOSETTLE_SEVERITY=vlm`. It evaluates visual facts; deterministic rules map accepted facts to severity. It does not choose prices. With the default fail-closed confirmation gate, VLM-inconclusive findings are held for review instead of being priced automatically. If the VLM is disabled, measured-area rules are used; do not describe that mode as VLM-verified.

### Optional chat LLM

The chat model is separate from the vision model. To use it, run Ollama locally, pull the configured model (default `llama3.2:1b`), and set `AUTOSETTLE_LLM=1`. It can assist with message intent and wording, but the Python rules and state machine remain authoritative for claim state and all amounts. When it is unavailable or returns unusable output, the app can fall back to rules/templates.

## Deployment

The Vercel deployment is for the **frontend** in `frontend/`. Set the Vercel project root directory to `frontend`, use the Vite build (`npm run build`), and set `VITE_API_URL` to the public HTTPS URL of the separately running FastAPI backend. The included `frontend/vercel.json` rewrites app routes such as `/admin` to `index.html`.

The Python API and computer-vision models do not run on Vercel as part of this frontend deployment. Host the backend on a machine/service that supports the required Python, PyTorch, model downloads, memory, and persistent storage. For a local presentation demo, a secure HTTPS tunnel can expose a laptop backend, but the laptop and backend process must remain available and the browser must be able to reach the tunnel URL.

For a hosted demo, set a strong `AUTOSETTLE_ADMIN_PASSWORD`, a stable private `AUTOSETTLE_SECRET`, and `AUTOSETTLE_CORS` to the exact frontend origin. Ensure claim and audit storage is persistent if it must survive backend restarts. The default CORS wildcard and generated local secrets are development conveniences, not recommended hosted settings.

## Demo policy and pricing

Policy records are synthetic examples in `pricing_rules.py`; vehicle registration lookup is a mock database, not an insurer integration. Pricing can use fallback demo rates and matching rows from `external_prices.csv`. The CSV includes provenance and fitment notes; a source label does not prove the price is current, correctly fitted, or representative of an invoice.

The example settlement progression is 80%, 88%, and 95% of eligible net payable, subject to server-side approval checks. It is a demonstration policy, not a legally reviewed or recommended insurance settlement policy. Offers are only available for complete, approved estimates; partial estimates remain provisional until every finding is resolved.

## Tests

The repository includes `tests/test_autosettle.py`. Install `pytest` if it is not already available, then run from the project root:

```bash
python -m pytest -q
```

These tests exercise pricing/review rules, overlap handling, negotiation-state behavior, and API behavior with models skipped or mocked. They do not measure real-world detector/VLM accuracy, GPU memory use, or end-to-end model quality.

Build the frontend with:

```bash
cd frontend
npm ci
npm run build
```

## Data handling and security

Uploaded evidence, analysis results, claim records, and audit data are stored on the backend under `AUTOSETTLE_CLAIMS_DIR` and `AUTOSETTLE_AUDIT_DIR`. Treat this information as sensitive and remove demo records when no longer needed. Do not upload real customer images or personal information to an uncontrolled environment.

The customer session is intentionally easy to start for a demo and is not real identity verification. The staff console has server-checked credentials, but this prototype has not had a production security audit. Before any real-world use, it would need appropriate identity and access management, secure storage, retention controls, monitoring, operational safeguards, and independent security and insurance review.

## Known limitations

- No insurer, policy-administration, payment, or identity-verification integration.
- Accepting an offer records a demo decision; it does not initiate payment.
- Pricing and policy assumptions are illustrative and require qualified review.
- Model accuracy, robustness, and false-positive rates have not been established on a representative validation dataset.
- Results depend on image quality, view coverage, detector behavior, model availability, and configuration; human review is intentionally retained for uncertain or unsupported findings.
- This prototype is not suitable for real claims or unsupervised production settlement.

## License and attribution

Add the project license and verify attribution and usage terms for all model weights, datasets, images, and third-party price sources before public redistribution.
