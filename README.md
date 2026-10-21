# Auto-Settle AI

Auto-Settle AI is a hackathon prototype for motor insurance claim assessment. It combines photo and video analysis, an explainable repair estimate, and a server-controlled settlement conversation. The customer and staff portals use synthetic demo policies. No payments are made.

> **Prototype notice:** This project is for demonstration and evaluation. Computer-vision results, policy rules, labour and paint rates, and some part prices are not production-grade insurance decisions. Do not use the estimates to settle real claims.

## What the prototype does

- Accepts a vehicle damage photo or video through the customer portal.
- Detects vehicle parts and damage using Grounding DINO and YOLO-based models.
- For video, samples keyframes and combines observations across frames.
- Optionally uses Qwen2.5-VL to assess physical facts and severity. The default backend configuration uses area-based rules instead.
- Calculates estimates with deterministic Python rules for repair method, parts, labour, paint, depreciation, and policy deductible.
- Offers a server-controlled settlement progression for approved claims. Customer messages cannot directly set the settlement amount.
- Provides a separate staff portal for claim details, pricing provenance, negotiation history, and audit records.

## Project structure

```text
api_server.py            FastAPI application and customer/admin API
pricing_rules.py         Pricing, policy demo data, and approval checks
negotiation.py           Settlement offer ladder and conversation state machine
run_pipeline.py          Single-photo analysis pipeline
video_pipeline.py        Video keyframe analysis and aggregation
grounding_dino_parts.py  Vehicle-part detection
 yolo_damage.py          Damage detection
severity.py              Area rules and optional VLM physical-fact assessment
fuse.py                  Part/damage matching and evidence fusion
refine_parts.py          Geometry-based part refinements
frontend/                React, TypeScript, Vite, and Tailwind portals
external_prices.csv      Example collected part-price records
seed_urls.csv            Optional price-scraper seed URLs
```

## Requirements

- Python 3.10 or later is recommended.
- Node.js and npm for the frontend.
- PyTorch and the Python packages listed in `requirements.txt`.
- A compatible machine for the selected models. GPU acceleration is recommended; model memory and inference time depend on hardware and configuration.
- Model weights may need to be downloaded the first time the backend starts. The project is not guaranteed to run offline on a fresh installation.

## Run locally

### 1. Install backend dependencies

Install the PyTorch build appropriate for your machine first. For example, the repository’s dependency notes mention CUDA 12.4; choose the official PyTorch command that matches your installed CUDA/runtime environment.

Then, from the project root:

```bash
python -m pip install -r requirements.txt
```

### 2. Start the backend

From the project root:

```bash
python api_server.py
```

The API listens on `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.

On startup, the backend loads the part detector, damage detector, and configured severity assessor. The first startup may download model files and take several minutes.

### 3. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal. The customer portal is at `/`; the staff portal is at `/admin`.

The frontend defaults to `http://localhost:8000` for its API. You can set `VITE_API_URL` at build time or change the backend URL in the customer login screen.

## Configuration

The backend supports these environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `AUTOSETTLE_SEVERITY` | Select `rules` or `vlm` severity assessment | `rules` |
| `AUTOSETTLE_DINO` | Grounding DINO model size (`tiny` or `base`) | `base` |
| `AUTOSETTLE_WEIGHTS` | Optional path to damage-detector weights | Model default behavior |
| `AUTOSETTLE_LLM` | Set to `1` to enable optional Ollama chat classification/wording | `0` |
| `AUTOSETTLE_ADMIN_USER` | Staff portal username | `admin` |
| `AUTOSETTLE_ADMIN_PASSWORD` | Staff portal password | Random value printed at startup if unset |
| `AUTOSETTLE_SECRET` | Secret used to sign customer session tokens | Generated and stored under the claims directory |
| `AUTOSETTLE_CLAIMS_DIR` | Location for uploaded evidence, claim records, and audit log | `claims` |
| `AUTOSETTLE_CORS` | Comma-separated allowed browser origins | `*` |

For a local demonstration, keep the backend accessible only to the intended demo environment. Do not expose the default configuration as a public service. Set a strong admin password and a persistent secret before deploying anywhere reachable by other people. Configure CORS to the exact frontend origin for a hosted demo.

### Optional VLM assessment

Set `AUTOSETTLE_SEVERITY=vlm` before starting the backend to load Qwen2.5-VL. This may require substantial memory and a first-run model download. If model output is invalid, the code falls back to area-based severity and marks the claim for review. VLM confirmation is not globally required for automated approval in the current default pricing configuration.

### Optional chat LLM

Set `AUTOSETTLE_LLM=1` to enable Ollama-based intent classification and reply wording. The Python negotiation state machine remains responsible for offer amounts and settlement state. Without this option, the app uses rules and fixed response templates.

## Demo policies and pricing

The policy records in `pricing_rules.py` are synthetic examples. The plate lookup is a mock policy database, not an insurer integration.

Pricing uses a mixture of:

- Explicit demo estimates for fallback part prices, labour, and paint.
- Rows from `external_prices.csv` when a make, model, and part match. The CSV contains entries with different provenance, including snapshot and inferred-fitment notes.
- Prototype depreciation and deductible assumptions defined in `pricing_rules.py`.

A source label does not by itself prove that a price is current, correctly fitted, or representative of a repair invoice. Review the row provenance and notes before presenting any figure.

The current settlement ladder is 80%, 88%, and 95% of eligible net payable, subject to server-side approval checks. It is a demonstration policy, not a recommended or legally approved insurance settlement policy.

## Data handling

The backend stores uploaded evidence, analysis output, claim state, and audit records under `AUTOSETTLE_CLAIMS_DIR` (default: `claims/`). Treat these files as sensitive even when using the demo. Remove demo claim data when it is no longer needed, and do not upload real customer photos or personal information to an uncontrolled environment.

The customer demo session is intentionally easy to start. It should not be treated as real customer authentication. The staff portal uses a separate server-checked password and bearer token, but this prototype has not undergone a security audit suitable for production use.

## Troubleshooting

- **Frontend cannot reach the backend:** confirm the API is running at port 8000 and check the configured API URL.
- **Assessment service is not ready:** wait for model initialization and check the backend terminal for missing dependencies or model download errors.
- **Out of memory or very slow analysis:** use a smaller supported detector configuration, reduce input video length/resolution, or use hardware with more memory. Model-dependent performance has not been benchmarked in this repository.
- **Frontend build reports `tsc` missing:** run `npm install` inside `frontend/`, then retry `npm run build`.
- **Staff login password unknown:** if `AUTOSETTLE_ADMIN_PASSWORD` is unset, use the random password printed in the backend terminal at startup.

## Current limitations

- No real insurer, policy-administration, payment, or identity-verification integration.
- No automated payment is made when an offer is accepted; acceptance is recorded as a demo decision.
- No project test suite is included in the archive.
- Model accuracy, robustness, and false-positive rates have not been established with a representative validation dataset.
- Policy rules and pricing assumptions require review by qualified insurance and repair professionals before any real-world use.
- Production deployment would require additional work on authentication, authorization, upload limits, operational monitoring, data retention, and independent security review.

## License and attribution

Add the project’s license and model/dataset attributions here before public redistribution. Verify the terms for all model weights, datasets, images, and third-party price sources used in a particular deployment.
