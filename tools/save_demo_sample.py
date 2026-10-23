"""Save a REAL assessed claim as an instant Demo Mode sample (run on the GPU laptop).

    python tools/save_demo_sample.py <claim_id> <name> "<title>" "<one-line description>"
    e.g. python tools/save_demo_sample.py 3f2a9c1b7d0e swift_front "Swift - bumper and door" "VLM-confirmed dent, OEM bumper price"

Copies the claim's recorded pipeline output (quote, analysis with its images, items, metadata) from
claims/<claim_id>/ to demo_samples/<name>/.  Nothing is invented: it is exactly what the live
pipeline produced.  When a judge opens the sample, the server clones it into a fresh claim; the
offer, chat, proof photos and garage-estimate checks then run live on it, and the page labels it
'recorded sample'.  Use a claim that has had no proof uploads and no chat.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("quote.json", "meta.json", "analysis.json", "items.json", "integrity.json")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    claim_id, name = argv[0], argv[1]
    title = argv[2] if len(argv) > 2 else name
    desc = argv[3] if len(argv) > 3 else ""
    claims = Path(__import__("os").environ.get("AUTOSETTLE_CLAIMS_DIR", ROOT / "claims"))
    src = claims / claim_id
    if not (src / "quote.json").is_file():
        print(f"no claim at {src}")
        return 1
    session = json.loads((src / "session.json").read_text())
    if session.get("evidence") or session.get("history"):
        print("warning: this claim already has chat turns or proof uploads - the sample keeps only the original "
              "assessment (quote and findings as first assessed is safer: pick an untouched claim)")
    dst = ROOT / "demo_samples" / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in FILES:
        if (src / f).is_file():
            shutil.copy(src / f, dst / f)
    meta = json.loads((dst / "meta.json").read_text())
    for k in ("owner", "last_activity"):
        meta.pop(k, None)
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    (dst / "sample.json").write_text(json.dumps({"title": title, "description": desc, "source_claim": claim_id,
                                                 "recorded_at": meta.get("created_at")}, indent=2))
    size = sum(p.stat().st_size for p in dst.iterdir()) / 1e6
    print(f"saved {dst} ({size:.1f} MB).  Commit the demo_samples/ folder; in Demo Mode the dashboard now offers it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
