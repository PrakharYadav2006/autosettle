"""
Auto-Settle AI — real spare-part price scraper (proof of concept)
================================================================

Collects REAL part prices from public web pages and saves them, with their
source URL and date, to external_prices.csv. pricing_rules.py then uses these
prices for part replacement wherever it has one, and the mock price list
everywhere else. Every quote line says which one it used.

Sources
-------
1. Maruti Suzuki Genuine Parts (official manufacturer store, MRP incl. taxes)
   - model category pages list products:  /genuine-parts/<model>/<category>
   - each product page states its price in   <meta property="product:price:amount">
   - robots.txt allows /genuine-parts/ (checked automatically every run)
2. Any product URL you add to seed_urls.csv (any make, any site). The price is read
   from standard e-commerce markup: product:price:amount / og:price:amount meta
   tags, schema.org JSON-LD "offers.price", or itemprop="price".

Politeness
----------
* robots.txt is checked before every request; disallowed URLs are skipped.
* One request every 1.5 s, identifying User-Agent, small page cap.
* Only public catalogue pages - no login, no cart, no APIs.

Usage
-----
    python price_scraper.py                         # Maruti: swift, baleno, dzire + seed_urls.csv
    python price_scraper.py --models swift baleno   # choose models
    python price_scraper.py --seed-only             # only the URLs in seed_urls.csv
    python price_scraper.py --dry-run               # list what would be fetched
Output: external_prices.csv (appends; re-running refreshes prices with a new date)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

USER_AGENT = "AutoSettleAI-PriceBot/1.0 (hackathon proof-of-concept; polite crawler)"
DELAY_S = 1.5
TIMEOUT_S = 20
MAX_PRODUCT_PAGES = 60
OUT_CSV = "external_prices.csv"
SEED_CSV = "seed_urls.csv"
FIELDS = ["make", "model", "part", "price_inr", "price_type", "product_title", "part_number",
          "source_site", "source_url", "fitment", "scraped_at", "extract_method"]

MARUTI = "https://www.marutisuzuki.com"
MARUTI_CATEGORIES = ["bumpers-and-grilles", "lighting", "mirrors", "body-exteriors",
                     "glass-and-windshield"]
# product URL = /genuine-parts/<name-slug>/<part number: 5 digits + letters/digits>
PRODUCT_RE = re.compile(r'/genuine-parts/([a-z0-9-]+)/(\d{5}[a-z0-9]*(?:-[a-z0-9]+)?)(?=["\'?#\s])', re.I)

# Product name -> our part names. First matching rule wins; EXCLUDE removes small
# fittings (brackets, bezels, clips) that would otherwise look like a "bumper".
EXCLUDE = re.compile(r"holder|bracket|bezel|clip|screw|retainer|member|garnish|seal|"
                     r"bulb|socket|switch|room|cover|absorber|reinforcement|stay|spacer|"
                     r"moulding|molding|inside|grommet|hinge|lock|handle|visor", re.I)
PART_RULES = [
    ("front_bumper", re.compile(r"front[\s_-]*bumper|bumper[\s_-]*front", re.I)),
    ("rear_bumper", re.compile(r"rear[\s_-]*bumper|bumper[\s_-]*rear", re.I)),
    ("fog_light", re.compile(r"fog", re.I)),
    ("headlight", re.compile(r"head[\s_-]*lamp|headlight", re.I)),
    ("taillight", re.compile(r"tail[\s_-]*lamp|rear[\s_-]*combination|tail[\s_-]*light", re.I)),
    ("side_mirror", re.compile(r"outside[\s_-]*rear[\s_-]*view|outer[\s_-]*mirror|o\.?r\.?v\.?m|door[\s_-]*mirror", re.I)),
    ("grille", re.compile(r"grille(?!.*lower)", re.I)),  # lower grille = small trim piece
    ("windshield", re.compile(r"windshield|windscreen|front[\s_-]*windshield", re.I)),
    ("car_hood", re.compile(r"\bhood\b|bonnet", re.I)),
    ("fender", re.compile(r"fender", re.I)),
    ("car_door", re.compile(r"door[\s_-]*(assy|assembly|panel|shell)|front[\s_-]*door|rear[\s_-]*door", re.I)),
]


def classify(name: str) -> Optional[str]:
    if EXCLUDE.search(name):
        return None
    for part, rx in PART_RULES:
        if rx.search(name):
            return part
    return None


# ----------------------------------------------------------------------------- #
# Polite fetching
# ----------------------------------------------------------------------------- #
class Fetcher:
    def __init__(self, delay: float = DELAY_S):
        self.delay = delay
        self.last = 0.0
        self.robots: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self.count = 0

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                body = self._get(base + "/robots.txt").decode("utf-8", "replace")
                rp.parse(body.splitlines())
            except urllib.error.HTTPError as exc:
                if exc.code in (404, 410):  # no robots.txt = no restrictions (standard rule)
                    rp.parse([])
                else:  # 401/403/5xx: be conservative
                    rp.parse(["User-agent: *", "Disallow: /"])
                    print(f"  [robots] {base}/robots.txt returned {exc.code} - skipping this site")
            except Exception:  # noqa: BLE001 - site unreachable: be conservative
                rp.parse(["User-agent: *", "Disallow: /"])
                print(f"  [robots] could not read {base}/robots.txt - skipping this site")
            self.robots[base] = rp
        return self.robots[base].can_fetch(USER_AGENT, url)

    def _get(self, url: str) -> bytes:
        wait = self.delay - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "text/html,application/xhtml+xml"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return r.read()
        finally:
            self.last = time.time()
            self.count += 1

    def get(self, url: str) -> Optional[str]:
        if not self.allowed(url):
            print(f"  [robots] disallowed, skipped: {url}")
            return None
        try:
            return self._get(url).decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  [fetch] failed {url}: {exc}")
            return None


# ----------------------------------------------------------------------------- #
# Price extraction (works on any site using standard e-commerce markup)
# ----------------------------------------------------------------------------- #
def _meta(page: str, key: str) -> Optional[str]:
    for pat in (rf'<meta[^>]+(?:property|name)=["\']{re.escape(key)}["\'][^>]*content=["\']([^"\']+)',
                rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]*(?:property|name)=["\']{re.escape(key)}["\']'):
        m = re.search(pat, page, re.I)
        if m:
            return html.unescape(m.group(1)).strip()
    return None


def _to_price(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    digits = re.sub(r"[^\d.]", "", str(text).replace(",", ""))
    try:
        v = float(digits)
    except ValueError:
        return None
    return int(round(v)) if 10 <= v <= 500_000 else None  # sanity range for a car part


def extract_price(page: str) -> Optional[Dict]:
    """Returns {price, method, title} or None. Never guesses from loose page text."""
    title = (_meta(page, "og:title") or
             (re.search(r"<title>(.*?)</title>", page, re.I | re.S) or [None, ""])[1]).strip()
    title = html.unescape(re.sub(r"\s+", " ", title))
    for key in ("product:price:amount", "og:price:amount"):
        p = _to_price(_meta(page, key))
        if p:
            cur = (_meta(page, key.replace("amount", "currency")) or "INR").upper()
            if cur == "INR":
                return {"price": p, "method": f"meta:{key}", "title": title}
    for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', page, re.I | re.S):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        for node in (data if isinstance(data, list) else [data]):
            offers = node.get("offers") if isinstance(node, dict) else None
            for off in (offers if isinstance(offers, list) else [offers] if offers else []):
                if isinstance(off, dict) and str(off.get("priceCurrency", "INR")).upper() == "INR":
                    p = _to_price(off.get("price") or off.get("lowPrice"))
                    if p:
                        return {"price": p, "method": "json-ld:offers.price",
                                "title": node.get("name") or title}
    m = re.search(r'itemprop=["\']price["\'][^>]*content=["\']([^"\']+)', page, re.I)
    if m and _to_price(m.group(1)):
        return {"price": _to_price(m.group(1)), "method": "itemprop:price", "title": title}
    return None


# ----------------------------------------------------------------------------- #
# Sources
# ----------------------------------------------------------------------------- #
def maruti_targets(f: Fetcher, models: Iterable[str], dry: bool) -> List[Dict]:
    """Walk model category pages, keep product links whose NAME maps to one of our parts."""
    seen, out = set(), []
    for model in models:
        for cat in MARUTI_CATEGORIES:
            url = f"{MARUTI}/genuine-parts/{model}/{cat}"
            print(f"category  {url}")
            if dry:
                continue
            page = f.get(url)
            if not page:
                continue
            for slug, partno in PRODUCT_RE.findall(page):
                part = classify(slug.replace("-", " "))
                key = partno.lower()
                if part and key not in seen:
                    seen.add(key)
                    out.append({"make": "maruti", "model": model, "part": part,
                                "url": f"{MARUTI}/genuine-parts/{slug}/{partno}".lower(),
                                "part_number": partno.upper(), "price_type": "genuine_oem_mrp",
                                "fitment": f"listed on {model} category page"})
    return out


def seed_targets(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("url") or "").startswith("http"):
                continue
            out.append({"make": r["make"].strip().lower(), "model": r["model"].strip().lower(),
                        "part": r["part"].strip().lower(), "url": r["url"].strip(),
                        "part_number": r.get("part_number", "").strip(),
                        "price_type": r.get("price_type", "listed_price").strip() or "listed_price",
                        "fitment": r.get("fitment", "manually listed").strip() or "manually listed"})
    return out


def scrape(targets: List[Dict], f: Fetcher, limit: int) -> List[Dict]:
    rows, now = [], datetime.now(timezone.utc).isoformat(timespec="seconds")
    for t in targets[:limit]:
        page = f.get(t["url"])
        if not page:
            continue
        got = extract_price(page)
        if not got:
            print(f"  no price markup found: {t['url']}")
            continue
        # double-check the product page name still maps to the same part
        if t.get("fitment", "").startswith("listed on") and classify(got["title"]) not in (t["part"], None):
            print(f"  name mismatch, skipped: {got['title']}")
            continue
        rows.append({"make": t["make"], "model": t["model"], "part": t["part"],
                     "price_inr": got["price"], "price_type": t["price_type"],
                     "product_title": got["title"][:120], "part_number": t["part_number"],
                     "source_site": urlparse(t["url"]).netloc, "source_url": t["url"],
                     "fitment": t["fitment"], "scraped_at": now, "extract_method": got["method"]})
        print(f"  Rs {got['price']:>6,}  {t['make']} {t['model']:<8} {t['part']:<13} {got['title'][:60]}")
    return rows


def save(rows: List[Dict], path: str = OUT_CSV) -> None:
    """Merge with existing file: a newer scrape of the same URL replaces the old row."""
    existing: Dict[str, Dict] = {}
    p = Path(path)
    if p.exists():
        with p.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                existing[r["source_url"]] = r
    for r in rows:
        existing[r["source_url"]] = r
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(existing.values(), key=lambda r: (r["make"], r["model"], r["part"])):
            w.writerow({k: r.get(k, "") for k in FIELDS})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Scrape real spare-part prices (polite, robots.txt-aware)")
    ap.add_argument("--models", nargs="+", default=["swift", "baleno", "dzire"],
                    help="Maruti model slugs as in marutisuzuki.com/genuine-parts/<model>/...")
    ap.add_argument("--seed", default=SEED_CSV, help="CSV of extra product URLs (any site)")
    ap.add_argument("--seed-only", action="store_true")
    ap.add_argument("--limit", type=int, default=MAX_PRODUCT_PAGES, help="max product pages")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    f = Fetcher()
    targets = [] if a.seed_only else maruti_targets(f, a.models, a.dry_run)
    seeds = seed_targets(a.seed)
    known = {t["url"] for t in targets}
    targets += [s for s in seeds if s["url"] not in known]
    print(f"\n{len(targets)} product page(s) to read (limit {a.limit})")
    if a.dry_run:
        for t in targets:
            print(f"  {t['make']} {t['model']} {t['part']:<13} {t['url']}")
        return 0
    rows = scrape(targets, f, a.limit)
    if not rows:
        print("No prices collected (site down, layout changed, or blocked). "
              "external_prices.csv left unchanged.")
        return 1
    save(rows, a.out)
    by = {}
    for r in rows:
        by.setdefault((r["make"], r["model"], r["part"]), []).append(int(r["price_inr"]))
    print(f"\nSaved {len(rows)} price(s) -> {a.out}   ({f.count} requests)")
    for (mk, md, pt), v in sorted(by.items()):
        v.sort()
        print(f"  {mk} {md:<8} {pt:<13} n={len(v)}  Rs {v[0]:,} .. {v[-1]:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
