"""
Auto-Settle AI - Stage 6: settlement negotiation chatbot
========================================================

Golden rule: THE LANGUAGE MODEL NEVER DECIDES MONEY.

    customer message
        -> intent router  (ACCEPT / NEGOTIATE / EXPLAIN / ESCALATE / OTHER)
           keyword rules first, optional small local LLM (Ollama) as a second opinion
        -> Python state machine decides the outcome and the exact amount
           offer ladder: 80 % -> 88 % -> 95 % of the insurer-payable amount
           every offer passes pricing_rules.validate_chatbot_action + range check
        -> reply: fixed template, or LLM rephrasing that is CHECKED - any rupee
           figure the LLM writes must be one Python authorised, else the template
           is used
        -> every turn is written to an audit log (negotiations/<claim_id>.jsonl)

States:  OPEN -> CONFIRMING -> SETTLED
              \\-> ESCALATED (human adjuster)
         NO_OFFER (quote was not APPROVED: nothing to negotiate, only explain / escalate)

Acceptance is two-step ("reply CONFIRM") so a stray "ok" never closes a claim.

Terminal demo (no frontend needed):
    python negotiation.py results\\car\\quote.json
    python negotiation.py results\\car\\quote.json --llm      # use Ollama for wording

Optional LLM:  install Ollama, then  `ollama pull llama3.2:1b`
               (set OLLAMA_MODEL / OLLAMA_URL to change).  Without it the bot
               still works with rules + templates.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pricing_rules as pr

# Evidence-based negotiation: there is NO price ladder.  The cash offer is one fixed share of
# the assessed payable amount; it changes only when new EVIDENCE (an extra photo re-assessed by
# the same vision pipeline) changes the assessment.  Asking, insisting or naming a figure never
# moves it.  A garage estimate is checked line by line (pricing_rules.check_garage_estimate).
OFFER_LADDER = [pr.OPEN_OFFER_RATIO]
MAX_EVIDENCE = 3    # proof uploads per claim
MAX_TURNS = 40   # safety cap on one chat; normal Q&A never gets near it
AUDIT_DIR = Path(os.environ.get("AUTOSETTLE_AUDIT_DIR", "negotiations"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
INTENTS = ("ACCEPT", "NEGOTIATE", "EXPLAIN", "ESCALATE", "OTHER")


# =============================================================================
# Helpers
# =============================================================================
def inr(x: int) -> str:
    """Indian digit grouping: 123456 -> 'Rs 1,23,456'."""
    s = str(int(x))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = f"{head},{tail}"
    return f"Rs {s}"


def ladder(quote: Dict) -> List[int]:
    payable = quote["totals"]["insurer_payable_inr"]
    rng = quote["totals"]["settlement_range"]
    amounts = [pr.money(payable * r) for r in OFFER_LADDER]
    # never outside the range pricing_rules authorised
    return [min(max(a, rng["opening_offer_inr"]), rng["maximum_offer_inr"]) for a in amounts]


AMOUNT_RE = re.compile(r"(?:₹|rs\.?|inr|rupees?)\s*([\d,]+(?:\.\d+)?)\s*(k|thousand|hazaar|hazar)?"
                       r"|([\d,]{4,}(?:\.\d+)?)\s*(?:rs|rupees?|/-)?"
                       r"|(\d+(?:\.\d+)?)\s*(k|thousand|hazaar|hazar)\b", re.I)


def parse_amount(text: str) -> Optional[int]:
    """Largest rupee figure in the message (only clear money mentions)."""
    vals = []
    for m in AMOUNT_RE.finditer(text):
        num, mult = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), None) if m.group(3) \
            else (m.group(4), m.group(5))
        try:
            v = float(num.replace(",", ""))
        except (ValueError, AttributeError):
            continue
        if mult:
            v *= 1000
        if 50 <= v <= 10_000_000:
            vals.append(int(v))
    return max(vals) if vals else None


# =============================================================================
# Intent router
# =============================================================================
NEGATION = re.compile(r"\b(not|no|nahi|nahin|nhi|never|mat|don'?t|cannot|can'?t|won'?t)\b", re.I)
RULES = [
    ("ESCALATE", re.compile(r"\b(human|agent|person|surveyor|manager|complain\w*|lawyer|court|legal|"
                            r"ombudsman|insaan|aadmi|call me|supervisor|escalat\w*)\b", re.I)),
    ("NEGOTIATE", re.compile(r"\b(low|less|more|increase|higher|raise|better|too little|not enough|"
                             r"kam|zyada|jyada|badha\w*|aur do|negotiat\w*|counter|unfair|reject|reconsider\w*|revis\w*)\b", re.I)),
    ("EXPLAIN", re.compile(r"\b(why|how|what|explain|breakdown|break up|details?|calculat\w*|"
                           r"depreciation|deductible|kaise|kyun|kyu|kya|samjha\w*|batao)\b|\?", re.I)),
    ("ACCEPT", re.compile(r"\b(accept\w*|agree\w*|ok|okay|deal|fine|done|yes|haan|ha|theek|thik|"
                          r"manzoor|chalega|confirm\w*|sure|go ahead)\b", re.I)),
]


def rule_intent(text: str, current_offer: Optional[int]) -> str:
    t = text.strip()
    asked = parse_amount(t)
    if asked and current_offer and asked > current_offer:
        return "NEGOTIATE"                       # names a higher figure = counter-offer
    for intent, rx in RULES:
        if rx.search(t):
            if intent == "ACCEPT" and NEGATION.search(t):
                return "NEGOTIATE"               # "not ok", "nahi chalega"
            return intent
    if asked and current_offer and asked <= current_offer:
        return "ACCEPT"
    return "OTHER"


def ollama(messages: List[Dict], max_tokens: int = 120, timeout: float = 20) -> Optional[str]:
    body = json.dumps({"model": OLLAMA_MODEL, "messages": messages, "stream": False,
                       "options": {"temperature": 0, "num_predict": max_tokens}}).encode()
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["message"]["content"].strip()
    except Exception:  # noqa: BLE001 - LLM is optional, never fatal
        return None


def llm_intent(text: str) -> Optional[str]:
    out = ollama([
        {"role": "system", "content": "Classify the customer's message about an insurance cash offer. "
                                      "Answer with exactly one word: ACCEPT, NEGOTIATE, EXPLAIN, ESCALATE or OTHER."},
        {"role": "user", "content": text}], max_tokens=5)
    if not out:
        return None
    word = re.sub(r"[^A-Z]", "", out.split()[0].upper()) if out.split() else ""
    return word if word in INTENTS else None


def classify(text: str, current_offer: Optional[int], use_llm: bool) -> Tuple[str, str]:
    """Returns (intent, source).  Rules are trusted for money-related signals."""
    r = rule_intent(text, current_offer)
    if not use_llm or r in ("NEGOTIATE", "ESCALATE") or parse_amount(text):
        return r, "rules"
    l = llm_intent(text)
    if l is None:
        return r, "rules (llm unavailable)"
    if r == "ACCEPT" and l != "ACCEPT":          # never accept on a disagreement
        return l, "llm"
    return (l if r == "OTHER" else r), ("llm" if r == "OTHER" else "rules")


# =============================================================================
# State machine
# =============================================================================
def new_session(claim_id: str, quote: Dict) -> Dict:
    s = {"claim_id": claim_id, "quote_status": quote["status"], "turns": 0, "history": [],
         "tier": 0, "offer_inr": None, "state": "NO_OFFER", "settled_amount_inr": None}
    offerable = quote["status"] == "APPROVED" or bool(quote.get("scoped_offer"))   # scoped = Demo Mode only
    if offerable and pr.validate_chatbot_action({"type": "present_offer"}, quote)["accepted"]:
        s["ladder_inr"] = ladder(quote)
        s["offer_inr"] = s["ladder_inr"][0]
        s["state"] = "OPEN"
    return s


def explain_text(quote: Dict) -> str:
    t = quote.get("totals") or {}
    parts = []
    for l in quote["line_items"]:
        parts.append(f"{l['part'].replace('_', ' ')} ({l['damage']}, {l['severity']}): "
                     f"{l['method'].replace('_', ' ').lower()} {inr(l['garage_cost'])}")
    if not parts or not t:
        return "No repair on this claim could be priced automatically yet."
    lines = "; ".join(parts)
    partial = t.get("estimate_scope") == "partial" or quote.get("status") in ("REVIEW_REQUIRED", "ESCALATED")
    head = "Here is the provisional estimate so far" if partial else "Here is how we calculated it"
    txt = (f"{head}. Repairs: {lines}. Garage estimate {inr(t['garage_estimate_inr'])}, "
           f"minus depreciation {inr(t['depreciation_inr'])} and the policy deductible "
           f"{inr(t['deductible_inr'])}, gives {inr(t['insurer_payable_inr'])}.")
    pending = [f for f in quote.get("findings", []) if f["status"] in PENDING]
    if pending:
        txt += (f" This is not a final amount: {len(pending)} item(s) still need a specialist's check "
                "and are not included.")
    elif partial:
        txt += " This is not a final amount: the claim needs a specialist's check before any offer."
    if t.get("settlement_range"):
        txt += (" An instant cash settlement is offered at a little below that amount because it is paid "
                "today with no garage visit or survey.")
    return txt


# =============================================================================
# Factual questions answered from server data (no model involved, no state change)
# =============================================================================
PENDING = ("uncertain_review", "unsupported_review", "duplicate_or_overlap")
# reason code -> plain words for the customer (no confidences, no internal limits)
PLAIN = {
    "SUSPICIOUS": "the vehicle or policy details need to be checked",
    "MAKE_UNKNOWN": "the vehicle details could not be matched to a policy",
    "POLICY_INACTIVE": "the policy is not active",
    "MAKE_NOT_IN_PRICE_LIST": "our price list does not cover this vehicle make",
    "VEHICLE_UNVERIFIED": "the exact vehicle model or year is not confirmed",
    "IMAGE_QUALITY": "the photo is too small or unclear to judge every detail",
    "OVERLAP": "one damaged area spans two panels and a person must decide which repair applies",
    "VIDEO_WEAK": "a possible damage was seen only briefly in the video",
    "VLM_ONLY": "a possible damage needs a closer look",
    "VLM_INCONCLUSIVE": "it is unclear from the photos whether a mark is real damage",
    "VLM_FAILED": "an automatic check could not be completed for one item",
    "VLM_NOT_CONFIRMED": "one item could not be double-checked automatically",
    "PART_LOCATION_UNCLEAR": "the damaged panel could not be identified from the photo",
    "PART_NOT_IN_PRICE_LIST": "a damaged part is not in our automatic price list",
    "ALWAYS_INSPECT": "a damaged part always needs a physical inspection (for example glass, wheels or structure)",
    "STRUCTURAL_INSPECT": "the damage may affect the structure and needs an inspection",
    "UNKNOWN_DAMAGE": "a type of damage could not be classified",
    "SEVERITY_UNKNOWN": "how serious one damage is could not be decided",
    "SEVERITY_REVIEW": "how serious one damage is needs a specialist's check",
    "LOW_CONFIDENCE": "one possible damage is not clear enough in the photo",
    "CONFIDENCE_INVALID": "one item could not be assessed",
    "ABOVE_AUTO_LIMIT": "the estimate is above the limit for instant settlement, so a surveyor checks it",
    "NO_PRICEABLE_DAMAGE": "no damage could be priced automatically",
    "PREVIOUSLY_SETTLED": "this part was already paid on your policy, so a person checks the damage is new",
}

TOPICS = [   # first match wins; checked only for EXPLAIN / OTHER messages
    ("garage_estimate", re.compile(r"\b(garage|workshop|mechanic|service cent(er|re))\b.*\b(estimat\w*|quot\w*|"
                                   r"bill|invoice)\b|\b(quotation|invoice|garage bill|repair bill|job ?card)\b|"
                                   r"\b(estimate|bill)\b.*\b(garage|workshop|mechanic)\b", re.I)),
    ("transfer", re.compile(r"\b(money|pay(ment|out)?|paid|transfer\w*|bank|account|upi|paisa|paise|"
                            r"credited|receive)\b", re.I)),
    ("why_no_offer", re.compile(r"\b(why|kyu|kyun|kyon)\b.*\b(no|not|review|specialist|escalat\w*|offer)\b|"
                                r"\b(no|not)\b.*\boffer\b|\boffer\b.*\b(kyu|kyun|nahi|nhi)\b|\bpending\b|"
                                r"\bunder review\b", re.I)),
    ("next", re.compile(r"\b(next|what happens|what now|how long|kab tak|kab|aage|status|update)\b", re.I)),
    ("vehicle", re.compile(r"\b(which|what|kaun\s?si|konsi|kaunsi)\s+(car|vehicle|bike|scooter|scooty|gaadi|gadi|"
                           r"model|make)\b|\b(make|model|year|cc|engine|policy (number|details))\b|"
                           r"\bmy (car|vehicle|bike|scooter) details\b", re.I)),
    ("damage", re.compile(r"\b(damages?|detected|found|dents?|scratch\w*|cracks?|broken|items?|parts?|"
                          r"nuksaan|nuksan)\b", re.I)),
    ("calc", re.compile(r"\b(calculat\w*|breakdown|break up|how much|estimate|depreciation|deductible|excess|"
                        r"payable|kaise|kitna|hisaab|why)\b", re.I)),
    ("greeting", re.compile(r"^\s*(hi+|hello|hey|namaste|namaskar|good (morning|afternoon|evening)|help)\b", re.I)),
]


def info_topic(text: str) -> Optional[str]:
    for name, rx in TOPICS:
        if rx.search(text):
            return name
    return None


def _items(fs: List[Dict]) -> str:
    seen = []
    for f in fs:
        w = f"{str(f['part']).replace('_', ' ')} ({str(f['damage']).replace('_', ' ')})"
        if w not in seen:
            seen.append(w)
    return ", ".join(seen)


def plain_blockers(quote: Dict) -> List[str]:
    out = []
    for b in quote.get("blockers", []):
        w = PLAIN.get(b.get("code"))
        if w and w not in out:
            out.append(w)
    return out


def options_line(session: Dict) -> str:
    if session["state"] in ("OPEN", "CONFIRMING"):
        return ("I can explain how the offer was calculated, list the damage we found, accept the offer, "
                "re-assess a proof photo you upload, or connect you to a person.")
    return ("I can explain the estimate, tell you which items need review, tell you what happens next, "
            "or connect you to a person.")


def garage_estimate_text(session: Dict) -> str:
    cap = f"{pr.GARAGE_LINE_CAP * 100:.0f}%"
    if not can_submit_evidence(session):
        return ("No more documents can be added to this claim here. You can ask to speak to a person, and they "
                "can look at your garage estimate.")
    return ("Please upload it with the 'Garage estimate' button - a clear photo or a PDF of the itemised estimate. "
            "We read each line and check it against the damage in your photos and our repair rates. Where a line "
            f"is above our rate, we accept it up to {cap} of our rate (our demo limit); lines for parts we didn't "
            "find damaged, a different repair method, or amounts above that limit are left out of the instant "
            "offer and noted for a specialist. The estimate's grand total is never used directly, and the offer "
            "is then recalculated by the same pricing rules.")


def answer_question(topic: str, session: Dict, quote: Dict) -> str:
    """Deterministic answers from the claim record only.  Missing data -> said plainly."""
    st, status = session["state"], quote.get("status")
    findings = quote.get("findings", [])
    if topic == "garage_estimate":
        return garage_estimate_text(session)
    if topic == "transfer":
        if st == "SETTLED":
            return (f"Your acceptance of {inr(session['settled_amount_inr'])} is recorded. This is a demo, "
                    "so no money is transferred.")
        return ("This is a demo. Accepting an offer records your decision, but no money is transferred.")
    if topic == "vehicle":
        v = quote.get("vehicle") or {}
        if not v.get("make"):
            return "The vehicle details on this claim are not confirmed yet."
        desc = " ".join(str(x) for x in (v.get("year"), str(v.get("make", "")).title(), v.get("model")) if x)
        cc = f", {v['engine_cc']} cc" if v.get("engine_cc") else ""
        zd = " with zero-depreciation cover" if v.get("zero_dep") else ""
        return f"This claim is for your {desc}{cc}{zd}, as recorded on your policy."
    if topic == "damage":
        priced = [f for f in findings if f["status"] == "confirmed_priceable"]
        pending = [f for f in findings if f["status"] in PENDING]
        ruled = [f for f in findings if f["status"] == "rejected"]
        if not findings:
            return "No damage findings are recorded on this claim yet."
        parts = []
        if priced:
            parts.append(f"Included in the estimate: {_items(priced)}.")
        if pending:
            parts.append(f"Waiting for a specialist's check (not priced yet): {_items(pending)}.")
        if ruled:
            parts.append(f"{len(ruled)} possible mark(s) were ruled out as reflections or not damage.")
        return " ".join(parts)
    if topic == "calc":
        return explain_text(quote) + (f" Our current offer is {inr(session['offer_inr'])}." + scope_text(quote)
                                      if st in ("OPEN", "CONFIRMING") else "")
    if topic == "why_no_offer":
        if st in ("OPEN", "CONFIRMING"):
            return f"An instant offer is available on this claim: {inr(session['offer_inr'])}." + scope_text(quote)
        if st == "SETTLED" and quote.get("scoped_offer"):
            return ("You accepted the partial offer for the listed items. The remaining items are still under "
                    "review, so they have no offer yet.")
        if st == "SETTLED":
            return "You have already accepted an offer on this claim."
        if status == "NO_PAYOUT":
            return ("The repair estimate is within your policy's compulsory excess, so there is no payable "
                    "amount and no cash offer.")
        why = plain_blockers(quote)
        if not why:
            return "This claim needs a specialist's check before an offer can be made."
        return "An instant offer isn't available because " + "; ".join(why[:4]) + "."
    if topic == "next":
        if st == "SETTLED" and quote.get("scoped_offer"):
            return ("The listed items are settled. The remaining items stay under review; a specialist will "
                    "contact you about them. Accepting did not waive them. No money is transferred in this demo.")
        if st == "SETTLED":
            return "Your decision is recorded and this claim is closed. This is a demo, so no money is transferred."
        if st == "ESCALATED":
            return "A claims specialist has your claim and will contact you. Your photos and estimate are saved."
        if st in ("OPEN", "CONFIRMING"):
            return ("You can accept the offer, ask for reconsideration, or ask for a person. Nothing is decided "
                    "until you confirm.")
        return ("A claims specialist will check the items that need review and contact you. If you'd like a "
                "person sooner, just say 'talk to a person'.")
    return options_line(session)   # greeting / anything else


def authorise(amount: int, quote: Dict) -> bool:
    """Python guard: action allowed AND amount inside the approved range."""
    v = pr.validate_chatbot_action({"type": "present_offer"}, quote)
    if not v["accepted"]:
        return False
    rng = v["range"]
    return rng["opening_offer_inr"] <= amount <= rng["maximum_offer_inr"]


def decide(session: Dict, quote: Dict, intent: str, text: str) -> Dict:
    """Pure Python: returns {"outcome", "amount", "template"} and updates session."""
    st = session["state"]
    asked = parse_amount(text)

    topic = info_topic(text) if intent in ("EXPLAIN", "OTHER") else None
    if intent == "NEGOTIATE" and info_topic(text) == "garage_estimate" and st in ("OPEN", "CONFIRMING", "NO_OFFER"):
        # "my garage quoted 15000": the figure in chat changes nothing; the estimate itself is checked
        return {"outcome": "EVIDENCE_NEEDED", "amount": session.get("offer_inr") if st != "NO_OFFER" else None,
                "template": "A figure in the chat can't change the offer, but the estimate itself can be checked. "
                            + garage_estimate_text(session), "topic": "garage_estimate"}
    if st in ("SETTLED", "ESCALATED") and topic and topic != "greeting":
        return {"outcome": "ANSWERED", "amount": None, "template": answer_question(topic, session, quote),
                "topic": topic}
    if st in ("SETTLED", "ESCALATED"):
        if st == "SETTLED" and quote.get("scoped_offer"):
            msg = (f"Your acceptance of {inr(session['settled_amount_inr'])} is recorded for the listed items only. "
                   "The remaining items stay under review and a specialist will contact you about them.")
        else:
            msg = (f"You have accepted {inr(session['settled_amount_inr'])} for this claim. Your decision is recorded."
                   if st == "SETTLED" else "Your claim is with a claims specialist, who will contact you.")
        return {"outcome": "CLOSED", "amount": session.get("settled_amount_inr"), "template": msg}

    if intent == "ESCALATE" or session["turns"] >= MAX_TURNS:
        session["state"] = "ESCALATED"
        return {"outcome": "ESCALATED", "amount": None,
                "template": "I've passed your claim to a claims specialist. They will review it and contact you. "
                            "Your claim details and photos are saved with the claim."}

    if topic and topic != "greeting":
        return {"outcome": "ANSWERED", "amount": session.get("offer_inr") if st in ("OPEN", "CONFIRMING") else None,
                "template": answer_question(topic, session, quote), "topic": topic}

    if st == "NO_OFFER":
        no_pay = quote.get("status") == "NO_PAYOUT"
        tail = (" Because the repair cost is within your policy excess, there is no payable amount." if no_pay
                else " Some details need a check by a claims specialist before an offer can be made.")
        if intent == "EXPLAIN" and quote.get("line_items"):
            return {"outcome": "EXPLAINED", "amount": None, "template": explain_text(quote) + tail}
        if intent == "EXPLAIN":
            return {"outcome": "EXPLAINED", "amount": None,
                    "template": "This claim could not be priced automatically." + tail}
        if intent in ("ACCEPT", "NEGOTIATE"):
            return {"outcome": "NO_OFFER", "amount": None,
                    "template": "There is no cash offer on this claim to accept or change yet." + tail
                                + evidence_hint(session)}
        # greeting / unclear message: ask, don't repeat the same refusal
        return {"outcome": "CLARIFY", "amount": None,
                "template": "How can I help? " + options_line(session)}

    offer = session["offer_inr"]

    if st == "CONFIRMING":
        if intent == "ACCEPT" and not NEGATION.search(text):
            if not authorise(offer, quote):
                session["state"] = "ESCALATED"
                return {"outcome": "ESCALATED", "amount": None,
                        "template": "I couldn't authorise this amount, so a human adjuster will finish your claim."}
            session["state"] = "SETTLED"
            session["settled_amount_inr"] = offer
            return {"outcome": "SETTLED", "amount": offer,
                    "template": (f"Your acceptance of {inr(offer)} is recorded for the listed items only. "
                                 "The remaining items stay under review." if quote.get("scoped_offer") else
                                 f"Thank you. Your acceptance of {inr(offer)} is recorded for this claim.")}
        session["state"] = "OPEN"          # anything else cancels the confirmation
        st = "OPEN"
        if intent == "ACCEPT":
            intent = "OTHER"

    if intent == "ACCEPT":
        session["state"] = "CONFIRMING"
        return {"outcome": "CONFIRM_REQUEST", "amount": offer,
                "template": (f"Great. Please reply CONFIRM to accept {inr(offer)} for the listed items only. "
                             "The items under review are not included and stay open." if quote.get("scoped_offer") else
                             f"Great. Please reply CONFIRM to accept {inr(offer)} as full and final settlement "
                             "of this claim.")}

    if intent == "EXPLAIN":
        return {"outcome": "EXPLAINED", "amount": offer,
                "template": explain_text(quote) + f" Our current offer is {inr(offer)}."}

    if intent == "NEGOTIATE":
        if asked is not None and asked <= offer:   # they asked for less than we offer
            session["state"] = "CONFIRMING"
            return {"outcome": "CONFIRM_REQUEST", "amount": offer,
                    "template": f"Our offer of {inr(offer)} already covers that. Reply CONFIRM to accept it."}
        # No proof, no change: the amount comes from the assessed damage, not from how hard
        # the customer pushes.  Tell them exactly how to change it with evidence.
        return {"outcome": "EVIDENCE_NEEDED", "amount": offer,
                "template": f"I can't change the offer on request alone - {inr(offer)} is calculated from the damage "
                            "we found." + calc_line(quote) + evidence_hint(session)}

    if intent == "OTHER" and not topic:   # unclear message: ask instead of repeating the offer
        return {"outcome": "CLARIFY", "amount": offer,
                "template": f"Sorry, I didn't quite get that. Your current offer is {inr(offer)}. "
                            + options_line(session)}
    return {"outcome": "OFFER", "amount": offer,
            "template": f"Your cash settlement offer is {inr(offer)}. " + options_line(session)}


# =============================================================================
# Reply wording (optional LLM, always checked)
# =============================================================================
NUM_RE = re.compile(r"\d[\d,]*")


def safe_rephrase(template: str, allowed: set, use_llm: bool) -> Tuple[str, str]:
    if not use_llm:
        return template, "template"
    out = ollama([
        {"role": "system", "content": "You are a polite motor-insurance claims assistant in India. Rewrite the "
                                      "message below in a warm, natural tone in at most 3 sentences. Keep every "
                                      "number and rupee amount EXACTLY as written. Do not add any new numbers, "
                                      "promises or conditions."},
        {"role": "user", "content": template}], max_tokens=150)
    if not out:
        return template, "template (llm unavailable)"
    nums = {int(n.replace(",", "")) for n in NUM_RE.findall(out) if n.replace(",", "").isdigit()}
    must = {int(n.replace(",", "")) for n in NUM_RE.findall(template) if n.replace(",", "").isdigit()}
    if not nums <= (allowed | must) or not must <= nums or len(out) > 600:
        return template, "template (llm reply rejected by guard)"
    return out, "llm"


def allowed_numbers(session: Dict, quote: Dict) -> set:
    t = quote.get("totals") or {}
    vals = {v for v in t.values() if isinstance(v, int)}
    vals |= set(session.get("ladder_inr") or [])
    vals |= {l["garage_cost"] for l in quote.get("line_items", [])}
    v = quote.get("vehicle") or {}
    vals |= {x for x in (v.get("year"), v.get("engine_cc")) if isinstance(x, int)}
    vals |= set(range(0, 21))   # small counts ("2 items", "3 sentences")
    return vals


# =============================================================================
# Public entry point (used by api_server.py and the CLI)
# =============================================================================
def handle_message(session: Dict, quote: Dict, text: str, use_llm: bool = False) -> Dict:
    text = (text or "").strip()[:500]
    intent, src = classify(text, session.get("offer_inr"), use_llm)
    session["turns"] += 1
    d = decide(session, quote, intent, text)
    reply, wording = safe_rephrase(d["template"], allowed_numbers(session, quote), use_llm)
    turn = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "user": text,
            "intent": intent, "intent_source": src, "outcome": d["outcome"], "amount_inr": d["amount"],
            "state": session["state"], "tier": session["tier"], "reply": reply, "wording": wording,
            "topic": d.get("topic")}
    session["history"].append(turn)
    audit(session["claim_id"], turn)
    return {"reply": reply, "intent": intent, "outcome": d["outcome"], "state": session["state"],
            "offer_inr": session.get("offer_inr"), "settled_amount_inr": session.get("settled_amount_inr"),
            "tier": session["tier"], "wording": wording, "can_reconsider": can_reconsider(session)}


def calc_line(quote: Dict) -> str:
    """One sentence of the calculation, from the stored quote (facts only)."""
    t = quote.get("totals") or {}
    if not t.get("garage_estimate_inr"):
        return ""
    return (f" Repairs {inr(t['garage_estimate_inr'])}, minus depreciation {inr(t['depreciation_inr'])} and the "
            f"deductible {inr(t['deductible_inr'])}, gives {inr(t['insurer_payable_inr'])}; the instant cash offer "
            "is a fixed share of that.")


def can_reconsider(session: Dict) -> bool:
    """Kept for older clients: a reconsideration without evidence never changes the offer."""
    return False


def can_submit_evidence(session: Dict) -> bool:
    return session["state"] in ("OPEN", "CONFIRMING", "NO_OFFER") and \
        len(session.get("evidence", [])) < MAX_EVIDENCE


def evidence_hint(session: Dict) -> str:
    if not can_submit_evidence(session):
        return " You can ask to speak to a person at any time."
    return (" If we missed damage, or it is worse than we assessed, upload a clear close-up photo of it "
            "with 'Upload proof' and we will re-assess it. You can also upload an itemised garage estimate: "
            "we check each line against your photos and our rates. Or accept, or ask to speak to a person.")


def apply_evidence(session: Dict, old_quote: Dict, new_quote: Optional[Dict], summary: str) -> Dict:
    """After a proof photo was re-assessed by the vision pipeline + pricing engine (never by a
    language model): move the session to the new authorised offer, or explain why nothing changed.
    new_quote None = the evidence changed nothing material."""
    before = session.get("offer_inr")
    if new_quote is None:
        tmpl = summary + (f" Your offer stays at {inr(before)}." if before else "")
        return {"outcome": "EVIDENCE_NO_CHANGE", "amount": before, "template": tmpl}
    fresh = new_session(session["claim_id"], new_quote)
    session.update(state=fresh["state"], offer_inr=fresh["offer_inr"], ladder_inr=fresh.get("ladder_inr"),
                   quote_status=new_quote["status"], tier=0)
    if session["state"] == "OPEN":
        tail = (f" Your offer is now {inr(session['offer_inr'])}" + (f" (was {inr(before)})." if before else ".")
                + scope_text(new_quote))
        return {"outcome": "EVIDENCE_OFFER_UPDATED", "amount": session["offer_inr"], "template": summary + tail}
    why = "; ".join(plain_blockers(new_quote)[:3])
    return {"outcome": "EVIDENCE_REVIEW", "amount": None,
            "template": summary + " A claims specialist now needs to check the claim"
                        + (f" ({why})" if why else "") + ", so there is no instant offer for the moment."}


def scope_text(quote: Dict) -> str:
    """For a scoped (partial) demo offer: exactly what it covers and what it leaves out."""
    so = quote.get("scoped_offer")
    if not so:
        return ""
    fs = {f["id"]: f for f in quote.get("findings", [])}
    inc = _items([fs[i] for i in so["finding_ids"] if i in fs])
    exc = _items([fs[i] for i in so["excluded_finding_ids"] if i in fs and fs[i]["status"] in PENDING])
    return (f" This offer covers only: {inc}." + (f" Not included, still under review: {exc}." if exc else "")
            + " Demo simulation - no money is transferred.")


def opening_message(session: Dict, quote: Dict) -> str:
    if session["state"] == "OPEN" and quote.get("scoped_offer"):
        return (f"Part of your claim could be assessed. Your partial demo offer is {inr(session['offer_inr'])}."
                + scope_text(quote) + " You can accept it, ask how it was calculated, upload proof if we "
                "missed any damage, or speak to a person.")
    if session["state"] == "OPEN":
        return (f"Your claim has been assessed. Your cash settlement offer is {inr(session['offer_inr'])}. "
                "You can accept it, ask how it was calculated, upload proof if we missed any damage, "
                "or speak to a person.")
    if quote["status"] == "NO_PAYOUT":
        return ("Your claim has been assessed, but the repair cost is below your policy deductible, so there is "
                "no payable amount. Ask 'why' to see the breakdown.")
    return ("Your claim has been assessed and needs a check by a claims specialist before an offer can be made. "
            "You can ask how the estimate was calculated or ask to speak to a person.")


def audit(claim_id: str, turn: Dict) -> None:
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_DIR / f"{claim_id}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(turn) + "\n")
    except OSError:
        pass


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Stage 6 negotiation chat in the terminal")
    ap.add_argument("quote_json")
    ap.add_argument("--llm", action="store_true", help=f"use Ollama ({OLLAMA_MODEL}) for intent + wording")
    a = ap.parse_args(argv)
    quote = json.loads(Path(a.quote_json).read_text())
    session = new_session(Path(a.quote_json).parent.name or "claim", quote)
    print(f"[quote status: {quote['status']}]  type 'quit' to exit\n")
    print("Bot:", opening_message(session, quote))
    while session["state"] not in ("SETTLED", "ESCALATED"):
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip().lower() in ("quit", "exit"):
            break
        r = handle_message(session, quote, text, a.llm)
        print(f"Bot: {r['reply']}\n     [{r['intent']} -> {r['outcome']}, state {r['state']}, "
              f"offer {r['offer_inr']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
