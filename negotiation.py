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

OFFER_LADDER = [Decimal("0.80"), Decimal("0.88"), Decimal("0.95")]   # last = ceiling (MAX_OFFER_RATIO)
MAX_TURNS = 12
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
                             r"kam|zyada|jyada|badha\w*|aur do|negotiat\w*|counter|unfair|reject)\b", re.I)),
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
    if quote["status"] == "APPROVED" and pr.validate_chatbot_action({"type": "present_offer"}, quote)["accepted"]:
        s["ladder_inr"] = ladder(quote)
        s["offer_inr"] = s["ladder_inr"][0]
        s["state"] = "OPEN"
    return s


def explain_text(quote: Dict) -> str:
    t = quote["totals"]
    parts = []
    for l in quote["line_items"]:
        parts.append(f"{l['part'].replace('_', ' ')} ({l['damage']}, {l['severity']}): "
                     f"{l['method'].replace('_', ' ').lower()} {inr(l['garage_cost'])}")
    lines = "; ".join(parts) if parts else "no priced items"
    txt = (f"Here is how we calculated it. Repairs: {lines}. Garage estimate {inr(t['garage_estimate_inr'])}, "
           f"minus depreciation {inr(t['depreciation_inr'])} and the policy deductible "
           f"{inr(t['deductible_inr'])}, gives {inr(t['insurer_payable_inr'])}.")
    if t.get("settlement_range"):
        txt += (" An instant cash settlement is offered at a little below that amount because it is paid "
                "today with no garage visit or survey.")
    return txt


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

    if st in ("SETTLED", "ESCALATED"):
        msg = (f"This claim is already settled at {inr(session['settled_amount_inr'])}. The payment is being processed."
               if st == "SETTLED" else "Your claim is with a human adjuster, who will contact you shortly.")
        return {"outcome": "CLOSED", "amount": session.get("settled_amount_inr"), "template": msg}

    if intent == "ESCALATE" or session["turns"] >= MAX_TURNS:
        session["state"] = "ESCALATED"
        return {"outcome": "ESCALATED", "amount": None,
                "template": "I've passed your claim to a human adjuster. They will review it and contact you "
                            "within one working day. Your claim details and photos are saved."}

    if st == "NO_OFFER":
        if intent == "EXPLAIN" and quote.get("line_items"):
            return {"outcome": "EXPLAINED", "amount": None, "template": explain_text(quote)
                    + f" However, this claim needs a review first ({'; '.join(quote['reasons'][:2])})."}
        why = "; ".join(quote.get("reasons", [])[:2]) or "it needs a human check"
        if intent == "EXPLAIN":
            return {"outcome": "EXPLAINED", "amount": None,
                    "template": f"This claim can't be priced automatically because {why}. "
                                "A human adjuster will check it. Type 'human' to be contacted now."}
        return {"outcome": "NO_OFFER", "amount": None,
                "template": f"I can't make an instant cash offer on this claim because {why}. "
                            "A human adjuster will review it. Type 'human' if you'd like to be contacted now."}

    offer = session["offer_inr"]
    lad = session["ladder_inr"]

    if st == "CONFIRMING":
        if intent == "ACCEPT" and not NEGATION.search(text):
            if not authorise(offer, quote):
                session["state"] = "ESCALATED"
                return {"outcome": "ESCALATED", "amount": None,
                        "template": "I couldn't authorise this amount, so a human adjuster will finish your claim."}
            session["state"] = "SETTLED"
            session["settled_amount_inr"] = offer
            return {"outcome": "SETTLED", "amount": offer,
                    "template": f"Done! Your claim is settled at {inr(offer)}. The amount will be credited to "
                                "your registered bank account within 48 hours. Thank you."}
        session["state"] = "OPEN"          # anything else cancels the confirmation
        st = "OPEN"
        if intent == "ACCEPT":
            intent = "OTHER"

    if intent == "ACCEPT":
        session["state"] = "CONFIRMING"
        return {"outcome": "CONFIRM_REQUEST", "amount": offer,
                "template": f"Great. Please reply CONFIRM to accept {inr(offer)} as full and final settlement "
                            "of this claim."}

    if intent == "EXPLAIN":
        return {"outcome": "EXPLAINED", "amount": offer,
                "template": explain_text(quote) + f" Our current offer is {inr(offer)}."}

    if intent == "NEGOTIATE":
        if asked is not None and asked <= offer:   # they asked for less than we offer
            session["state"] = "CONFIRMING"
            return {"outcome": "CONFIRM_REQUEST", "amount": offer,
                    "template": f"Our offer of {inr(offer)} already covers that. Reply CONFIRM to accept it."}
        if session["tier"] + 1 >= len(lad):        # already at the ceiling
            session["state"] = "ESCALATED"
            return {"outcome": "ESCALATED", "amount": None,
                    "template": f"{inr(offer)} is the highest amount I'm able to authorise for an instant "
                                "settlement. I've passed your request to a human adjuster, who can review the "
                                "claim in more detail and will contact you soon."}
        session["tier"] += 1
        new = lad[session["tier"]]
        if asked is not None and offer < asked < new:
            new = asked                            # meet them at their own (lower) number
        if not authorise(new, quote):
            session["state"] = "ESCALATED"
            return {"outcome": "ESCALATED", "amount": None,
                    "template": "I couldn't authorise a higher amount, so a human adjuster will take over."}
        session["offer_inr"] = new
        last = session["tier"] + 1 >= len(lad)
        return {"outcome": "COUNTER_OFFER", "amount": new,
                "template": f"I understand. I can increase the offer to {inr(new)}"
                            + (", which is the maximum I can authorise for an instant settlement." if last else ".")
                            + " Reply ACCEPT if that works for you."}

    return {"outcome": "OFFER", "amount": offer,
            "template": f"Your instant cash settlement offer is {inr(offer)}. Reply ACCEPT to take it, ask "
                        "'why' to see the breakdown, or tell me if you think it's too low."}


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
    vals |= {48, 24, 1, 3}   # "48 hours", "one working day", sentence counts
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
            "state": session["state"], "tier": session["tier"], "reply": reply, "wording": wording}
    session["history"].append(turn)
    audit(session["claim_id"], turn)
    return {"reply": reply, "intent": intent, "outcome": d["outcome"], "state": session["state"],
            "offer_inr": session.get("offer_inr"), "settled_amount_inr": session.get("settled_amount_inr"),
            "tier": session["tier"], "wording": wording}


def opening_message(session: Dict, quote: Dict) -> str:
    if session["state"] == "OPEN":
        return (f"Your claim has been assessed. We can settle it right now for {inr(session['offer_inr'])}, "
                "paid to your bank account within 48 hours. Reply ACCEPT, ask 'why' for the breakdown, or tell me "
                "if you think it's too low.")
    if quote["status"] == "NO_PAYOUT":
        return ("Your claim has been assessed, but the repair cost is below your policy deductible, so there is "
                "no payable amount. Ask 'why' to see the breakdown.")
    return ("Your claim has been assessed and needs a quick review by our team before an offer can be made. "
            "Ask 'why' for details or type 'human' to be contacted.")


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
