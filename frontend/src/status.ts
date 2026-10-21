import type { ClaimPhase, DamageFinding, QuoteStatus } from "./types";
import type { Tone } from "./components/ui";

export function claimStatus(phase: ClaimPhase, quote?: QuoteStatus): { label: string; tone: Tone } {
  switch (phase) {
    case "PROCESSING": return { label: "Assessing", tone: "neutral" };
    case "AWAITING_DECISION": return { label: "Offer ready", tone: "teal" };
    case "SETTLED": return { label: "Settled (demo)", tone: "teal" };
    case "ESCALATED": return { label: "Human review", tone: "amber" };
    case "ASSESSED": return quote === "NO_PAYOUT" ? { label: "No payout", tone: "neutral" } : { label: "Assessed", tone: "neutral" };
    default: return { label: "Draft", tone: "neutral" };
  }
}

/** Only claim "verified" when the backend actually verified it. */
export function findingStatus(d: DamageFinding): { label: string; tone: Tone } {
  // The pricing engine's status is authoritative when present.
  switch (d.status) {
    case "rejected": return { label: "Ruled out", tone: "neutral" };
    case "uncertain_review": return { label: "Needs review", tone: "amber" };
    case "unsupported_review": return { label: "Needs inspection", tone: "amber" };
    case "duplicate_or_overlap": return { label: "Overlaps another panel", tone: "amber" };
    case "confirmed_priceable": return d.vlm_confirmed === true ? { label: "Verified", tone: "teal" }
      : d.video_status === "confirmed" ? { label: `Seen in ${d.frames_seen?.length ?? 2} frames`, tone: "teal" }
      : { label: "Included", tone: "teal" };
  }
  if (d.vlm_confirmed === false) return { label: "Not damage", tone: "neutral" };
  if (d.video_status === "inconsistent" || d.video_status === "weak") return { label: "Uncertain", tone: "amber" };
  if (d.needs_review) return { label: "Needs review", tone: "amber" };
  if (d.vlm_confirmed === true) return { label: "Verified", tone: "teal" };
  if (d.video_status === "confirmed") return { label: `Seen in ${d.frames_seen?.length ?? 2} frames`, tone: "teal" };
  if (d.severity_source === "vlm_only") return { label: "Needs review", tone: "amber" };
  return { label: "Detected", tone: "neutral" };
}

export function nowTime(): string {
  return new Date().toLocaleTimeString("en-IN", { hour: "numeric", minute: "2-digit" });
}
