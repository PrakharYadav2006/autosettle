// API contracts - mirror api_server.py / pricing_rules.py / negotiation.py exactly.

export type VehicleType = "car" | "motorcycle" | "scooter";
export type QuoteStatus = "APPROVED" | "REVIEW_REQUIRED" | "ESCALATED" | "NO_PAYOUT";
export type ChatState = "OPEN" | "CONFIRMING" | "SETTLED" | "ESCALATED" | "NO_OFFER";
export type Intent = "ACCEPT" | "NEGOTIATE" | "EXPLAIN" | "ESCALATE" | "OTHER";
export type Severity = "low" | "medium" | "high";
/** pricing_rules.FINDING_STATUSES - what happened to each damage finding. */
export type FindingStatusCode = "confirmed_priceable" | "uncertain_review" | "unsupported_review" | "rejected" | "duplicate_or_overlap";
/** complete = everything found is priced or ruled out; partial = some items are pending review. */
export type EstimateScope = "complete" | "partial" | "none";

/** Claim lifecycle shown on the dashboard. */
export type ClaimPhase = "DRAFT" | "PROCESSING" | "ASSESSED" | "AWAITING_DECISION" | "SETTLED" | "PARTIALLY_SETTLED" | "ESCALATED";

export interface Policy {
  plate: string;
  policy_no: string;
  holder: string;
  make: string;
  model: string;
  year: number;
  engine_cc: number;
  vehicle_type: VehicleType;
  zero_dep: boolean;
  policy_active: boolean;
  policy_start: string;
  policy_end: string;
  deductible_inr: number;
  endorsements: string[];
  auto_approval_limit_inr: number;
}

export type DemoTier = "auto" | "economy" | "mid" | "luxury" | "two_wheeler";
export const DEMO_TIERS: { value: DemoTier; label: string }[] = [
  { value: "auto", label: "Auto / unknown" }, { value: "economy", label: "Economy" },
  { value: "mid", label: "Mid-range" }, { value: "luxury", label: "Luxury" }, { value: "two_wheeler", label: "Two-wheeler" },
];

/** A policyholder session (policy from the record) or a Demo Mode session (no policy at all). */
export interface Session {
  token: string;
  mode: "policyholder" | "demo";
  policy: Policy | null;
  demo?: { tier: DemoTier; tier_label: string; note: string };
}
/** @deprecated old name */
export type DemoSession = Session;

export interface DemoPricing {
  tier_requested: DemoTier;
  tier_requested_label: string;
  tier_used: string | null;
  tier_source: string | null;
  note: string;
}

export interface PriceSource {
  type: "scraped" | "demo_estimate" | "none";
  price?: number;
  values?: number[];
  n?: number;
  sites?: string[];
  urls?: string[];
  price_type?: string;
  scraped_at?: string;
  stale?: boolean;
  tier_price?: number;
  multiplier?: number;
}

/** One priced repair line (pricing_rules.cost_line + price_claim). */
export interface LineItem {
  part: string;            // canonical key, e.g. "front_bumper"
  damage: string;          // canonical, e.g. "broken"
  severity: Severity;
  severity_source?: string;
  severity_reasons?: string[] | null;
  method: string;          // REPLACE / REPAINT / PDR ...
  material: "metal" | "plastic" | "glass";
  part_cost: number;
  labour: number;
  paint: number;
  garage_cost: number;
  depreciation_pct: number;
  depreciation: number;
  part_price_source: PriceSource;
  confidence: number;
  low_confidence?: boolean;
  also_covers?: string[];
  evidence?: string | null;
  /** components taken from a checked garage estimate: {labour: {ours, garage}, ...} */
  garage_estimate_applied?: Record<string, { ours: number; garage: number }>;
}

export type EstimateLineStatus = "accepted" | "within_our_estimate" | "above_cap" | "unsupported"
  | "method_mismatch" | "not_compared" | "duplicate" | "not_checked" | "compared";

export interface EstimateCheck {
  n: number;
  time: string;
  result: string | null;
  offer_before: number | null;
  offer_after: number | null;
  cap_pct: number | null;
  problems: string[];
  lines: { description: string; type: string; amount: number; status: EstimateLineStatus; reason: string;
           ours?: number; cap_amount?: number }[];
  authenticity: string;
}

export interface SettlementRange {
  opening_offer_inr: number;
  maximum_offer_inr: number;
  open_ratio: string;
  max_ratio: string;
}

/** Pricing breakdown block (quote.totals). Empty object when nothing could be priced. */
export interface PricingBreakdown {
  garage_estimate_inr: number;
  depreciation_inr: number;
  deductible_inr: number;
  insurer_payable_inr: number;
  zero_depreciation_cover: boolean;
  price_tier: string;
  vehicle_type: VehicleType;
  make_used: string;
  parts_priced_from_real_data: number;
  parts_priced_from_demo_estimates: number;
  estimate_scope: EstimateScope;
  demo_mode?: boolean;
  tier_requested?: string;
  tier_source?: string;
  items_included: number;
  items_pending: number;
  settlement_range?: SettlementRange;
}

/** Customer-safe pending item (no confidences, no reason codes). */
export interface PendingItem {
  part: string;
  price_key: string | null;
  damage: string;
  status: FindingStatusCode;
  reason: string;
}

/** Customer view of the quote (api_server.customer_quote). */
export interface Quote {
  status: QuoteStatus;
  reasons: string[];
  plain_reasons: string[];
  pending_items: PendingItem[];
  ruled_out_count: number;
  /** Demo Mode scoped offer: exactly which findings the offer covers. */
  offer_scope?: { partial: boolean; included: { part: string; damage: string }[]; excluded: { part: string; damage: string }[] };
  line_items: LineItem[];
  totals: PricingBreakdown | Record<string, never>;
  vehicle: Record<string, unknown>;
  authorized: boolean;
  requires_human: boolean;
  currency: "INR";
}

/** Box in 0..1 image coordinates. */
export interface BoundingBox {
  id: string;
  kind: "part" | "damage" | "vlm";
  label: string;
  part?: string;
  price_key: string | null;
  confidence: number | null;
  severity?: Severity | null;
  bbox: [number, number, number, number];
}

export interface Viewer {
  image: string;
  width: number;
  height: number;
  boxes: BoundingBox[];
  caption?: string;
}

export interface DamageFinding {
  part: string;
  price_key?: string | null;
  damage: string | null;
  severity?: Severity | null;
  severity_source?: string | null;
  severity_reasons?: string[] | null;
  detector_confidence?: number;
  vlm_confirmed?: boolean | null;
  vlm_agreement?: number | null;              // staff view only
  severity_facts?: Record<string, boolean | string> | null;  // staff view only
  part_uncertain?: boolean;
  video_status?: string;
  frames_seen?: number[];
  needs_review: boolean;
  status?: FindingStatusCode;
  /** set when the finding came from a proof photo, e.g. "proof photo 1" */
  source?: string | null;
  status_code?: string | null;             // staff view only
  vlm_verification?: string | null;        // staff view only
  vlm_valid_views?: number | null;         // staff view only
}

/** Customer view of the offer - never contains settlement limits or negotiation steps. */
export interface OfferBlock {
  status: QuoteStatus;
  state: ChatState;
  current_offer_inr: number | null;
  insurer_payable_inr: number | null;
  estimate_scope: EstimateScope;
  offer_scope: "partial" | "complete" | null;
  /** After acceptance: "partial" = only the listed items are settled; the rest is still under review. */
  settlement_scope?: "partial" | "full" | null;
  can_accept: boolean;
  can_reconsider: boolean;
  /** Evidence-based negotiation: the offer changes only when uploaded proof changes the assessment. */
  can_submit_evidence?: boolean;
  evidence_left?: number;
  settled_amount_inr: number | null;
  decision_recorded_at: string | null;
}

export interface ClaimEvent {
  time: string;
  title: string;
  detail?: string;
  tone: "neutral" | "teal" | "amber" | "red";
}

export interface ChatMessage {
  from: "bot" | "user";
  text: string;
  time: string | null;
  outcome?: string;
  failed?: boolean;
}

export interface ClaimState {
  claim_id: string;
  media_type: "photo" | "video";
  quote: Quote;
  images: { name: string; image: string | null }[];
  viewer: Viewer | null;
  damages: DamageFinding[];
  vehicle: { type: VehicleType; mismatch: string | null };
  policy?: Policy | null;
  offer: OfferBlock;
  chat: { state: ChatState; messages: ChatMessage[] };
  events: ClaimEvent[];
  evidence_retained: boolean;
  mode?: "policyholder" | "demo";
  demo: boolean;
  demo_pricing?: DemoPricing | null;
  estimate_checks?: EstimateCheck[];
}

export interface ChatReply {
  claim_id: string;
  reply: string;
  outcome: string;
  state: ChatState;
  offer: OfferBlock;
  events: ClaimEvent[];
}

export interface Health {
  ok: boolean;
  models_loaded: boolean;
  demo_mode?: boolean;
}

