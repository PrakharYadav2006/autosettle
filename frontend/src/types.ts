// API contracts - mirror api_server.py / pricing_rules.py / negotiation.py exactly.

export type VehicleType = "car" | "motorcycle" | "scooter";
export type QuoteStatus = "APPROVED" | "REVIEW_REQUIRED" | "ESCALATED" | "NO_PAYOUT";
export type ChatState = "OPEN" | "CONFIRMING" | "SETTLED" | "ESCALATED" | "NO_OFFER";
export type Intent = "ACCEPT" | "NEGOTIATE" | "EXPLAIN" | "ESCALATE" | "OTHER";
export type Severity = "low" | "medium" | "high";

/** Claim lifecycle shown on the dashboard. */
export type ClaimPhase = "DRAFT" | "PROCESSING" | "ASSESSED" | "AWAITING_DECISION" | "SETTLED" | "ESCALATED";

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

export interface DemoSession {
  token: string;
  mode: "demo";
  policy: Policy;
  other_demo_policies: Policy[];
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
  settlement_range?: SettlementRange;
}

export interface Quote {
  status: QuoteStatus;
  reasons: string[];
  line_items: LineItem[];
  totals: PricingBreakdown | Record<string, never>;
  vehicle: Record<string, unknown>;
  notes: string[];
  authorized: boolean;
  requires_human: boolean;
  currency: "INR";
  pricing_data: string;
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
}

/** Customer view of the offer - never contains settlement limits or negotiation steps. */
export interface OfferBlock {
  status: QuoteStatus;
  state: ChatState;
  current_offer_inr: number | null;
  insurer_payable_inr: number | null;
  can_accept: boolean;
  can_reconsider: boolean;
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
  demo: boolean;
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
}

