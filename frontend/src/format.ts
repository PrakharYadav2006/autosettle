/** Indian grouping: 123456 -> ₹1,23,456 */
export function inr(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return "₹" + Math.round(n).toLocaleString("en-IN");
}

export function title(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function methodLabel(m: string): string {
  const map: Record<string, string> = {
    REPLACE: "Replace",
    REPAINT: "Repaint panel",
    TOUCH_UP: "Polish / touch-up",
    DENT_REPAIR: "Denting + paint",
    PDR: "Paintless dent repair",
    PLASTIC_WELD: "Plastic weld",
  };
  return map[m] ?? title(m.toLowerCase());
}

export function shortDate(iso?: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

export function cx(...c: (string | false | null | undefined)[]): string {
  return c.filter(Boolean).join(" ");
}

/** Backend reason strings -> readable sentences (wording only, never numbers). */
export function prettyReason(r: string): string {
  return r
    .replace(/^suspicious: /, "Vehicle check: ")
    .replace(/^([a-z ]+)-([a-z ]+): /, (_m, part: string, dmg: string) => `${title(part)} (${dmg}): `)
    .replace(/'([a-z_]+)'/g, (_m, k: string) => k.replace(/_/g, " "))
    .replace(/_/g, " ");
}
