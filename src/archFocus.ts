import { splineArch, type ArchResult, type Pt } from './arch/arch';
import { sample, type Volume } from './volume';

export interface FocusRefinement {
  arch: ArchResult;
  applied: boolean;
  maxShiftMm: number;
  supportedColumns: number;
  beforeCoverage: number;
  afterCoverage: number;
}

const quantile = (values: number[], fraction: number) => {
  values.sort((a, b) => a - b);
  return values[Math.floor((values.length - 1) * fraction)] ?? 0;
};

/** Refine an already plausible arch against a small tube of original dental data.
 * Metal is capped before aggregation, so it cannot set the threshold for incisors.
 * This is one bounded CPU pass, with no inference, persistence or render-time work.
 * A single smooth curve is retained: rows are never independently warped.
 */
export function refineDentalFocus(v: Volume, original: ArchResult, anchorZ: number): FocusRefinement {
  const unchanged = (): FocusRefinement => ({ arch: original, applied: false, maxShiftMm: 0, supportedColumns: 0, beforeCoverage: 0, afterCoverage: 0 });
  if (original.points.length < 3 || original.length < 20 || !original.controls) return unchanged();
  const count = Math.ceil(original.length) + 1, step = original.length / (count - 1);
  const offsets = Array.from({ length: 33 }, (_, i) => -8 + i * 0.5);
  const coverage = (profile: number[], d: number) => profile.reduce((sum, w, j) => sum + (Math.abs(offsets[j] - d) <= 3 ? w : 0), 0);
  const shifts = new Float64Array(count), valid = new Uint8Array(count);
  const profiles: number[][] = [];
  const pointAt = (i: number) => Math.round(i * (original.points.length - 1) / (count - 1));
  for (let i = 0; i < count; i++) {
    const k = pointAt(i), p = original.points[k], n = original.normals[k];
    const rows: number[][] = [];
    for (let dz = -12; dz <= 12; dz++) {
      const z = anchorZ + dz / v.sz;
      if (z < 0 || z > v.nz - 1) continue;
      rows.push(offsets.map(d => Math.min(2500, Math.max(0, sample(v, (p[0] + n[0] * d) / v.sx, (p[1] + n[1] * d) / v.sy, z)))));
    }
    if (rows.length < 6) return unchanged();
    const profile = offsets.map((_, j) => Math.max(0, quantile(rows.map(row => row[j]), 0.75) - 800));
    profiles.push(profile);
    const mass = profile.reduce((a, b) => a + b, 0);
    // Require dental signal on multiple heights, not a single bright streak.
    const centres: number[] = [];
    for (const row of rows) {
      const w = row.map(h => Math.max(0, h - 800));
      const total = w.reduce((a, b) => a + b, 0);
      if (total >= 1200 && Math.max(...row) >= 1500) centres.push(w.reduce((sum, h, j) => sum + h * offsets[j], 0) / total);
    }
    if (mass < 1200 || Math.max(...profile) < 700 || centres.length < Math.ceil(rows.length / 4)) continue;
    const spread = quantile([...centres], 0.75) - quantile([...centres], 0.25);
    if (spread > 4) continue;
    const centroid = profile.reduce((sum, w, j) => sum + w * offsets[j], 0) / mass;
    // Preserve the extended posterior tails; the tube there contains ramus, not crowns.
    const endDistance = Math.min(i * step, original.length - i * step);
    const taper = Math.max(0, Math.min(1, (endDistance - 12) / 12));
    valid[i] = 1;
    const shift = Math.max(-6, Math.min(6, centroid)) * taper;
    if (coverage(profile, shift) >= coverage(profile, 0) * 1.1) shifts[i] = shift;
  }
  const supportedColumns = valid.reduce((a, b) => a + b, 0);
  if (supportedColumns < 10) return unchanged();
  // Include unsupported zero shifts in smoothing; do not extrapolate across missing teeth.
  const smoothed = shifts.map((_, i) => {
    let total = 0, weight = 0;
    const radius = Math.ceil(9 / step);
    for (let j = Math.max(0, i - radius); j <= Math.min(count - 1, i + radius); j++) {
      const w = Math.exp(-0.5 * ((j - i) * step / 3) ** 2);
      total += shifts[j] * w; weight += w;
    }
    return total / weight;
  });
  const controls = original.controls;
  const moved: Pt[] = controls.map((p, controlIndex) => {
    if (controlIndex === 0 || controlIndex === controls.length - 1) return [...p];
    let k = 0, nearest = Infinity;
    for (let j = 0; j < original.points.length; j++) {
      const distance = Math.hypot(p[0] - original.points[j][0], p[1] - original.points[j][1]);
      if (distance < nearest) { nearest = distance; k = j; }
    }
    const i = Math.round(k * (count - 1) / (original.points.length - 1));
    const d = valid[i] && coverage(profiles[i], smoothed[i]) >= coverage(profiles[i], 0) * 1.1 ? smoothed[i] : 0;
    return [p[0] + original.normals[k][0] * d, p[1] + original.normals[k][1] * d];
  });
  const candidate = splineArch(moved, original.points.length);
  candidate.controls = moved;
  for (const p of candidate.points) {
    let nearest = Infinity;
    for (const q of original.points) nearest = Math.min(nearest, Math.hypot(p[0] - q[0], p[1] - q[1]));
    if (nearest > 6.5) return unchanged();
  }
  // Map the candidate back to each original normal. Require forward progression
  // and bounded displacement before comparing dental-band coverage.
  let maxShiftMm = 0, before = 0, after = 0, total = 0;
  const regions = Array.from({ length: Math.ceil(original.length / 15) + 1 }, () => ({ before: 0, after: 0, total: 0, count: 0 }));
  for (let i = 0; i < count; i++) {
    const k = pointAt(i), p = original.points[k], n = original.normals[k];
    let nearest = Infinity, q = p, tangent = original.tangents[k];
    for (let j = 0; j < candidate.points.length; j++) {
      const distance = Math.hypot(p[0] - candidate.points[j][0], p[1] - candidate.points[j][1]);
      if (distance < nearest) { nearest = distance; q = candidate.points[j]; tangent = candidate.tangents[j]; }
    }
    if (nearest > 6.5 || tangent[0] * original.tangents[k][0] + tangent[1] * original.tangents[k][1] < 0.5) return unchanged();
    maxShiftMm = Math.max(maxShiftMm, nearest);
    const d = (q[0] - p[0]) * n[0] + (q[1] - p[1]) * n[1];
    if (!valid[i]) continue;
    const region = regions[Math.floor(i * step / 15)]; region.count++;
    for (let j = 0; j < offsets.length; j++) {
      const w = profiles[i][j]; total += w; region.total += w;
      if (Math.abs(offsets[j]) <= 3) { before += w; region.before += w; }
      if (Math.abs(offsets[j] - d) <= 3) { after += w; region.after += w; }
    }
  }
  // An anterior gain must not conceal a material loss in a posterior region.
  if (regions.some(r => r.count >= 5 && r.after + 0.05 * r.total < 0.9 * r.before)) return unchanged();
  // Do not change an already centred case just because a centroid is nonzero.
  if (maxShiftMm < 0.75 || total === 0 || after < before * 1.05) return unchanged();
  return { arch: candidate, applied: true, maxShiftMm, supportedColumns, beforeCoverage: before / total, afterCoverage: after / total };
}
