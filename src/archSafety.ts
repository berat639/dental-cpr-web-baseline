import type { Pt } from './arch/arch';
import type { ArchResult } from './arch/arch';

/**
 * Remove metal-streak / cortical-plate evidence and keep only the main chain of tooth-sized structures.
 *
 * 1. Rasterise to a coarse grid (cell >= 0.5 mm).
 * 2. Morphological opening (~1 mm): streaks and thin plates vanish, crowns survive.
 * 3. Drop components smaller than a tooth cross-section.
 * 4. Group survivors that lie within `linkMm` of each other (an arch is a chain of neighbouring teeth)
 *    and keep the group with the most dental area. Isolated dense blobs elsewhere (condyle, implants far
 *    away, skull base) no longer pull the curve.
 */
export function cleanDentalEvidence(points: Pt[], opts: { cellMm?: number; openMm?: number; minAreaMm2?: number; linkMm?: number } = {}): Pt[] {
  if (points.length < 20) return points;
  const cell = Math.max(0.5, opts.cellMm ?? 0.5), openR = Math.max(1, Math.round((opts.openMm ?? 1) / cell));
  const minCells = Math.max(3, Math.round((opts.minAreaMm2 ?? 3) / (cell * cell)));
  const linkR = Math.max(1, Math.round((opts.linkMm ?? 8) / cell / 2));
  let maxX = 0, maxY = 0;
  for (const p of points) { if (p[0] > maxX) maxX = p[0]; if (p[1] > maxY) maxY = p[1]; }
  const W = Math.ceil(maxX / cell) + 2, H = Math.ceil(maxY / cell) + 2;
  let grid: Uint8Array = new Uint8Array(W * H);
  for (const p of points) grid[Math.floor(p[1] / cell) * W + Math.floor(p[0] / cell)] = 1;

  const disc = (r: number) => { const o: [number, number][] = []; for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) if (dx * dx + dy * dy <= r * r) o.push([dx, dy]); return o; };
  const morph = (src: Uint8Array, r: number, erode: boolean): Uint8Array => {
    const out = new Uint8Array(W * H), offs = disc(r);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let hit = erode;
      for (const [dx, dy] of offs) {
        const xx = x + dx, yy = y + dy;
        const v = xx >= 0 && yy >= 0 && xx < W && yy < H ? src[yy * W + xx] : 0;
        if (erode ? !v : v) { hit = !erode; break; }
      }
      out[y * W + x] = hit ? 1 : 0;
    }
    return out;
  };
  grid = morph(morph(grid, openR, true), openR, false);

  const label = (src: Uint8Array): { ids: Int32Array; sizes: number[] } => {
    const ids = new Int32Array(W * H).fill(-1), sizes: number[] = [];
    for (let s = 0; s < W * H; s++) {
      if (!src[s] || ids[s] >= 0) continue;
      const id = sizes.length; let count = 0; const stack = [s]; ids[s] = id;
      while (stack.length) {
        const c = stack.pop()!; count++;
        const cx = c % W, cy = (c - cx) / W;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const xx = cx + dx, yy = cy + dy;
          if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
          const n = yy * W + xx;
          if (src[n] && ids[n] < 0) { ids[n] = id; stack.push(n); }
        }
      }
      sizes.push(count);
    }
    return { ids, sizes };
  };
  const first = label(grid);
  const kept = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) if (grid[i] && first.sizes[first.ids[i]] >= minCells) kept[i] = 1;
  const linked = label(morph(kept, linkR, false));
  const area = new Map<number, number>();
  for (let i = 0; i < W * H; i++) if (kept[i]) area.set(linked.ids[i], (area.get(linked.ids[i]) ?? 0) + 1);
  let bestId = -1, bestArea = 0;
  for (const [id, a] of area) if (a > bestArea) { bestArea = a; bestId = id; }
  if (bestId < 0) return [];
  const out: Pt[] = [];
  for (let i = 0; i < W * H; i++) if (kept[i] && linked.ids[i] === bestId) out.push([((i % W) + 0.5) * cell, (Math.floor(i / W) + 0.5) * cell]);
  return out;
}

export interface ArchPlausibility { ok: boolean; reason?: string; lengthMm: number; depthMm: number; chordMm: number; turningDeg: number; }

/** Anatomical sanity of a candidate curve: rejects loops, spirals, skull outlines and absurd sizes. */
export function assessArchPlausibility(arch: ArchResult, opts: { aiMask?: boolean; toothLabels?: boolean } = {}): ArchPlausibility {
  // Without an AI mask the curve spans teeth only (molar to molar + short extensions): tighter bounds than the
  // mask-guided curve, which also follows the rami.
  const maxLen = opts.aiMask || opts.toothLabels ? 250 : 180, minDepthRatio = opts.toothLabels ? 0.25 : opts.aiMask ? 0.45 : 0.38;
  const pts = arch.points, n = pts.length;
  const a = pts[0], b = pts[n - 1];
  const chord = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1e-6;
  const cx = (b[0] - a[0]) / chord, cy = (b[1] - a[1]) / chord;
  let depth = 0;
  for (let i = 0; i < n; i++) depth = Math.max(depth, Math.abs((pts[i][0] - a[0]) * -cy + (pts[i][1] - a[1]) * cx));
  // Total turning is measured on an ~8 mm resampling so millimetre-scale wiggle of the fit is not mistaken for a loop.
  const stride = Math.max(1, Math.round(8 / (arch.length / (n - 1))));
  let turning = 0, prev: number | null = null;
  for (let i = stride; i < n; i += stride) {
    const th = Math.atan2(pts[i][1] - pts[i - stride][1], pts[i][0] - pts[i - stride][0]);
    if (prev !== null) { let d = th - prev; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; turning += Math.abs(d); }
    prev = th;
  }
  // A real loop crosses itself; a posterior ramus that merely curls inward does not.
  const coarse: Pt[] = [];
  for (let i = 0; i < n; i += stride) coarse.push(pts[i]);
  if ((n - 1) % stride) coarse.push(pts[n - 1]);
  const cross = (a1: Pt, a2: Pt, b1: Pt, b2: Pt) => {
    const d = (p: Pt, q: Pt, r: Pt) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
    return d(a1, a2, b1) * d(a1, a2, b2) < 0 && d(b1, b2, a1) * d(b1, b2, a2) < 0;
  };
  let selfCross = false;
  for (let i = 0; i + 1 < coarse.length && !selfCross; i++)
    for (let j = i + 2; j + 1 < coarse.length; j++)
      if (cross(coarse[i], coarse[i + 1], coarse[j], coarse[j + 1])) { selfCross = true; break; }
  const result = { lengthMm: arch.length, depthMm: depth, chordMm: chord, turningDeg: (turning * 180) / Math.PI };
  // A dental arch from molar to molar with posterior extensions: ~80-230 mm long, a single non-self-intersecting U.
  if (arch.length < 60 || arch.length > maxLen) return { ok: false, reason: `arch uzunluğu ${arch.length.toFixed(0)} mm (beklenen 60–${maxLen})`, ...result };
  if (selfCross || result.turningDeg > 450) return { ok: false, reason: `eğri kendini kesiyor/döngülü (${result.turningDeg.toFixed(0)}° dönüş)`, ...result };
  if (depth < (opts.toothLabels ? 15 : 24)) return { ok: false, reason: `eğri çok sığ (${depth.toFixed(0)} mm derinlik)`, ...result };
  if (depth / chord < minDepthRatio || depth / chord > 1.8) return { ok: false, reason: 'eğri U şekline benzemiyor', ...result };
  return { ok: true, ...result };
}
