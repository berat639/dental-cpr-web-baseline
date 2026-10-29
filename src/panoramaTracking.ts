import type { ArchResult, Pt } from './arch/arch';
import type { Volume } from './volume';

const smoothstep = (x: number) => { const t = Math.max(0, Math.min(1, x)); return t * t * (3 - 2 * t); };
// A 6 mm panoramic trough can recover modest inferior drift without chasing
// the outer cortical edge. Larger shifts are a mask/geometry ambiguity, so
// the whole optional track is rejected rather than creating a discontinuity.
const MAX_SAFE_TRACK_SHIFT_MM = 12;

/**
 * An offset smaller than 12 mm can still fold a tight bend inside out. Check
 * the actual sampled surface: adjacent columns must advance along the original
 * tangent. Also limit along-arch offset changes to 1 mm/mm; steeper transitions
 * shear the panorama when a supported mask segment abruptly falls back to zero.
 * This is a rendering regularity check, not a measure of anatomical accuracy.
 */
export function isRegularPanoramaTrack(arch: ArchResult, offsets: Float32Array, nz: number): boolean {
  const cols = arch.points.length;
  if (cols < 2 || offsets.length !== cols * nz) return false;
  for (let z = 0; z < nz; z++) {
    const row = z * cols;
    for (let c = 0; c < cols; c++) {
      const offset = offsets[row + c];
      if (!Number.isFinite(offset)) return false;
      if (!c) continue;
      const p = arch.points[c], prev = arch.points[c - 1];
      const n = arch.normals[c], prevN = arch.normals[c - 1], prevOffset = offsets[row + c - 1];
      const dx = p[0] - prev[0], dy = p[1] - prev[1], distance = Math.hypot(dx, dy);
      if (distance < 1e-6 || Math.abs(offset - prevOffset) > distance + 1e-6) return false;
      const trackedDx = dx + n[0] * offset - prevN[0] * prevOffset;
      const trackedDy = dy + n[1] * offset - prevN[1] * prevOffset;
      if (trackedDx * dx + trackedDy * dy <= 1e-6 * distance * distance) return false;
    }
  }
  return true;
}

/**
 * Follow the mandibular body below the dental arch. This is geometry processing
 * on the CPU, not another inference. The original dental curve and cross-section
 * planes remain unchanged. Only the panorama's sampling centre moves *within*
 * those planes; the axial and cross overlays display that actual centre.
 *
 * A 1 mm grid is enough for the smooth offsets. Unsupported, tiny, clipped or
 * remote mask components cannot move the curve. Above the dental transition the
 * offset is exactly zero (the mandible model says nothing about the maxilla).
 */
export function trackMandible(v: Volume, arch: ArchResult, anchorZ: number): void {
  delete arch.panorama;
  if (!v.mask || v.mask.length !== v.data.length || arch.points.length < 3 || Math.abs(v.direction[8]) < 0.8) return;
  const nc = Math.max(3, Math.ceil(arch.length / 1.5) + 1);
  const nr = Math.max(2, Math.ceil((v.nz - 1) * v.sz) + 1);
  const cols = arch.points.length, values = new Float32Array(nc * nr), valid = new Uint8Array(nc * nr);
  const limit = 32, step = 0.5, samples = Math.round(2 * limit / step) + 1;
  const superior = Math.sign(v.direction[8]);
  const isMandible = v.maskSchema === 'dental-segmentator-v1'
    ? (m: number) => m === 2 || m === 4 || m === 5
    : (m: number) => m > 0;
  for (let c = 0; c < nc; c++) {
    const index = Math.round(c * (cols - 1) / (nc - 1)), p = arch.points[index], n = arch.normals[index];
    const pixels = new Int32Array(samples).fill(-1);
    for (let k = 0; k < samples; k++) {
      const offset = -limit + k * step, x = Math.round((p[0] + n[0] * offset) / v.sx), y = Math.round((p[1] + n[1] * offset) / v.sy);
      if (x >= 0 && x < v.nx && y >= 0 && y < v.ny) pixels[k] = x + v.nx * y;
    }
    for (let r = 0; r < nr; r++) {
      const z = Math.round(r * (v.nz - 1) / (nr - 1));
      if ((anchorZ - z) * v.sz * superior <= 8) continue;
      const base = z * v.nx * v.ny;
      let bestScore = Infinity, bestOffset = 0;
      for (let start = 0; start < samples; start++) {
        if (pixels[start] < 0 || !isMandible(v.mask[base + pixels[start]])) continue;
        let end = start, gap = 0, hits = 0;
        for (let k = start; k < samples; k++) {
          if (pixels[k] >= 0 && isMandible(v.mask[base + pixels[k]])) { end = k; gap = 0; hits++; }
          else if (++gap > 3) break; // tolerate ≤1.5 mm holes, not separate bones
        }
        const lo = -limit + start * step, hi = -limit + end * step, width = hi - lo;
        if (start > 0 && end < samples - 1 && width >= 2 && width <= 30 && hits * step >= 2) {
          const center = (lo + hi) / 2;
          const distance = lo <= 0 && hi >= 0 ? 0 : Math.min(Math.abs(lo), Math.abs(hi));
          const score = distance + 0.2 * Math.abs(center);
          if (score < bestScore) { bestScore = score; bestOffset = center; }
        }
        start = Math.max(start, end);
      }
      if (bestScore < Infinity) { values[r * nc + c] = bestOffset; valid[r * nc + c] = 1; }
    }
  }
  // Smooth only locally supported tracks; an isolated mask island is not evidence.
  const smoothed = new Float32Array(values.length);
  for (let r = 0; r < nr; r++) for (let c = 0; c < nc; c++) {
    let weight = 0, total = 0, possible = 0;
    for (let dr = -2; dr <= 2; dr++) for (let dc = -2; dc <= 2; dc++) {
      const rr = r + dr, cc = c + dc;
      if (rr < 0 || rr >= nr || cc < 0 || cc >= nc) continue;
      const w = (3 - Math.abs(dr)) * (3 - Math.abs(dc)); possible += w;
      if (valid[rr * nc + cc]) { weight += w; total += w * values[rr * nc + cc]; }
    }
    if (weight >= possible * 0.6 && valid[r * nc + c]) smoothed[r * nc + c] = total / weight;
  }
  const offsets = new Float32Array(cols * v.nz);
  let maxOffsetMm = 0;
  for (let z = 0; z < v.nz; z++) {
    const blend = smoothstep(((anchorZ - z) * v.sz * superior - 8) / 12);
    if (!blend) continue;
    const rf = z * (nr - 1) / Math.max(1, v.nz - 1), r0 = Math.floor(rf), r1 = Math.min(nr - 1, r0 + 1), ry = rf - r0;
    for (let c = 0; c < cols; c++) {
      const cf = c * (nc - 1) / (cols - 1), c0 = Math.floor(cf), c1 = Math.min(nc - 1, c0 + 1), cx = cf - c0;
      const a = smoothed[r0 * nc + c0] * (1 - cx) + smoothed[r0 * nc + c1] * cx;
      const b = smoothed[r1 * nc + c0] * (1 - cx) + smoothed[r1 * nc + c1] * cx;
      const offset = (a * (1 - ry) + b * ry) * blend;
      offsets[z * cols + c] = offset; maxOffsetMm = Math.max(maxOffsetMm, Math.abs(offset));
    }
  }
  if (maxOffsetMm >= 0.5 && maxOffsetMm <= MAX_SAFE_TRACK_SHIFT_MM && isRegularPanoramaTrack(arch, offsets, v.nz))
    arch.panorama = { offsets, nz: v.nz, anchorZ, maxOffsetMm };
}

export function panoramaOffset(arch: ArchResult, column: number, z: number, enabled = true): number {
  const field = enabled && arch.panorama;
  if (!field) return 0;
  const cols = arch.points.length, c = Math.max(0, Math.min(cols - 1, column)), r = Math.max(0, Math.min(field.nz - 1, z));
  const c0 = Math.floor(c), c1 = Math.min(cols - 1, c0 + 1), r0 = Math.floor(r), r1 = Math.min(field.nz - 1, r0 + 1);
  const x = c - c0, y = r - r0, data = field.offsets;
  return (data[r0 * cols + c0] * (1 - x) + data[r0 * cols + c1] * x) * (1 - y)
    + (data[r1 * cols + c0] * (1 - x) + data[r1 * cols + c1] * x) * y;
}

export function panoramaPoint(arch: ArchResult, index: number, z: number, enabled = true): Pt {
  const p = arch.points[index], n = arch.normals[index], offset = panoramaOffset(arch, index, z, enabled);
  return [p[0] + n[0] * offset, p[1] + n[1] * offset];
}
