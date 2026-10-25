import type { ArchResult, Pt } from './arch/arch';
import { sample, type Volume } from './volume';
import { panoramaOffset } from './panoramaTracking';

/** `thicknessMm` is the total focal-trough thickness, not a half-width. */
export interface Slab { thicknessMm: number; mode: 'mean' | 'max'; trackMandible?: boolean; }

/** Collect bone voxel (mm) positions: Z-MIP of thresholded mask within a z slab. */
export function boneMipPoints(v: Volume, zMin: number, zMax: number, threshold = 500, step = 2): Pt[] {
  const pts: Pt[] = [];
  for (let y = 0; y < v.ny; y += step)
    for (let x = 0; x < v.nx; x += step) {
      let hit = false;
      for (let z = zMin; z <= zMax && !hit; z++) {
        const i = x + v.nx * (y + v.ny * z);
        hit = v.mask ? v.mask[i] > 0 : v.data[i] > threshold;
      }
      if (hit) pts.push([x * v.sx, y * v.sy]);
    }
  return pts;
}

export const TOOTH_HU = 1400;

/**
 * CBCT values are vendor/protocol dependent. Estimate a dense-tissue threshold
 * inside the selected slab instead of assuming every source has CT-calibrated HU.
 */
export function denseThreshold(v: Volume, zMin: number, zMax: number): number {
  const values: number[] = [];
  const stride = Math.max(1, Math.round(Math.max(v.nx, v.ny) / 180));
  for (let z = zMin; z <= zMax; z += Math.max(1, Math.round((zMax - zMin) / 10)))
    for (let y = 0; y < v.ny; y += stride)
      for (let x = 0; x < v.nx; x += stride) values.push(v.data[x + v.nx * (y + v.ny * z)]);
  if (values.length < 20) return TOOTH_HU;
  values.sort((a, b) => a - b);
  // At 98.5% cortical skull bone is still included in a CBCT MIP. Crown enamel
  // occupies a much smaller fraction, so use the extreme tail and never lower
  // a conventional CT series to a generic bone threshold.
  const q = values[Math.floor(values.length * 0.998)];
  return Math.max(700, q);
}

/** Axial slice with the most dense voxels = occlusal level; -1 if no dense structure is present. */
export function findOcclusalSlice(v: Volume, minCount = 200): number {
  const sample: number[] = [];
  const stride = Math.max(1, Math.round(Math.max(v.nx, v.ny, v.nz) / 120));
  for (let z = 0; z < v.nz; z += stride)
    for (let y = 0; y < v.ny; y += stride)
      for (let x = 0; x < v.nx; x += stride) sample.push(v.data[x + v.nx * (y + v.ny * z)]);
  if (sample.length < 20) return -1;
  sample.sort((a, b) => a - b);
  const threshold = Math.max(700, sample[Math.floor(sample.length * 0.998)]);
  const per = v.nx * v.ny, cnt = new Float32Array(v.nz);
  for (let z = 0; z < v.nz; z++) {
    let c = 0;
    for (let i = z * per, e = i + per; i < e; i++) if (v.data[i] > threshold) c++;
    cnt[z] = c;
  }
  const r = Math.max(1, Math.round(2 / v.sz)); // ~2 mm smoothing
  let best = -1, bestV = minCount;
  for (let z = 0; z < v.nz; z++) {
    let s = 0;
    for (let k = Math.max(0, z - r); k <= Math.min(v.nz - 1, z + r); k++) s += cnt[k];
    if (s > bestV) { bestV = s; best = z; }
  }
  return best;
}

/** Dense (tooth) voxel columns inside a slab - arch evidence that follows the crowns, not the basal bone. */
export function toothMipPoints(v: Volume, zMin: number, zMax: number, step = 2, threshold = TOOTH_HU): Pt[] {
  const pts: Pt[] = [];
  for (let y = 0; y < v.ny; y += step)
    for (let x = 0; x < v.nx; x += step) {
      for (let z = zMin; z <= zMax; z++) {
        if (v.data[x + v.nx * (y + v.ny * z)] > threshold) { pts.push([x * v.sx, y * v.sy]); break; }
      }
    }
  return pts;
}

export interface ArchEvidenceAssessment {
  /** Fraction of the proposed curve with dense dental/jaw evidence within `radiusMm`. */
  supportFraction: number;
  /** Longest uninterrupted curve span without nearby evidence. */
  longestUnsupportedMm: number;
  maxDistanceMm: number;
}

/**
 * Reject a curve that merely bridges separate dense structures through empty
 * anatomy. This is a safety gate, not a diagnostic confidence score: a passed
 * curve still remains editable and needs clinical review.
 */
export function assessArchEvidence(arch: ArchResult, evidence: Pt[], radiusMm = 3): ArchEvidenceAssessment {
  if (!evidence.length || arch.points.length < 2) return { supportFraction: 0, longestUnsupportedMm: Infinity, maxDistanceMm: Infinity };
  let supported = 0, run = 0, longestRun = 0, maxDistance = 0;
  const stepMm = arch.length / (arch.points.length - 1);
  // Spatial grid (cell = radius): only the 3x3 neighbourhood can hold a point within `radiusMm`.
  const cell = Math.max(1e-6, radiusMm), grid = new Map<number, Pt[]>();
  const key = (cx: number, cy: number) => cx * 73856093 ^ cy * 19349663;
  for (const p of evidence) {
    const k = key(Math.floor(p[0] / cell), Math.floor(p[1] / cell));
    const bucket = grid.get(k);
    if (bucket) bucket.push(p); else grid.set(k, [p]);
  }
  for (const curvePoint of arch.points) {
    const cx = Math.floor(curvePoint[0] / cell), cy = Math.floor(curvePoint[1] / cell);
    let nearest = Infinity;
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++) {
        const bucket = grid.get(key(cx + dx, cy + dy));
        if (!bucket) continue;
        for (const point of bucket) nearest = Math.min(nearest, Math.hypot(curvePoint[0] - point[0], curvePoint[1] - point[1]));
      }
    // Distances beyond the search neighbourhood are reported as the neighbourhood bound (lower bound).
    maxDistance = Math.max(maxDistance, Math.min(nearest, 2 * radiusMm));
    if (nearest <= radiusMm) { supported++; run = 0; }
    else { run++; longestRun = Math.max(longestRun, run); }
  }
  return {
    supportFraction: supported / arch.points.length,
    longestUnsupportedMm: longestRun * stepMm,
    maxDistanceMm: maxDistance,
  };
}

function slabSample(v: Volume, p: Pt, n: Pt, zVox: number, slab: Slab, offsetMm: number, maxSteps = Infinity): number {
  const halfMm = slab.thicknessMm / 2;
  // `maxSteps` gives a coarser (draft) slab while the user is dragging; the full slab is rendered once idle.
  const steps = Math.max(1, Math.min(maxSteps, Math.round(halfMm / Math.min(v.sx, v.sy))));
  let acc = slab.mode === 'max' ? -Infinity : 0, cnt = 0;
  // slab is along the tangent direction for cross-sections and along the normal for panorama
  for (let s = -steps; s <= steps; s++) {
    const o = (s / steps) * halfMm + offsetMm;
    const val = sample(v, (p[0] + n[0] * o) / v.sx, (p[1] + n[1] * o) / v.sy, zVox);
    acc = slab.mode === 'max' ? Math.max(acc, val) : acc + val;
    cnt++;
  }
  return slab.mode === 'max' ? acc : acc / cnt;
}

/** Panoramic image: columns = arch positions, rows = z (top = highest). Returns HU. */
export function renderPanorama(v: Volume, arch: ArchResult, slab: Slab, maxSteps = Infinity): { data: Float32Array; w: number; h: number } {
  const w = arch.points.length, h = v.nz;
  const data = new Float32Array(w * h);
  for (let i = 0; i < w; i++)
    for (let r = 0; r < h; r++)
      data[(h - 1 - r) * w + i] = slabSample(v, arch.points[i], arch.normals[i], r, slab, panoramaOffset(arch, i, r, slab.trackMandible), maxSteps);
  return { data, w, h };
}

export interface ArchFrame { p: Pt; n: Pt; t: Pt }

const unit = (v: Pt): Pt => { const l = Math.hypot(v[0], v[1]) || 1; return [v[0] / l, v[1] / l]; };

/** Point, normal and tangent at arc-length `sMm` (clamped to the curve); the arch is equidistant in mm. */
export function archFrameAt(arch: ArchResult, sMm: number): ArchFrame {
  const last = arch.points.length - 1;
  const f = Math.max(0, Math.min(last, (sMm / arch.length) * last));
  const i0 = Math.floor(f), i1 = Math.min(last, i0 + 1), u = f - i0;
  const mix = (a: Pt, b: Pt): Pt => [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];
  return {
    p: mix(arch.points[i0], arch.points[i1]),
    n: unit(mix(arch.normals[i0], arch.normals[i1])),
    t: unit(mix(arch.tangents[i0], arch.tangents[i1])),
  };
}

/** Arc-length (mm) of arch point `i`. */
export const archS = (arch: ArchResult, i: number): number => (i / Math.max(1, arch.points.length - 1)) * arch.length;

/** 1-based cross-section number for a position, counted in `spacingMm` steps from the arch start. */
export const sectionNumber = (sMm: number, spacingMm: number): number => Math.round(sMm / spacingMm) + 1;

function renderCrossFrame(v: Volume, frame: ArchFrame, halfWidthMm: number, thicknessMm: number, mode: 'mean' | 'max', maxSteps = Infinity): { data: Float32Array; w: number; h: number } {
  const w = Math.round((2 * halfWidthMm) / Math.min(v.sx, v.sy)), h = v.nz;
  const data = new Float32Array(w * h);
  const slab: Slab = { thicknessMm, mode };
  for (let c = 0; c < w; c++) {
    const off = -halfWidthMm + (c / (w - 1)) * 2 * halfWidthMm;
    const q: Pt = [frame.p[0] + frame.n[0] * off, frame.p[1] + frame.n[1] * off];
    for (let r = 0; r < h; r++) data[(h - 1 - r) * w + c] = slabSample(v, q, frame.t, r, slab, 0, maxSteps);
  }
  return { data, w, h };
}

/** Cross-section perpendicular to the arch at index i: columns = offset along normal, rows = z. */
export function renderCross(v: Volume, arch: ArchResult, i: number, halfWidthMm: number, thicknessMm: number, mode: 'mean' | 'max'): { data: Float32Array; w: number; h: number } {
  return renderCrossFrame(v, { p: arch.points[i], n: arch.normals[i], t: arch.tangents[i] }, halfWidthMm, thicknessMm, mode);
}

/** Cross-section at an arbitrary arc-length position (mm). */
export function renderCrossAt(v: Volume, arch: ArchResult, sMm: number, halfWidthMm: number, thicknessMm: number, mode: 'mean' | 'max', maxSteps = Infinity): { data: Float32Array; w: number; h: number } {
  return renderCrossFrame(v, archFrameAt(arch, sMm), halfWidthMm, thicknessMm, mode, maxSteps);
}

export function toCanvas(canvas: HTMLCanvasElement, img: { data: Float32Array; w: number; h: number }, wl: { center: number; width: number }, yScale = 1) {
  // Resizing a canvas reallocates and clears it; only do it when the size really changes.
  if (canvas.width !== img.w || canvas.height !== img.h) { canvas.width = img.w; canvas.height = img.h; }
  canvas.dataset.yScale = String(yScale);
  const ctx = canvas.getContext('2d')!;
  const id = ctx.createImageData(img.w, img.h);
  const lo = wl.center - wl.width / 2, scale = 255 / wl.width;
  const px = new Uint32Array(id.data.buffer), src = img.data;
  for (let i = 0; i < src.length; i++) {
    const g = Math.max(0, Math.min(255, (src[i] - lo) * scale)) | 0;
    px[i] = 0xff000000 | (g << 16) | (g << 8) | g; // little-endian RGBA, opaque grey
  }
  ctx.putImageData(id, 0, 0);
}
