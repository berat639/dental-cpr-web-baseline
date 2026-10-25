import { computeAngularArch, computeSplineArch, splineArch, type ArchResult } from './arch/arch';
import { assessArchEvidence, boneMipPoints, denseThreshold, toothMipPoints, type ArchEvidenceAssessment } from './cpr';
import { assessArchPlausibility, cleanDentalEvidence, type ArchPlausibility } from './archSafety';
import type { Pt } from './arch/arch';
import type { Volume } from './volume';
import { refineDentalFocus } from './archFocus';

export interface ArchCandidate { arch: ArchResult; ev: ArchEvidenceAssessment; plaus: ArchPlausibility; score: number; zc: number; fit: 'polar' | 'spline'; }
export interface ArchSearchResult { chosen: ArchCandidate | null; best: ArchCandidate | null; anyEvidence: boolean; }

/** Points of `pts` lying within `radiusMm` of any point of `ref` (grid lookup). */
function keepNear(pts: Pt[], ref: Pt[], radiusMm: number): Pt[] {
  const cell = radiusMm, set = new Set<string>();
  for (const p of ref) set.add(`${Math.floor(p[0] / cell)},${Math.floor(p[1] / cell)}`);
  return pts.filter((p) => {
    const cx = Math.floor(p[0] / cell), cy = Math.floor(p[1] / cell);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) if (set.has(`${cx + dx},${cy + dy}`)) return true;
    return false;
  });
}

/** Fraction of `ref` points lying within `radiusMm` of the curve - how well the curve passes through the crowns. */
function crownCoverage(arch: ArchResult, ref: Pt[], radiusMm: number): number {
  if (!ref.length) return 0;
  const cell = radiusMm, grid = new Map<string, Pt[]>();
  for (const p of arch.points) {
    const k = `${Math.floor(p[0] / cell)},${Math.floor(p[1] / cell)}`;
    const b = grid.get(k); if (b) b.push(p); else grid.set(k, [p]);
  }
  let hit = 0;
  for (const p of ref) {
    const cx = Math.floor(p[0] / cell), cy = Math.floor(p[1] / cell);
    let ok = false;
    for (let dx = -1; dx <= 1 && !ok; dx++) for (let dy = -1; dy <= 1 && !ok; dy++) {
      const b = grid.get(`${cx + dx},${cy + dy}`);
      if (b) for (const q of b) if (Math.hypot(p[0] - q[0], p[1] - q[1]) <= radiusMm) { ok = true; break; }
    }
    if (ok) hit++;
  }
  return hit / ref.length;
}

export const ARCH_MIN_SUPPORT = 0.8;
export const ARCH_MAX_GAP_MM = 12;

interface Stage { step: number; zOffsetsMm: number[]; slabScales: number[]; thresholdScales: number[]; }
/** Coarse pass is ~10x cheaper and settles the common case; the detailed pass only runs when it finds nothing safe. */
const STAGES: Stage[] = [
  { step: 2, zOffsetsMm: [0, -3, 3, -6, 6], slabScales: [1, 2], thresholdScales: [1, 0.8] },
  { step: 1, zOffsetsMm: [0, -2, 2, -4, 4, -6, 6, -8, 8, -10, 10], slabScales: [1, 1.5, 2], thresholdScales: [1, 0.85, 0.7] },
];

/**
 * UI-independent arch search: several axial levels, slab thicknesses and density thresholds, two fits
 * (polar tooth-following and legacy axis-binned), evidence cleaning and anatomical plausibility.
 * The evidence gate is strict; we only try harder to pass it. Fails closed.
 */
export async function searchArch(
  v: Volume, z: number, baseSlabMm: number, useMandibleMask: boolean, isCancelled: () => boolean = () => false,
): Promise<ArchSearchResult> {
  const outLen = Math.max(240, Math.round((v.nx * v.sx + v.ny * v.sy) * 2));
  let best: ArchCandidate | null = null, chosen: ArchCandidate | null = null, anyEvidence = false;
  let chosenEvidence: Pt[] = [];
  const thrCache = new Map<string, number>();
  // Crown reference: dense crowns over a thick range around the occlusal level (a tilted head spreads the teeth over
  // several slices). Candidates are ranked by how many of these crowns the curve actually passes through.
  const halfRef = Math.max(2, Math.round(12 / v.sz));
  const rz0 = Math.max(0, z - halfRef), rz1 = Math.min(v.nz - 1, z + halfRef);
  const refThr = Math.max(700, denseThreshold(v, rz0, rz1));
  const refStep = 2, refCell = Math.max(0.5, refStep * Math.max(v.sx, v.sy));
  let crownRef = cleanDentalEvidence(toothMipPoints(v, rz0, rz1, refStep, refThr), { cellMm: refCell });
  if (useMandibleMask) {
    const refBone = boneMipPoints(v, rz0, rz1, refThr, 2);
    if (refBone.length >= 150) crownRef = keepNear(crownRef, refBone, 4);
  }
  for (const stage of STAGES) {
    const seen = new Set<string>();
    const cell = Math.max(0.5, stage.step * Math.max(v.sx, v.sy));
    for (const dz of stage.zOffsetsMm) {
      const zc = Math.max(0, Math.min(v.nz - 1, z + Math.round(dz / v.sz)));
      for (const ss of stage.slabScales) {
        const halfSlices = Math.max(1, Math.round((baseSlabMm * ss) / v.sz));
        const z0 = Math.max(0, zc - halfSlices), z1 = Math.min(v.nz - 1, zc + halfSlices);
        const key = `${zc}:${z0}:${z1}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const thrKey = `${z0}:${z1}`;
        let baseThr = thrCache.get(thrKey);
        if (baseThr === undefined) { baseThr = denseThreshold(v, z0, z1); thrCache.set(thrKey, baseThr); }
        for (const ts of stage.thresholdScales) {
          // Keep the UI responsive: yield between candidates and stop when a newer detection supersedes this one.
          await new Promise<void>((resolve) => setTimeout(resolve, 0));
          if (isCancelled()) return { chosen: null, best, anyEvidence };
          const thr = Math.max(700, baseThr * ts);
          // Streak/blob/cortex evidence is filtered to the main chain of tooth-sized structures before fitting.
          const cleanTeeth = cleanDentalEvidence(toothMipPoints(v, z0, z1, stage.step, thr), { cellMm: cell });
          const bone = useMandibleMask ? boneMipPoints(v, z0, z1, thr, stage.step) : [];
          // With an AI mandible mask, keep only crowns that belong to the mandible (drops upper teeth in the slab).
          const teeth = useMandibleMask && bone.length >= 150 ? keepNear(cleanTeeth, bone, 4) : cleanTeeth;
          const sources: { pts: Pt[]; pref: Pt[]; bonus: number }[] = [];
          // Full mask coverage guided by crowns ranks first; crowns alone (partial coverage) next; bare bone band last (edentulous / weak crowns).
          if (teeth.length >= 60) sources.push({ pts: teeth, pref: [], bonus: 0.2 });
          if (bone.length >= 150) sources.push({ pts: bone, pref: teeth.length >= 60 ? teeth : [], bonus: teeth.length >= 60 ? 0.4 : 0 });
          if (!sources.length) continue;
          anyEvidence = true;
          for (const src of sources) {
            const points = src.pts;
            const fits: { arch: ArchResult; fit: 'polar' | 'spline' }[] = [];
            try { fits.push({ arch: computeAngularArch(points, src.pref, outLen, 12), fit: 'polar' }); } catch { /* try next fit */ }
            try { fits.push({ arch: computeSplineArch(points, outLen, 12), fit: 'spline' }); } catch { /* try next fit */ }
            for (const { arch, fit } of fits) {
              const ev = assessArchEvidence(arch, points);
              // The polar (tooth-following) fit gets a bonus so a bone-centre curve never outranks it.
              const crowns = crownCoverage(arch, crownRef, 2.5);
              const score = ev.supportFraction - Math.min(1, ev.longestUnsupportedMm / 60) + (fit === 'polar' ? 0.3 : 0) + src.bonus + 2 * crowns;
              const plaus = assessArchPlausibility(arch, { aiMask: useMandibleMask });
              const cand: ArchCandidate = { arch, ev, plaus, score: plaus.ok ? score : score - 1, zc, fit };
              if (!best || cand.score > best.score) best = cand;
              const passes = plaus.ok && ev.supportFraction >= ARCH_MIN_SUPPORT && ev.longestUnsupportedMm <= ARCH_MAX_GAP_MM;
              if (passes && (!chosen || cand.score > chosen.score)) { chosen = cand; chosenEvidence = points; }
            }
          }
        }
      }
    }
    if (chosen) break;
  }
  const selected = chosen as ArchCandidate | null;
  if (selected) {
    // Use the same small set of handles the user sees. Dense polar fits can introduce
    // short S bends at crown/bone transitions, rotating the section planes away
    // from the jaw. Accept the simpler curve only if the source evidence still
    // supports it; never smooth an arch into a plausible-looking unsupported U.
    const original = selected.arch;
    for (const handleCount of [9, 11, 13]) {
      const controls = Array.from({ length: handleCount }, (_, i) => original.points[Math.round(i * (original.points.length - 1) / (handleCount - 1))]);
      const arch = splineArch(controls, original.points.length);
      const ev = assessArchEvidence(arch, chosenEvidence), plaus = assessArchPlausibility(arch, { aiMask: useMandibleMask });
      let maxDeviation = 0;
      for (const p of arch.points) {
        let distance = Infinity;
        for (const q of original.points) distance = Math.min(distance, Math.hypot(p[0] - q[0], p[1] - q[1]));
        maxDeviation = Math.max(maxDeviation, distance);
      }
      if (plaus.ok && maxDeviation <= 4 && ev.supportFraction >= Math.max(ARCH_MIN_SUPPORT, selected.ev.supportFraction - 0.03)
        && ev.longestUnsupportedMm <= ARCH_MAX_GAP_MM
        && crownCoverage(arch, crownRef, 2.5) >= crownCoverage(original, crownRef, 2.5) * 0.85) {
        arch.controls = controls;
        chosen = { ...selected, arch, ev, plaus };
        break;
      }
    }
  }
  if (chosen && !isCancelled()) {
    // Broad mandible support can miss incisors when metal dominates the global
    // crown threshold. Refine locally using original voxels only after the
    // initial curve and its editable control points have passed the search.
    const focus = refineDentalFocus(v, chosen.arch, z);
    if (focus.applied) {
      const plaus = assessArchPlausibility(focus.arch, { aiMask: useMandibleMask });
      if (plaus.ok) chosen = { ...chosen, arch: focus.arch, plaus, ev: assessArchEvidence(focus.arch, chosenEvidence) };
    }
  }
  return { chosen, best, anyEvidence };
}
