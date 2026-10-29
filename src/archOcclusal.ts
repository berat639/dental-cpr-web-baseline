import { splineArch, type ArchResult, type Pt } from './arch/arch';
import { assessArchEvidence, type ArchEvidenceAssessment } from './cpr';
import { assessArchPlausibility, cleanDentalEvidence, type ArchPlausibility } from './archSafety';
import type { Volume } from './volume';

export interface OcclusalArchResult {
  arch: ArchResult | null;
  evidence: Pt[];
  assessment: ArchEvidenceAssessment | null;
  plausibility: ArchPlausibility | null;
  reason?: string;
}

export const OCCLUSAL_CONTROL_POINTS = 17;

const quantile = (values: number[], q: number): number => {
  if (!values.length) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const pos = Math.max(0, Math.min(ordered.length - 1, q * (ordered.length - 1)));
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return lo === hi ? ordered[lo] : ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo);
};

const median = (values: number[]): number => quantile(values, 0.5);

/**
 * Collapses the 3-D segmentation into crown-focused dental and alveolar/ramus
 * axial presence maps.
 *
 * - For tooth labels (3 = upper, 4 = lower), restricting to the occlusal/cervical
 *   60% of each label's Z distribution prevents flaring palatal root apices
 *   (label 3) or lingually tilted root apices (label 4) from pulling the curve
 *   into the oral cavity.
 * - For the mandible (label 2), restricting to the dentoalveolar + ramus Z band
 *   avoids high condylar/coronoid tips near the zygomatic arch while keeping
 *   the full left and right ascending rami at the occlusal level.
 */
async function dentalProjection(
  v: Volume, toothLabel: number, cancelled: () => boolean,
): Promise<{ selectedCrowns: Pt[]; crowns: Pt[]; mandible: Pt[] }> {
  if (!v.mask) return { selectedCrowns: [], crowns: [], mandible: [] };
  const stride = Math.max(1, Math.round(0.5 / Math.min(v.sx, v.sy)));
  const width = Math.ceil(v.nx / stride), height = Math.ceil(v.ny / stride);
  const plane = v.nx * v.ny;

  // Pass 1: Z distribution of labels 2, 3, 4 to locate the occlusal crown band automatically.
  const zCounts2 = new Float64Array(v.nz);
  const zCounts3 = new Float64Array(v.nz);
  const zCounts4 = new Float64Array(v.nz);
  let total2 = 0, total3 = 0, total4 = 0;
  for (let z = 0; z < v.nz; z++) {
    const z0 = z * plane;
    let c2 = 0, c3 = 0, c4 = 0;
    for (let y = 0; y < v.ny; y += stride) {
      const row = z0 + y * v.nx;
      for (let x = 0; x < v.nx; x += stride) {
        const label = v.mask[row + x];
        if (label === 2) c2++;
        else if (label === 3) c3++;
        else if (label === 4) c4++;
      }
    }
    zCounts2[z] = c2; total2 += c2;
    zCounts3[z] = c3; total3 += c3;
    zCounts4[z] = c4; total4 += c4;
  }

  const zQuantile = (counts: Float64Array, total: number, q: number): number => {
    if (total <= 0) return 0;
    const target = total * q;
    let acc = 0;
    for (let z = 0; z < counts.length; z++) {
      acc += counts[z];
      if (acc >= target) return z;
    }
    return counts.length - 1;
  };

  const upperIsHighZ = total3 > 0 && total4 > 0
    ? zQuantile(zCounts3, total3, 0.5) >= zQuantile(zCounts4, total4, 0.5)
    : total4 > 0 && total2 > 0
      ? zQuantile(zCounts4, total4, 0.5) >= zQuantile(zCounts2, total2, 0.5)
      : v.direction[8] >= 0;

  let zMandLo = 0, zMandHi = v.nz - 1;
  if (total2 > 0 && (total3 > 0 || total4 > 0)) {
    const a = total4 > 0 ? zQuantile(zCounts4, total4, upperIsHighZ ? 0.10 : 0.90) : zQuantile(zCounts3, total3, upperIsHighZ ? 0.05 : 0.95);
    const b = total3 > 0 ? zQuantile(zCounts3, total3, upperIsHighZ ? 0.85 : 0.15) : zQuantile(zCounts4, total4, upperIsHighZ ? 0.98 : 0.02);
    if (Math.abs(b - a) >= 6) {
      zMandLo = Math.min(a, b);
      zMandHi = Math.max(a, b);
    }
  } else if (total2 > 0) {
    const a = zQuantile(zCounts2, total2, 0.20), b = zQuantile(zCounts2, total2, 0.80);
    if (Math.abs(b - a) >= 6) { zMandLo = Math.min(a, b); zMandHi = Math.max(a, b); }
  }

  // Track the occlusal-most Z slice per (gx, gy) column for labels 3 and 4 so flaring
  // palatal/lingual root apices can be rejected locally even when the occlusal plane is tilted.
  const z3Occl = new Int32Array(width * height).fill(upperIsHighZ ? 99999 : -99999);
  const z4Occl = new Int32Array(width * height).fill(upperIsHighZ ? -99999 : 99999);
  const has3 = new Uint8Array(width * height);
  const has4 = new Uint8Array(width * height);
  const jawHit = new Uint8Array(width * height);

  for (let z = 0; z < v.nz; z++) {
    const z0 = z * plane;
    const in2 = z >= zMandLo && z <= zMandHi;
    for (let gy = 0, y = 0; y < v.ny; gy++, y += stride) {
      const row = z0 + y * v.nx, out = gy * width;
      for (let gx = 0, x = 0; x < v.nx; gx++, x += stride) {
        const label = v.mask[row + x], idx = out + gx;
        if (label === 3) {
          has3[idx] = 1;
          if (upperIsHighZ ? z < z3Occl[idx] : z > z3Occl[idx]) z3Occl[idx] = z;
        } else if (label === 4) {
          has4[idx] = 1;
          if (upperIsHighZ ? z > z4Occl[idx] : z < z4Occl[idx]) z4Occl[idx] = z;
        } else if (label === 2 && in2) {
          jawHit[idx] = 1;
        }
      }
    }
    if ((z & 7) === 7) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      if (cancelled()) return { selectedCrowns: [], crowns: [], mandible: [] };
    }
  }

  // Separable 2-D local occlusal Z reference within a ~10 mm window.
  // Any root column whose occlusal-facing end sits > 3.5 mm (upper) or > 4.5 mm (lower)
  // apical to its neighbouring crown is a diverging root apex rather than a crown.
  const cellMm = stride * Math.min(v.sx, v.sy);
  const winR = Math.max(2, Math.round(10 / cellMm));
  const localExtrema = (src: Int32Array, takeMin: boolean): Int32Array => {
    const tmp = new Int32Array(width * height);
    const out = new Int32Array(width * height);
    for (let gy = 0; gy < height; gy++) {
      const row = gy * width;
      for (let gx = 0; gx < width; gx++) {
        let best = src[row + gx];
        const x0 = Math.max(0, gx - winR), x1 = Math.min(width - 1, gx + winR);
        for (let xx = x0; xx <= x1; xx++) {
          const val = src[row + xx];
          if (takeMin ? val < best : val > best) best = val;
        }
        tmp[row + gx] = best;
      }
    }
    for (let gy = 0; gy < height; gy++) {
      const y0 = Math.max(0, gy - winR), y1 = Math.min(height - 1, gy + winR);
      for (let gx = 0; gx < width; gx++) {
        let best = tmp[gy * width + gx];
        for (let yy = y0; yy <= y1; yy++) {
          const val = tmp[yy * width + gx];
          if (takeMin ? val < best : val > best) best = val;
        }
        out[gy * width + gx] = best;
      }
    }
    return out;
  };

  const ref3 = localExtrema(z3Occl, upperIsHighZ);
  const ref4 = localExtrema(z4Occl, !upperIsHighZ);
  const maxApical3 = Math.max(2, Math.round(3.5 / v.sz));
  const maxApical4 = Math.max(2, Math.round(4.5 / v.sz));

  const selectedHit = new Uint8Array(width * height);
  const crownHit = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const keep3 = has3[i] && Math.abs(z3Occl[i] - ref3[i]) <= maxApical3;
    const keep4 = has4[i] && Math.abs(z4Occl[i] - ref4[i]) <= maxApical4;
    if (keep3 || keep4) crownHit[i] = 1;
    if ((toothLabel === 3 && keep3) || (toothLabel === 4 && keep4)) selectedHit[i] = 1;
  }

  const rawSelected: Pt[] = [], rawCrowns: Pt[] = [], mandible: Pt[] = [];
  for (let gy = 0; gy < height; gy++) {
    for (let gx = 0; gx < width; gx++) {
      const idx = gy * width + gx;
      if (crownHit[idx] || jawHit[idx]) {
        const point: Pt = [Math.min(v.nx - 1, gx * stride) * v.sx, Math.min(v.ny - 1, gy * stride) * v.sy];
        if (selectedHit[idx]) rawSelected.push(point);
        if (crownHit[idx]) rawCrowns.push(point);
        if (jawHit[idx]) mandible.push(point);
      }
    }
  }

  // Clean isolated dental specks while keeping the entire dental arch chain intact.
  const cleanedCrowns = cleanDentalEvidence(rawCrowns, { cellMm: 0.6, openMm: 0.6, minAreaMm2: 2, linkMm: 18 });
  const crowns = cleanedCrowns.length >= 60 ? cleanedCrowns : rawCrowns;
  return { selectedCrowns: rawSelected, crowns, mandible };
}

/**
 * Fit a smooth, outlier-proof 17-point B-spline arch around the jaw concavity.
 *
 * Key properties that prevent failure across diverse scans:
 * 1. The polar origin `(uO, vO)` is placed at the posterior base of the U-arch
 *    (between the posterior ends of the rami), so radial rays intersect the left
 *    ramus, molars, incisors, and right ramus transversely. Posterior rami never
 *    collapse into a single angular bin.
 * 2. In each angular bin, any tooth/root voxel that sits > 4.5 mm lingual or
 *    > 8 mm buccal to the mandibular bone arch is rejected as an outlier
 *    (e.g. flaring palatal roots), preventing 90° inward hooks.
 * 3. Missing-tooth (edentulous) bins are smoothly bridged using the mandibular
 *    alveolar ridge and linear radial interpolation before multi-pass Laplacian
 *    smoothing, eliminating S-bends around isolated or tilted molars.
 * 4. A lateral monotonicity guard at the terminal ramus bins prevents medial
 *    condylar poles from hooking the ends of the curve inward.
 */
function fitAngularOcclusalArch(
  selectedCrowns: Pt[], crowns: Pt[], mandible: Pt[], samples: number,
): ArchResult {
  if (crowns.length + mandible.length < 80) throw new Error('Yeterli diş etiketi yok');
  const allPoints = mandible.length >= 80 ? [...crowns, ...mandible] : crowns;

  const mx = allPoints.reduce((sum, p) => sum + p[0], 0) / allPoints.length;
  const my = allPoints.reduce((sum, p) => sum + p[1], 0) / allPoints.length;
  let xx = 0, xy = 0, yy = 0;
  for (const [x, y] of allPoints) {
    const dx = x - mx, dy = y - my;
    xx += dx * dx; xy += dx * dy; yy += dy * dy;
  }
  // Fold PCA eigenvector into the horizontal (left-right) axis and clamp head yaw to ±25°
  // so dolichocephalic jaws (yy > xx) or unilateral edentulism never swap u/v by 90° or flip R/L.
  let angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  while (angle > Math.PI / 4) angle -= Math.PI / 2;
  while (angle < -Math.PI / 4) angle += Math.PI / 2;
  const maxYaw = (25 * Math.PI) / 180;
  angle = Math.max(-maxYaw, Math.min(maxYaw, angle));
  const ux = Math.cos(angle), uy = Math.sin(angle), vx = -uy, vy = ux;
  const toUV = (p: Pt) => ({
    u: (p[0] - mx) * ux + (p[1] - my) * uy,
    v: (p[0] - mx) * vx + (p[1] - my) * vy,
  });

  const pu = allPoints.map(toUV);
  let uMin = Infinity, uMax = -Infinity;
  for (const p of pu) { uMin = Math.min(uMin, p.u); uMax = Math.max(uMax, p.u); }
  const span = uMax - uMin;
  if (span < 30) throw new Error('Diş etiketi arch için çok dar');

  const uc = (uMin + uMax) / 2;
  const mid = pu.filter((p) => Math.abs(p.u - uc) < span * 0.18).map((p) => p.v);
  const ends = pu.filter((p) => Math.abs(p.u - uc) > span * 0.32).map((p) => p.v);
  if (mid.length < 3 || ends.length < 3) throw new Error('Diş etiketi kesintili');

  const vMid = median(mid), vEnd = median(ends);
  const sign = vMid >= vEnd ? 1 : -1;
  // Place polar origin at the posterior base of the arch (12th percentile of signed v).
  const vSigned = pu.map((p) => p.v * sign);
  const vOSigned = quantile(vSigned, mandible.length >= 80 ? 0.12 : 0.06);
  const uO = uc, vO = vOSigned * sign;

  const polar = (p: Pt) => {
    const q = toUV(p);
    const du = q.u - uO, dv = (q.v - vO) * sign;
    return { phi: Math.atan2(du, dv), r: Math.hypot(du, dv) };
  };

  const pSelected = selectedCrowns.map(polar);
  const pCrowns = crowns.map(polar);
  const pMand = mandible.map(polar);
  const pAll = [...pCrowns, ...pMand];

  let phiMin = Infinity, phiMax = -Infinity;
  for (const p of pAll) { phiMin = Math.min(phiMin, p.phi); phiMax = Math.max(phiMax, p.phi); }
  const phiSpan = phiMax - phiMin;
  if (phiSpan < 0.8) throw new Error('Diş etiketi arch açısı için yetersiz');

  const binRad = (3.5 * Math.PI) / 180;
  const nb = Math.max(25, Math.ceil(phiSpan / binRad));
  const selBins = Array.from({ length: nb }, () => [] as number[]);
  const crownBins = Array.from({ length: nb }, () => [] as number[]);
  const mandBins = Array.from({ length: nb }, () => [] as number[]);
  const slot = (phi: number) => Math.min(nb - 1, Math.max(0, Math.floor(((phi - phiMin) / (phiSpan || 1)) * nb)));

  for (const p of pSelected) selBins[slot(p.phi)].push(p.r);
  for (const p of pCrowns) crownBins[slot(p.phi)].push(p.r);
  for (const p of pMand) mandBins[slot(p.phi)].push(p.r);

  const minMandCount = mandible.length >= 80 ? 6 : 3;
  const binRadius: (number | null)[] = Array.from({ length: nb }, (_, i) => {
    const hasMand = mandBins[i].length >= minMandCount;
    const rMand = hasMand ? median(mandBins[i]) : null;
    const validCrowns = crownBins[i].filter((r) => rMand === null || (r >= rMand - 4.5 && r <= rMand + 8.0));
    const validSel = selBins[i].filter((r) => rMand === null || (r >= rMand - 4.5 && r <= rMand + 8.0));
    const hasCr = validCrowns.length >= 3;
    const hasSel = validSel.length >= 3;
    if (!hasCr && !hasMand) return null;
    // 60th percentile of the crown projection aligns with the occlusal crown centre rather than lingual root shadow.
    const rCr = hasSel ? 0.65 * quantile(validSel, 0.6) + 0.35 * quantile(validCrowns, 0.6) : (hasCr ? quantile(validCrowns, 0.6) : null);
    if (rCr !== null && rMand !== null) return 0.9 * rCr + 0.1 * rMand;
    return rCr ?? rMand;
  });

  // Preserve the longest contiguous angular run, allowing up to 3 empty bins (~10.5 deg) for edentulous spaces.
  let bestStart = 0, bestEnd = -1, start = 0, lastActive = -1, gap = 0, bestScore = -1;
  const closeRun = () => {
    const score = lastActive >= start ? lastActive - start + 1 : 0;
    if (score > bestScore) { bestScore = score; bestStart = start; bestEnd = lastActive; }
  };
  for (let i = 0; i < nb; i++) {
    if (binRadius[i] !== null) { lastActive = i; gap = 0; continue; }
    if (gap++ <= 2) continue;
    closeRun(); start = i + 1; lastActive = -1; gap = 0;
  }
  closeRun();
  if (bestEnd - bestStart < 8) throw new Error('Diş etiketi kesintili');

  // Collect active bins and linearly interpolate any interior gap bins so smoothing is uniform in angle.
  const rows: { phi: number; r: number }[] = [];
  for (let i = bestStart; i <= bestEnd; i++) {
    const phi = phiMin + (i / (nb - 1)) * phiSpan;
    let r = binRadius[i];
    if (r === null) {
      let prevIdx = i - 1, nextIdx = i + 1;
      while (prevIdx >= bestStart && binRadius[prevIdx] === null) prevIdx--;
      while (nextIdx <= bestEnd && binRadius[nextIdx] === null) nextIdx++;
      if (prevIdx >= bestStart && nextIdx <= bestEnd) {
        const f = (i - prevIdx) / (nextIdx - prevIdx);
        r = binRadius[prevIdx]! * (1 - f) + binRadius[nextIdx]! * f;
      } else {
        continue;
      }
    }
    rows.push({ phi, r });
  }
  if (rows.length < 5) throw new Error('Diş merkezi örnekleri yetersiz');

  // Median-3 followed by 6 passes of [1, 2, 1] Laplacian radial smoothing.
  let smoothed = rows.map((row, i) => ({
    phi: row.phi,
    r: median(rows.slice(Math.max(0, i - 1), Math.min(rows.length, i + 2)).map((q) => q.r)),
  }));
  for (let pass = 0; pass < 6; pass++) {
    const prev = smoothed;
    smoothed = prev.map((row, i) => {
      if (i === 0 || i === prev.length - 1) return row;
      return { phi: row.phi, r: (prev[i - 1].r + 2 * row.r + prev[i + 1].r) / 4 };
    });
  }

  // Convert back to (u, v) and guard the terminal 4 bins at each posterior tip against medial condylar hooks.
  const uvCtrl: Pt[] = smoothed.map(({ phi, r }) => [uO + r * Math.sin(phi), vO + sign * r * Math.cos(phi)]);
  for (let k = Math.min(4, uvCtrl.length - 1); k > 0; k--) {
    if (uvCtrl[k - 1][0] > uvCtrl[k][0]) uvCtrl[k - 1][0] = uvCtrl[k][0];
  }
  for (let k = Math.max(0, uvCtrl.length - 5); k < uvCtrl.length - 1; k++) {
    if (uvCtrl[k + 1][0] < uvCtrl[k][0]) uvCtrl[k + 1][0] = uvCtrl[k][0];
  }

  const worldCtrl: Pt[] = uvCtrl.map(([u, v]) => [mx + u * ux + v * vx, my + u * uy + v * vy]);
  const denseArch = splineArch(worldCtrl, samples);
  const last = denseArch.points.length - 1;
  const controls17: Pt[] = Array.from(
    { length: OCCLUSAL_CONTROL_POINTS },
    (_, i) => [...denseArch.points[Math.round((i * last) / (OCCLUSAL_CONTROL_POINTS - 1))]] as Pt,
  );
  const arch = splineArch(controls17, samples);
  arch.controls = controls17;
  return arch;
}

/**
 * DentalSegmentator path. It uses crown-level tooth labels (3/4) and the
 * mandibular arch (label 2), building a 17-point B-spline whose curve and
 * cross-section angles are computed directly from the 17 points.
 */
export async function searchOcclusalSplineArch(
  v: Volume, toothLabel: number, isCancelled: () => boolean = () => false,
): Promise<OcclusalArchResult> {
  if (!v.mask) return { arch: null, evidence: [], assessment: null, plausibility: null, reason: 'Diş segmentasyonu yok' };
  const projection = await dentalProjection(v, toothLabel, isCancelled);
  const evidence = projection.crowns;
  if (isCancelled()) return { arch: null, evidence: [], assessment: null, plausibility: null, reason: 'İptal edildi' };
  try {
    const samples = Math.max(240, Math.round((v.nx * v.sx + v.ny * v.sy) * 2));
    const arch = fitAngularOcclusalArch(projection.selectedCrowns, evidence, projection.mandible, samples);
    const supportPoints = [...evidence, ...projection.mandible];
    const assessment = assessArchEvidence(arch, supportPoints, 5);
    const plausibility = assessArchPlausibility(arch, { aiMask: true, toothLabels: true });
    if (!plausibility.ok) return { arch: null, evidence, assessment, plausibility, reason: plausibility.reason };
    if (assessment.supportFraction < 0.60 || assessment.longestUnsupportedMm > 24)
      return { arch: null, evidence, assessment, plausibility, reason: 'Diş kanıtı eğri boyunca yeterince sürekli değil' };
    return { arch, evidence, assessment, plausibility };
  } catch (error) {
    return { arch: null, evidence, assessment: null, plausibility: null, reason: (error as Error).message };
  }
}
