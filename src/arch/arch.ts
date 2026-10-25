export type Pt = [number, number];

export interface ArchResult {
  /** Arc-length equidistant points, in mm. */
  points: Pt[];
  /** Unit normals (perpendicular to tangent), in mm space. */
  normals: Pt[];
  /** Unit tangents. */
  tangents: Pt[];
  /** Arc length of the whole curve (mm). */
  length: number;
  /** Warnings are advisory only; they are not a clinical confidence score. */
  warnings?: string[];
  controls?: Pt[];
  /** Fresh, case-local panorama offsets; never persisted or reused for another load. */
  panorama?: { offsets: Float32Array; nz: number; anchorZ: number; maxOffsetMm: number };
}

/** Solve 3x3 linear system A·x = b with Cramer's rule. */
export function solve3x3(A: number[][], b: number[]): [number, number, number] {
  const det = (m: number[][]) =>
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const d = det(A);
  if (Math.abs(d) < 1e-12) throw new Error('Singular matrix in quadratic fit');
  const col = (k: number) => A.map((row, i) => row.map((v, j) => (j === k ? b[i] : v)));
  return [det(col(0)) / d, det(col(1)) / d, det(col(2)) / d];
}

/**
 * Least-squares quadratic fit  v = a·u² + b·u + c.
 * Inputs are centred internally for numerical stability.
 */
export function fitQuadratic(U: number[], V: number[]): [number, number, number] {
  const n = U.length;
  if (n < 3) throw new Error('Need at least 3 points');
  const mu = U.reduce((s, x) => s + x, 0) / n;
  let s1 = 0, s2 = 0, s3 = 0, s4 = 0, t0 = 0, t1 = 0, t2 = 0;
  for (let i = 0; i < n; i++) {
    const u = U[i] - mu, v = V[i];
    const u2 = u * u;
    s1 += u; s2 += u2; s3 += u2 * u; s4 += u2 * u2;
    t0 += v; t1 += u * v; t2 += u2 * v;
  }
  // Normal equations for [a, b, c] (centred): 
  const [a, b, c] = solve3x3(
    [[s4, s3, s2], [s3, s2, s1], [s2, s1, n]],
    [t2, t1, t0],
  );
  // Un-centre: v = a(u-mu)² + b(u-mu) + c
  return [a, b - 2 * a * mu, a * mu * mu - b * mu + c];
}

function residual(U: number[], V: number[], k: [number, number, number]): number {
  let s = 0;
  for (let i = 0; i < U.length; i++) {
    const e = k[0] * U[i] * U[i] + k[1] * U[i] + k[2] - V[i];
    s += e * e;
  }
  return s / U.length;
}

/** Resample a dense polyline to `count` arc-length equidistant points with tangents/normals. */
export function resampleCurve(dense: Pt[], count: number): ArchResult {
  const N = dense.length - 1;
  const cum: number[] = [0];
  for (let i = 1; i <= N; i++) cum.push(cum[i - 1] + Math.hypot(dense[i][0] - dense[i - 1][0], dense[i][1] - dense[i - 1][1]));
  const length = cum[N];
  const out: ArchResult = { points: [], normals: [], tangents: [], length };
  let j = 0;
  for (let i = 0; i < count; i++) {
    const target = (length * i) / (count - 1);
    while (j < N - 1 && cum[j + 1] < target) j++;
    const seg = cum[j + 1] - cum[j] || 1;
    const t = (target - cum[j]) / seg;
    const p: Pt = [dense[j][0] + t * (dense[j + 1][0] - dense[j][0]), dense[j][1] + t * (dense[j + 1][1] - dense[j][1])];
    let tx = dense[j + 1][0] - dense[j][0], ty = dense[j + 1][1] - dense[j][1];
    const l = Math.hypot(tx, ty) || 1;
    tx /= l; ty /= l;
    out.points.push(p);
    out.tangents.push([tx, ty]);
    out.normals.push([-ty, tx]);
  }
  return out;
}

/**
 * Fit an anatomical parabola to bone points (mm) and return equidistant
 * (arc-length) points with tangents and normals. Automatically chooses
 * whether the arch is y = f(x) or x = f(y) based on residual.
 * `extendMm` lengthens both ends (so the posterior teeth/ramus are not cut off).
 */
export function computeArch(points: Pt[], count = 300, extendMm = 0): ArchResult {
  if (points.length < 3) throw new Error('Not enough bone points to fit arch');
  const X = points.map((p) => p[0]);
  const Y = points.map((p) => p[1]);
  const kyx = fitQuadratic(X, Y); // y = f(x)
  const kxy = fitQuadratic(Y, X); // x = f(y)
  const yOfX = residual(X, Y, kyx) <= residual(Y, X, kxy);
  const [U, k] = yOfX ? [X, kyx] : [Y, kxy];
  const uMin = U.reduce((m, v) => Math.min(m, v), Infinity) - extendMm;
  const uMax = U.reduce((m, v) => Math.max(m, v), -Infinity) + extendMm;

  const f = (u: number): Pt => {
    const v = k[0] * u * u + k[1] * u + k[2];
    return yOfX ? [u, v] : [v, u];
  };
  const N = 2000;
  const dense: Pt[] = [];
  for (let i = 0; i <= N; i++) dense.push(f(uMin + ((uMax - uMin) * i) / N));
  return resampleCurve(dense, count);
}

/** Catmull-Rom spline through user control points (mm), resampled by arc length. */
export function splineArch(ctrl: Pt[], count = 400): ArchResult {
  if (ctrl.length < 2) throw new Error('Need at least 2 control points');
  const P = [ctrl[0], ...ctrl, ctrl[ctrl.length - 1]];
  const dense: Pt[] = [];
  const per = 60;
  for (let i = 1; i < P.length - 2; i++) {
    const [p0, p1, p2, p3] = [P[i - 1], P[i], P[i + 1], P[i + 2]];
    for (let s = 0; s < per; s++) {
      const t = s / per, t2 = t * t, t3 = t2 * t;
      const c = (a: number, b: number, cc: number, d: number) =>
        0.5 * (2 * b + (-a + cc) * t + (2 * a - 5 * b + 4 * cc - d) * t2 + (-a + 3 * b - 3 * cc + d) * t3);
      dense.push([c(p0[0], p1[0], p2[0], p3[0]), c(p0[1], p1[1], p2[1], p3[1])]);
    }
  }
  dense.push(ctrl[ctrl.length - 1]);
  return resampleCurve(dense, count);
}

const median = (values: number[]) => {
  const ordered = [...values].sort((a, b) => a - b);
  const mid = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[mid] : (ordered[mid - 1] + ordered[mid]) / 2;
};

/**
 * Patient-specific arch estimate from a cloud of dental/alveolar evidence.
 *
 * Unlike the legacy global parabola, this finds the dominant in-plane axis,
 * takes robust centre samples along that axis, smooths them, then interpolates
 * a cubic spline. Empty bins bridge edentulous gaps without pulling the whole
 * arch toward a metal artefact.
 */
export function computeSplineArch(points: Pt[], count = 400, extendMm = 0): ArchResult {
  if (points.length < 12) throw new Error('Not enough dental evidence to estimate an arch');
  const mx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
  const my = points.reduce((sum, p) => sum + p[1], 0) / points.length;
  let xx = 0, xy = 0, yy = 0;
  for (const [x, y] of points) {
    const dx = x - mx, dy = y - my;
    xx += dx * dx; xy += dx * dy; yy += dy * dy;
  }
  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  const ux = Math.cos(angle), uy = Math.sin(angle);
  const vx = -uy, vy = ux;
  const projected = points.map(([x, y]) => ({ u: (x - mx) * ux + (y - my) * uy, v: (x - mx) * vx + (y - my) * vy }));
  const uMin = Math.min(...projected.map((p) => p.u));
  const uMax = Math.max(...projected.map((p) => p.u));
  const span = uMax - uMin;
  if (span < 8) throw new Error('Dental evidence is too narrow to estimate an arch');
  const bins = Math.max(9, Math.min(25, Math.round(span / 5)));
  const values = Array.from({ length: bins }, () => [] as number[]);
  for (const point of projected) {
    const i = Math.min(bins - 1, Math.max(0, Math.floor(((point.u - uMin) / span) * bins)));
    values[i].push(point.v);
  }
  const ctrl: Pt[] = [];
  for (let i = 0; i < bins; i++) {
    if (values[i].length < 2) continue;
    const u = uMin + ((i + 0.5) / bins) * span;
    ctrl.push([u, median(values[i])]);
  }
  if (ctrl.length < 4) throw new Error('Dental evidence is too fragmented to estimate an arch');
  // One pass of robust local smoothing preserves the U shape while suppressing isolated metal streaks.
  const smooth = ctrl.map(([u], i) => {
    const neighborhood = ctrl.slice(Math.max(0, i - 1), Math.min(ctrl.length, i + 2)).map((p) => p[1]);
    return [u, median(neighborhood)] as Pt;
  });
  if (extendMm > 0) {
    const left = smooth[0], nextLeft = smooth[1], right = smooth[smooth.length - 1], prevRight = smooth[smooth.length - 2];
    const extend = (a: Pt, b: Pt, sign: number): Pt => {
      const dx = a[0] - b[0], dy = a[1] - b[1], length = Math.hypot(dx, dy) || 1;
      return [a[0] + sign * extendMm * dx / length, a[1] + sign * extendMm * dy / length];
    };
    smooth.unshift(extend(left, nextLeft, 1));
    smooth.push(extend(right, prevRight, 1));
  }
  const worldCtrl = smooth.map(([u, v]) => [mx + u * ux + v * vx, my + u * uy + v * vy] as Pt);
  const result = splineArch(worldCtrl, count);
  const missing = bins - ctrl.length;
  if (missing > Math.ceil(bins / 3)) result.warnings = ['Weak evidence across part of the arch; review the control points.'];
  return result;
}

/**
 * Polar arch fit. The arch is swept by angle around a point inside its concavity, taking a robust radius per
 * angular bin. This stays correct at the anterior curve and along the steep posterior segments, where binning
 * along one principal axis collapses (many points share the same u). `preferred` (tooth crowns) wins over the
 * broad `points` cloud (bone mask) wherever it exists, so the curve follows the teeth, not the thick bone band.
 */
export function computeAngularArch(points: Pt[], preferred: Pt[] = [], count = 400, extendMm = 0): ArchResult {
  if (points.length < 12) throw new Error('Not enough dental evidence to estimate an arch');
  const mx = points.reduce((s, p) => s + p[0], 0) / points.length;
  const my = points.reduce((s, p) => s + p[1], 0) / points.length;
  let xx = 0, xy = 0, yy = 0;
  for (const [x, y] of points) { const dx = x - mx, dy = y - my; xx += dx * dx; xy += dx * dy; yy += dy * dy; }
  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  const ux = Math.cos(angle), uy = Math.sin(angle), vx = -uy, vy = ux;
  const toUV = (p: Pt) => ({ u: (p[0] - mx) * ux + (p[1] - my) * uy, v: (p[0] - mx) * vx + (p[1] - my) * vy });
  const pu = points.map(toUV);
  let uMin = Infinity, uMax = -Infinity;
  for (const p of pu) { uMin = Math.min(uMin, p.u); uMax = Math.max(uMax, p.u); }
  const span = uMax - uMin;
  if (span < 8) throw new Error('Dental evidence is too narrow to estimate an arch');
  const uc = (uMin + uMax) / 2;
  const mid = pu.filter((p) => Math.abs(p.u - uc) < span * 0.15).map((p) => p.v);
  const ends = pu.filter((p) => Math.abs(p.u - uc) > span * 0.35).map((p) => p.v);
  if (mid.length < 3 || ends.length < 3) throw new Error('Dental evidence is too fragmented to estimate an arch');
  // The apex (front) side is where the central points sit relative to the arch ends.
  const vMid = median(mid), vEnd = median(ends);
  const sign = vMid >= vEnd ? 1 : -1;
  const uO = uc, vO = (vMid + vEnd) / 2;
  const polar = (p: Pt) => { const q = toUV(p); const du = q.u - uO, dv = (q.v - vO) * sign; return { phi: Math.atan2(du, dv), r: Math.hypot(du, dv) }; };
  const binRad = (4 * Math.PI) / 180;
  const bone = points.map(polar), teeth = preferred.map(polar);
  let phiMin = Infinity, phiMax = -Infinity;
  for (const p of bone) { phiMin = Math.min(phiMin, p.phi); phiMax = Math.max(phiMax, p.phi); }
  const nb = Math.max(4, Math.ceil((phiMax - phiMin) / binRad));
  const boneBins = Array.from({ length: nb }, () => [] as number[]);
  const toothBins = Array.from({ length: nb }, () => [] as number[]);
  const slot = (phi: number) => Math.min(nb - 1, Math.max(0, Math.floor(((phi - phiMin) / (phiMax - phiMin || 1)) * nb)));
  for (const p of bone) boneBins[slot(p.phi)].push(p.r);
  for (const p of teeth) toothBins[slot(p.phi)].push(p.r);
  let rows: { phi: number; r: number }[] = [];
  for (let i = 0; i < nb; i++) {
    const src = toothBins[i].length >= 3 ? toothBins[i] : boneBins[i].length >= 3 ? boneBins[i] : null;
    if (!src) continue;
    rows.push({ phi: phiMin + ((i + 0.5) / nb) * (phiMax - phiMin), r: median(src) });
  }
  if (rows.length < 5) throw new Error('Dental evidence is too fragmented to estimate an arch');
  // Median-3 then [1,2,1] smoothing: suppresses metal streaks / tooth-to-bone jumps without flattening the U.
  const med3 = rows.map((row, i) => ({ phi: row.phi, r: median(rows.slice(Math.max(0, i - 1), i + 2).map((q) => q.r)) }));
  rows = med3;
  for (let pass = 0; pass < 3; pass++) {
    const prevRows = rows;
    rows = prevRows.map((row, i) => {
      const a = prevRows[Math.max(0, i - 1)].r, b = prevRows[Math.min(prevRows.length - 1, i + 1)].r;
      return { phi: row.phi, r: (a + 2 * row.r + b) / 4 };
    });
  }
  const world = (phi: number, r: number): Pt => {
    const u = uO + r * Math.sin(phi), v = vO + sign * r * Math.cos(phi);
    return [mx + u * ux + v * vx, my + u * uy + v * vy];
  };
  const ctrl = rows.map((row) => world(row.phi, row.r));
  if (extendMm > 0) {
    const extend = (a: Pt, b: Pt): Pt => {
      const dx = a[0] - b[0], dy = a[1] - b[1], l = Math.hypot(dx, dy) || 1;
      return [a[0] + (extendMm * dx) / l, a[1] + (extendMm * dy) / l];
    };
    ctrl.unshift(extend(ctrl[0], ctrl[1]));
    ctrl.push(extend(ctrl[ctrl.length - 1], ctrl[ctrl.length - 2]));
  }
  return splineArch(ctrl, count);
}
