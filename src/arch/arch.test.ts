import { describe, expect, it } from 'vitest';
import { computeAngularArch, computeArch, computeSplineArch, fitQuadratic } from './arch';
import { assessArchEvidence } from '../cpr';

describe('fitQuadratic', () => {
  it('recovers exact coefficients', () => {
    const U = [-3, -2, -1, 0, 1, 2, 3, 4];
    const V = U.map((u) => 0.5 * u * u - 2 * u + 7);
    const [a, b, c] = fitQuadratic(U, V);
    expect(a).toBeCloseTo(0.5);
    expect(b).toBeCloseTo(-2);
    expect(c).toBeCloseTo(7);
  });
});

describe('computeArch', () => {
  // edentulous-like: gap in the middle of the data
  const pts: [number, number][] = [];
  for (let x = -30; x <= 30; x += 0.5) {
    if (Math.abs(x) < 8) continue;
    pts.push([x, 0.02 * x * x + 10]);
  }
  const arch = computeArch(pts, 100);

  it('returns unit normals perpendicular to tangents', () => {
    for (let i = 0; i < 100; i++) {
      expect(Math.hypot(...arch.normals[i])).toBeCloseTo(1);
      const d = arch.tangents[i][0] * arch.normals[i][0] + arch.tangents[i][1] * arch.normals[i][1];
      expect(d).toBeCloseTo(0);
    }
  });

  it('spaces points equally by arc length', () => {
    const d = (i: number) => Math.hypot(arch.points[i + 1][0] - arch.points[i][0], arch.points[i + 1][1] - arch.points[i][1]);
    expect(d(0)).toBeCloseTo(d(50), 1);
  });

  it('bridges the gap', () => {
    const mid = arch.points.reduce((b, p) => (Math.abs(p[0]) < Math.abs(b[0]) ? p : b));
    expect(mid[1]).toBeCloseTo(10, 0);
  });
});

import { splineArch, computeArch as _ca } from './arch';
describe('splineArch / extension', () => {
  it('passes through its control points', () => {
    const ctrl: [number, number][] = [[0, 0], [10, 8], [20, 10], [30, 8], [40, 0]];
    const a = splineArch(ctrl, 400);
    for (const c of ctrl) expect(Math.min(...a.points.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1])))).toBeLessThan(0.5);
  });
  it('extendMm lengthens the arch', () => {
    const pts: [number, number][] = [];
    for (let x = -20; x <= 20; x += 1) pts.push([x, 0.02 * x * x]);
    expect(_ca(pts, 200, 10).length).toBeGreaterThan(_ca(pts, 200, 0).length + 15);
  });
});

describe('computeSplineArch', () => {
  it('uses local centre evidence and bridges an edentulous gap', () => {
    const evidence: [number, number][] = [];
    for (let x = -45; x <= 45; x += 1) {
      if (Math.abs(x) < 9) continue;
      const y = 0.018 * x * x + 7 + 2 * Math.sin(x / 14);
      for (let thickness = -2; thickness <= 2; thickness++) evidence.push([x, y + thickness]);
    }
    // A small, distant metal-like cluster should not redraw the whole curve.
    for (let i = 0; i < 8; i++) evidence.push([38 + i / 3, -40]);
    const arch = computeSplineArch(evidence, 180, 5);
    expect(arch.points).toHaveLength(180);
    const middle = arch.points.reduce((best, point) => Math.abs(point[0]) < Math.abs(best[0]) ? point : best);
    expect(middle[1]).toBeGreaterThan(4);
    expect(middle[1]).toBeLessThan(14);
  });
});

describe('assessArchEvidence', () => {
  it('accepts a curve continuously supported by evidence', () => {
    const arch = splineArch([[0, 0], [10, 0], [20, 0]], 41);
    const evidence = Array.from({ length: 21 }, (_, x) => [x, 0] as [number, number]);
    const result = assessArchEvidence(arch, evidence);
    expect(result.supportFraction).toBeGreaterThan(0.99);
    expect(result.longestUnsupportedMm).toBe(0);
  });

  it('flags a long unsupported bridge between separated structures', () => {
    const arch = splineArch([[0, 0], [10, 0], [20, 0]], 41);
    const evidence: [number, number][] = [[0, 0], [1, 0], [19, 0], [20, 0]];
    const result = assessArchEvidence(arch, evidence);
    expect(result.supportFraction).toBeLessThan(0.5);
    expect(result.longestUnsupportedMm).toBeGreaterThan(10);
  });
});

describe('computeAngularArch', () => {
  // Tooth line: horseshoe (front at +y) with a 12 mm thick bone band lingual to the teeth.
  const teeth: [number, number][] = [], bone: [number, number][] = [];
  for (let k = 0; k <= 300; k++) {
    const t = (k / 300) * Math.PI;
    const x = 28 * Math.cos(t), y = 30 * Math.sin(t);
    const nx = Math.cos(t) / 28, ny = Math.sin(t) / 30, nl = Math.hypot(nx, ny);
    for (let w = -1; w <= 1; w += 0.5) teeth.push([x + (nx / nl) * w, y + (ny / nl) * w]);
    for (let w = 0; w <= 12; w += 1) bone.push([x - (nx / nl) * w, y - (ny / nl) * w]);
  }
  for (let y = -20; y < 0; y += 0.5) for (let w = 0; w <= 8; w++) { bone.push([28 - w, y]); bone.push([-28 + w, y]); }

  it('follows the tooth line, not the centre of the thick bone band', () => {
    const arch = computeAngularArch(bone, teeth, 300, 0);
    const apex = arch.points.reduce((b, p) => (p[1] > b[1] ? p : b));
    expect(apex[1]).toBeGreaterThan(27); // tooth apex y=30; bone-band centre would be ~24
    for (const [x, y] of arch.points.filter((p) => p[1] > 15)) {
      expect(Math.abs(Math.hypot(x / 28, y / 30) - 1)).toBeLessThan(0.1);
    }
  });
});
