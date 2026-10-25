import { describe, expect, it } from 'vitest';
import { computeSplineArch, type Pt } from './arch/arch';
import { archFrameAt, archS, renderCrossAt, sectionNumber } from './cpr';
import type { Volume } from './volume';

const curve: Pt[] = [];
for (let x = -30; x <= 30; x += 0.5) curve.push([x, 0.02 * x * x + 10]);
const arch = computeSplineArch(curve, 120, 0);

describe('archFrameAt', () => {
  it('matches the stored frame at exact sample positions', () => {
    for (const i of [0, 30, 60, arch.points.length - 1]) {
      const f = archFrameAt(arch, archS(arch, i));
      expect(f.p[0]).toBeCloseTo(arch.points[i][0], 6);
      expect(f.p[1]).toBeCloseTo(arch.points[i][1], 6);
    }
  });

  it('moves by the requested physical distance between neighbouring sections', () => {
    const mid = arch.length / 2;
    const a = archFrameAt(arch, mid).p, b = archFrameAt(arch, mid + 2).p;
    expect(Math.hypot(b[0] - a[0], b[1] - a[1])).toBeGreaterThan(1.9);
    expect(Math.hypot(b[0] - a[0], b[1] - a[1])).toBeLessThanOrEqual(2.0001);
  });

  it('returns unit, perpendicular normal and tangent and clamps outside the arch', () => {
    const f = archFrameAt(arch, arch.length / 3);
    expect(Math.hypot(...f.n)).toBeCloseTo(1, 5);
    expect(Math.hypot(...f.t)).toBeCloseTo(1, 5);
    expect(f.n[0] * f.t[0] + f.n[1] * f.t[1]).toBeCloseTo(0, 1);
    expect(archFrameAt(arch, -50).p).toEqual(arch.points[0]);
    expect(archFrameAt(arch, arch.length + 50).p).toEqual(arch.points[arch.points.length - 1]);
  });
});

describe('sectionNumber', () => {
  it('numbers sections in spacing steps, 1-based', () => {
    expect(sectionNumber(0, 1)).toBe(1);
    expect(sectionNumber(34, 1)).toBe(35);
    expect(sectionNumber(34, 2)).toBe(18);
    expect(sectionNumber(0.4, 0.5)).toBe(2);
  });
});

describe('renderCrossAt', () => {
  it('produces nz rows and a width of 2*half/spacing columns', () => {
    const v: Volume = {
      data: new Float32Array(20 * 20 * 6).fill(100), nx: 20, ny: 20, nz: 6, sx: 1, sy: 1, sz: 1,
      origin: [0, 0, 0], direction: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    };
    const img = renderCrossAt(v, arch, arch.length / 2, 5, 1, 'mean');
    expect(img.h).toBe(6);
    expect(img.w).toBe(10);
    expect(img.data).toHaveLength(60);
  });
});
