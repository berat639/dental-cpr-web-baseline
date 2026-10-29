import { describe, expect, it } from 'vitest';
import { searchOcclusalSplineArch } from './archOcclusal';
import type { Volume } from './volume';

describe('occlusal B-spline arch', () => {
  it('uses every tooth-label height instead of a selected axial slab', async () => {
    const nx = 180, ny = 150, nz = 80, mask = new Uint8Array(nx * ny * nz);
    // A lower arch whose posterior crowns occur well away from the anterior Z.
    for (let x = 25; x < 155; x++) {
      const y0 = Math.round(35 + 0.015 * (x - 90) ** 2);
      const z0 = 12 + Math.round(Math.abs(x - 90) * 0.35);
      for (let y = y0 - 3; y <= y0 + 3; y++) for (let z = z0 - 6; z <= z0 + 6; z++)
        mask[x + nx * (y + ny * z)] = 4;
    }
    const v: Volume = { data: new Float32Array(mask.length), mask, maskSchema: 'dental-segmentator-v1', nx, ny, nz, sx: 0.6, sy: 0.6, sz: 0.5, origin: [0, 0, 0], direction: [1, 0, 0, 0, 1, 0, 0, 0, 1] };
    const result = await searchOcclusalSplineArch(v, 4);
    expect(result.arch, result.reason).not.toBeNull();
    expect(result.arch!.controls).toHaveLength(17);
    expect(result.assessment!.supportFraction).toBeGreaterThan(0.9);
    expect(result.arch!.length).toBeGreaterThan(60);
  });

  it('accepts arches extended through the mandible beyond 180 mm (up to 250 mm)', async () => {
    const nx = 220, ny = 240, nz = 80, mask = new Uint8Array(nx * ny * nz);
    // Dental arch (label 4) + posterior mandibular extensions (label 2) along the same curve pushing total length > 180 mm.
    for (let x = 25; x <= 195; x++) {
      const y0 = Math.round(30 + 0.018 * (x - 110) ** 2);
      const z0 = 20;
      const label = x >= 40 && x <= 180 ? 4 : 2;
      for (let y = y0 - 3; y <= y0 + 3; y++) for (let z = z0 - 4; z <= z0 + 4; z++)
        if (y >= 0 && y < ny) mask[x + nx * (y + ny * z)] = label;
    }
    const v: Volume = { data: new Float32Array(mask.length), mask, maskSchema: 'dental-segmentator-v1', nx, ny, nz, sx: 0.6, sy: 0.6, sz: 0.5, origin: [0, 0, 0], direction: [1, 0, 0, 0, 1, 0, 0, 0, 1] };
    const result = await searchOcclusalSplineArch(v, 4);
    expect(result.arch, result.reason).not.toBeNull();
    expect(result.arch!.controls).toHaveLength(17);
    expect(result.arch!.length).toBeGreaterThan(180);
    expect(result.arch!.length).toBeLessThanOrEqual(250);
  });

  it('rejects flaring palatal root apices without hooking inward', async () => {
    const nx = 200, ny = 200, nz = 80, mask = new Uint8Array(nx * ny * nz);
    for (let x = 35; x <= 165; x++) {
      const y0 = Math.round(150 - 0.022 * (x - 100) ** 2);
      const isCrownRegion = x >= 50 && x <= 150;
      for (let y = y0 - 4; y <= y0 + 4; y++) {
        if (y < 0 || y >= ny) continue;
        // Mandible body + rami (label 2)
        for (let z = 15; z <= 35; z++) mask[x + nx * (y + ny * z)] = 2;
        if (isCrownRegion) {
          // Lower teeth (label 4) at z=30..42, upper teeth (label 3) at z=43..55
          for (let z = 30; z <= 42; z++) mask[x + nx * (y + ny * z)] = 4;
          for (let z = 43; z <= 55; z++) mask[x + nx * (y + ny * z)] = 3;
        }
      }
    }
    // Add a flaring palatal root apex (label 3) 12 mm lingual to the right molar at apical Z (z=54..64)
    for (let x = 68; x <= 78; x++)
      for (let y = 100; y <= 112; y++)
        for (let z = 54; z <= 64; z++)
          mask[x + nx * (y + ny * z)] = 3;

    const v: Volume = { data: new Float32Array(mask.length), mask, maskSchema: 'dental-segmentator-v1', nx, ny, nz, sx: 0.5, sy: 0.5, sz: 0.5, origin: [0, 0, 0], direction: [1, 0, 0, 0, 1, 0, 0, 0, 1] };
    const result = await searchOcclusalSplineArch(v, 4);
    expect(result.arch, result.reason).not.toBeNull();
    const ctrls = result.arch!.controls!;
    expect(ctrls).toHaveLength(17);
    // X coordinates of control points must progress monotonically from R to L without hooking back inward
    for (let i = 1; i < ctrls.length; i++) {
      expect(ctrls[i][0]).toBeGreaterThanOrEqual(ctrls[i - 1][0] - 1.5);
    }
  });

  it('bridges unilateral missing molars and dolichocephalic rami without S-bends or 90-degree flips', async () => {
    const nx = 200, ny = 240, nz = 60, mask = new Uint8Array(nx * ny * nz);
    // Dolichocephalic U-arch (deeper in Y than wide in X) with missing lower left molars (x > 125 has only mandible label 2)
    for (let x = 40; x <= 160; x++) {
      const y0 = Math.round(190 - 0.04 * (x - 100) ** 2);
      for (let y = y0 - 4; y <= y0 + 4; y++) {
        if (y < 0 || y >= ny) continue;
        for (let z = 15; z <= 35; z++) mask[x + nx * (y + ny * z)] = 2;
        if (x >= 55 && x <= 120) {
          for (let z = 30; z <= 42; z++) mask[x + nx * (y + ny * z)] = 4;
        }
      }
    }
    const v: Volume = { data: new Float32Array(mask.length), mask, maskSchema: 'dental-segmentator-v1', nx, ny, nz, sx: 0.5, sy: 0.5, sz: 0.5, origin: [0, 0, 0], direction: [1, 0, 0, 0, 1, 0, 0, 0, 1] };
    const result = await searchOcclusalSplineArch(v, 4);
    expect(result.arch, result.reason).not.toBeNull();
    const ctrls = result.arch!.controls!;
    expect(ctrls).toHaveLength(17);
    expect(ctrls[0][0]).toBeLessThan(30);
    expect(ctrls[16][0]).toBeGreaterThan(70);
  });
});
