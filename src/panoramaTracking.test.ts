import { describe, expect, it } from 'vitest';
import { resampleCurve, type ArchResult } from './arch/arch';
import { renderPanorama } from './cpr';
import { isRegularPanoramaTrack, panoramaOffset, panoramaPoint, trackMandible } from './panoramaTracking';
import type { Volume } from './volume';

function tiltedJaw(reverse = false) {
  const nx = 100, ny = 80, nz = 100, data = new Float32Array(nx * ny * nz).fill(-1000), mask = new Uint8Array(data.length);
  for (let z = 0; z < nz; z++) {
    const lower = reverse ? z - 20 : 80 - z;
    if (lower < 0 || lower > 60) continue;
    const center = 40 + lower * 0.2;
    for (let y = 10; y < 70; y++) for (let x = 0; x < nx; x++) if (Math.abs(x - center) <= 4) {
      const i = x + nx * (y + ny * z); data[i] = 1000; mask[i] = 1;
    }
  }
  const volume: Volume = { data, mask, nx, ny, nz, sx: 1, sy: 1, sz: 1, origin: [0, 0, 0], direction: [1, 0, 0, 0, 1, 0, 0, 0, reverse ? -1 : 1] };
  return { volume, arch: resampleCurve([[40, 20], [40, 60]], 41) };
}

describe('mandible panorama tracking', () => {
  it('recovers a displaced lower body without moving the dental arch or upper rows', () => {
    const { volume, arch } = tiltedJaw();
    trackMandible(volume, arch, 80);
    const p = panoramaPoint(arch, 20, 20);
    expect(p[0]).toBeGreaterThan(51);
    expect(p[0]).toBeLessThanOrEqual(52);
    expect(arch.points[20]).toEqual([40, 40]);
    expect(panoramaOffset(arch, 20, 80)).toBe(0);
    expect(panoramaOffset(arch, 20, 90)).toBe(0);
    const original = renderPanorama(volume, arch, { thicknessMm: 6, mode: 'mean', trackMandible: false });
    const tracked = renderPanorama(volume, arch, { thicknessMm: 6, mode: 'mean', trackMandible: true });
    const pixel = (volume.nz - 1 - 20) * original.w + 20;
    expect(original.data[pixel]).toBe(-1000);
    expect(tracked.data[pixel]).toBeGreaterThan(900);
    expect(Array.from(tracked.data.slice(0, 20 * tracked.w))).toEqual(Array.from(original.data.slice(0, 20 * original.w)));
  });

  it('follows inferior anatomy with reversed slice direction', () => {
    const { volume, arch } = tiltedJaw(true);
    trackMandible(volume, arch, 20);
    expect(panoramaPoint(arch, 20, 80)[0]).toBeGreaterThan(51);
    expect(panoramaPoint(arch, 20, 80)[0]).toBeLessThanOrEqual(52);
    expect(panoramaOffset(arch, 20, 10)).toBe(0);
  });

  it('never invents tracking without valid mask evidence, including when reusing an arch object', () => {
    const { volume, arch } = tiltedJaw();
    trackMandible(volume, arch, 80);
    expect(arch.panorama).toBeDefined();
    volume.mask = undefined;
    trackMandible(volume, arch, 80);
    expect(arch.panorama).toBeUndefined();
    volume.mask = new Uint8Array(volume.data.length);
    trackMandible(volume, arch, 80);
    expect(arch.panorama).toBeUndefined();
  });

  it('rejects isolated single-voxel islands and clips no remote component into the view', () => {
    const { volume, arch } = tiltedJaw();
    volume.mask!.fill(0);
    for (let z = 15; z < 70; z++) volume.mask![52 + volume.nx * (40 + volume.ny * z)] = 1;
    trackMandible(volume, arch, 80);
    expect(arch.panorama).toBeUndefined();
  });

  it('rejects an excessive displacement instead of following the outer cortical edge', () => {
    const { volume, arch } = tiltedJaw();
    volume.mask!.fill(0);
    for (let z = 15; z < 70; z++)
      for (let y = 10; y < 70; y++)
        for (let x = 52; x <= 76; x++) volume.mask![x + volume.nx * (y + volume.ny * z)] = 1;
    trackMandible(volume, arch, 80);
    expect(arch.panorama).toBeUndefined();
  });
});

describe('panorama tracking surface regularity', () => {
  // Analytic quarter-circle: its inward normals meet at the centre after 6 mm.
  const circularArch = (): ArchResult => {
    const angles = Array.from({ length: 31 }, (_, i) => i * Math.PI / 60);
    return {
      points: angles.map(a => [6 * Math.cos(a), 6 * Math.sin(a)]),
      normals: angles.map(a => [-Math.cos(a), -Math.sin(a)]),
      tangents: angles.map(a => [-Math.sin(a), Math.cos(a)]),
      length: 3 * Math.PI,
    };
  };

  it('rejects an inward fold even though every offset is below the 12 mm displacement limit', () => {
    const arch = circularArch();
    expect(isRegularPanoramaTrack(arch, new Float32Array(31).fill(8), 1)).toBe(false);
    expect(isRegularPanoramaTrack(arch, new Float32Array(31).fill(6), 1)).toBe(false);
    expect(isRegularPanoramaTrack(arch, new Float32Array(31).fill(2), 1)).toBe(true);
  });

  it('rejects abrupt local shear while accepting a smooth supported slope', () => {
    const arch = resampleCurve([[0, 0], [20, 0]], 21);
    const smooth = Float32Array.from({ length: 42 }, (_, i) => (i % 21) * 0.2);
    expect(isRegularPanoramaTrack(arch, smooth, 2)).toBe(true);
    const step = Float32Array.from({ length: 42 }, (_, i) => i >= 31 ? 4 : 0);
    expect(isRegularPanoramaTrack(arch, step, 2)).toBe(false);
  });

  it('rejects invalid offset fields instead of passing nonfinite coordinates to the renderer', () => {
    const arch = circularArch(), offsets = new Float32Array(31).fill(2);
    offsets[12] = NaN;
    expect(isRegularPanoramaTrack(arch, offsets, 1)).toBe(false);
    expect(isRegularPanoramaTrack(arch, new Float32Array(30), 1)).toBe(false);
  });
});
