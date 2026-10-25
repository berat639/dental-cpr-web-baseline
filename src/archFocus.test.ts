import { describe, expect, it } from 'vitest';
import { splineArch, type Pt } from './arch/arch';
import { refineDentalFocus } from './archFocus';
import { renderPanorama } from './cpr';
import type { Volume } from './volume';

/** A known dense tube in original voxel data, independent of any AI mask. */
function tubeFixture(offsetMm = 5, singleSlice = false) {
  const nx = 110, ny = 80, nz = 41;
  const data = new Float32Array(nx * ny * nz).fill(40);
  const anchorZ = 20, tubeY = 30 + offsetMm;
  for (let z = 0; z < nz; z++) {
    if (singleSlice ? z !== anchorZ : z < 10 || z > 30) continue;
    for (let y = 0; y < ny; y++) {
      if (Math.abs(y - tubeY) > 1) continue;
      for (let x = 8; x <= 92; x++) data[x + nx * (y + ny * z)] = singleSlice ? 12000 : 1800;
    }
  }
  const volume: Volume = {
    data, nx, ny, nz, sx: 1, sy: 1, sz: 1,
    origin: [0, 0, 0], direction: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  };
  const controls: Pt[] = Array.from({ length: 9 }, (_, i) => [10 + i * 10, 30]);
  const arch = splineArch(controls, 81);
  arch.controls = controls;
  return { volume, arch, anchorZ, tubeY };
}

describe('dental focus refinement', () => {
  it('aligns an off-centre dense tube and recovers its signal in a fixed 6 mm slab', () => {
    const { volume, arch, anchorZ, tubeY } = tubeFixture();
    const originalPoints = arch.points.map(p => [...p]);
    const result = refineDentalFocus(volume, arch, anchorZ);
    expect(result.applied).toBe(true);
    const middle = Math.floor(result.arch.points.length / 2);
    expect(Math.abs(result.arch.points[middle][1] - tubeY)).toBeLessThan(0.75);
    expect(result.afterCoverage).toBeGreaterThan(result.beforeCoverage + 0.4);
    const slab = { thicknessMm: 6, mode: 'mean' as const };
    const before = renderPanorama(volume, arch, slab), after = renderPanorama(volume, result.arch, slab);
    const pixel = (volume.nz - 1 - anchorZ) * before.w + middle;
    expect(before.data[pixel]).toBe(40);
    expect(after.data[pixel]).toBeGreaterThan(600);
    expect(arch.points).toEqual(originalPoints);
  });

  it('keeps the original arch when the source has no dense evidence', () => {
    const { volume, arch, anchorZ } = tubeFixture();
    volume.data.fill(40);
    const result = refineDentalFocus(volume, arch, anchorZ);
    expect(result.applied).toBe(false);
    expect(result.arch).toBe(arch);
  });

  it('does not follow a bright streak present on only one source slice', () => {
    const { volume, arch, anchorZ } = tubeFixture(5, true);
    const result = refineDentalFocus(volume, arch, anchorZ);
    expect(result.applied).toBe(false);
    expect(result.arch).toBe(arch);
  });

  it('keeps an already-centred tube and its existing controls unchanged', () => {
    const { volume, arch, anchorZ } = tubeFixture(0);
    const originalControls = arch.controls!.map(p => [...p]);
    const result = refineDentalFocus(volume, arch, anchorZ);
    expect(result.applied).toBe(false);
    expect(result.arch).toBe(arch);
    expect(arch.controls).toEqual(originalControls);
  });

  it('preserves both endpoints and exposes controls that reproduce the refined curve', () => {
    const { volume, arch, anchorZ } = tubeFixture();
    const result = refineDentalFocus(volume, arch, anchorZ);
    expect(result.applied).toBe(true);
    expect(result.arch.points[0]).toEqual(arch.points[0]);
    expect(result.arch.points.at(-1)).toEqual(arch.points.at(-1));
    expect(result.arch.controls).toHaveLength(arch.controls!.length);
    const rebuilt = splineArch(result.arch.controls!, result.arch.points.length);
    expect(rebuilt.points).toEqual(result.arch.points);
    expect(rebuilt.normals).toEqual(result.arch.normals);
    expect(rebuilt.tangents).toEqual(result.arch.tangents);
    expect(rebuilt.length).toBe(result.arch.length);
    const slab = { thicknessMm: 6, mode: 'mean' as const };
    expect(renderPanorama(volume, rebuilt, slab).data).toEqual(renderPanorama(volume, result.arch, slab).data);
  });
});
