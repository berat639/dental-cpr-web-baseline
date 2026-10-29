import type { Pt } from './arch/arch';

export type Vec3 = [number, number, number];

export interface Volume {
  data: Float32Array; // original DICOM stored values, x fastest
  nx: number; ny: number; nz: number;
  sx: number; sy: number; sz: number; // mm
  /** DICOM physical geometry, preserved by the local SimpleITK reader. */
  origin: Vec3;
  direction: [number, number, number, number, number, number, number, number, number];
  datasetId?: string;
  /** Optional AI jaw mask (same grid, x-fastest). */
  mask?: Uint8Array;
  maskSchema?: 'mandible-v1' | 'dental-segmentator-v1';
}

export const idx = (v: Volume, x: number, y: number, z: number) => x + v.nx * (y + v.ny * z);

/** Allocate only one browser-side copy of a volume. */
export function allocF32(n: number): Float32Array {
  return new Float32Array(n);
}

const identityDirection = (): Volume['direction'] => [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** Convert raw int16 data and SimpleITK geometry returned by the local API. */
export function volumeFromResponse(response: Response, buffer: ArrayBuffer, datasetId?: string): Volume {
  const n = (name: string) => Number(response.headers.get(name));
  const list = (name: string, count: number): number[] =>
    (response.headers.get(name) ?? '').split(',').filter(Boolean).map(Number).slice(0, count);
  const origin = list('X-Origin', 3);
  const direction = list('X-Direction', 9);
  const raw = new Int16Array(buffer);
  const data = allocF32(raw.length);
  for (let i = 0; i < raw.length; i++) data[i] = raw[i];
  return {
    data, nx: n('X-Nx'), ny: n('X-Ny'), nz: n('X-Nz'),
    sx: n('X-Sx'), sy: n('X-Sy'), sz: n('X-Sz'),
    origin: (origin.length === 3 ? origin : [0, 0, 0]) as Vec3,
    direction: (direction.length === 9 ? direction : identityDirection()) as Volume['direction'],
    datasetId,
  };
}

/** Trilinear sample at voxel coordinates. Returns -1000 outside. */
export function sample(v: Volume, x: number, y: number, z: number): number {
  if (x < 0 || y < 0 || z < 0 || x > v.nx - 1 || y > v.ny - 1 || z > v.nz - 1) return -1000;
  const x0 = Math.floor(x), y0 = Math.floor(y), z0 = Math.floor(z);
  const x1 = Math.min(x0 + 1, v.nx - 1), y1 = Math.min(y0 + 1, v.ny - 1), z1 = Math.min(z0 + 1, v.nz - 1);
  const fx = x - x0, fy = y - y0, fz = z - z0;
  const d = v.data;
  const c = (a: number, b: number, e: number) => d[a + v.nx * (b + v.ny * e)];
  const c00 = c(x0, y0, z0) * (1 - fx) + c(x1, y0, z0) * fx;
  const c10 = c(x0, y1, z0) * (1 - fx) + c(x1, y1, z0) * fx;
  const c01 = c(x0, y0, z1) * (1 - fx) + c(x1, y0, z1) * fx;
  const c11 = c(x0, y1, z1) * (1 - fx) + c(x1, y1, z1) * fx;
  return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
}

/** Synthetic jaw phantom (with a toothless gap) for demo/testing. */
export function makePhantom(): Volume {
  const nx = 180, ny = 180, nz = 90;
  const sx = 0.6, sy = 0.6, sz = 0.8;
  const data = allocF32(nx * ny * nz).fill(-1000);
  const curve: Pt[] = [];
  for (let x = -40; x <= 40; x += 0.5) curve.push([x, 0.025 * x * x - 12]);
  // soft tissue head ellipse
  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++) {
        const mx = (x - nx / 2) * sx, my = (y - ny / 2) * sy;
        const e = (mx / 48) ** 2 + (my / 46) ** 2;
        let h = e < 1 ? 40 : -1000;
        if (e < 1 && z > 20 && z < 70) {
          let dmin = 1e9;
          for (const [cx, cy] of curve) dmin = Math.min(dmin, Math.hypot(mx - cx, my - cy));
          const zc = (z - 45) * sz;
          const gap = Math.abs(mx) < 7 && my < -8; // edentulous gap: no teeth crowns
          const isTooth = z > 45 && !gap && dmin < 3.5;
          const bone = dmin < 6 && Math.abs(zc) < 12;
          if (isTooth) h = 1800;
          else if (bone) h = 900 + 300 * Math.cos(dmin);
        }
        data[x + nx * (y + ny * z)] = h + (Math.random() - 0.5) * 30;
      }
  return { data, nx, ny, nz, sx, sy, sz, origin: [0, 0, 0], direction: identityDirection() };
}
