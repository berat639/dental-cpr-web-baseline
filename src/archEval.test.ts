// @ts-nocheck -- opt-in developer harness (Node APIs; no @types/node in this project)
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'vitest';
import { searchArch } from './archDetect';
import { findOcclusalSlice } from './cpr';
import type { Volume } from './volume';

/**
 * Real-scan evaluation (opt-in): ARCH_EVAL_DIR=/tmp/arch_eval npx vitest run src/archEval.test.ts
 * Volumes come from scripts/export_eval_volumes.py. Writes <dir>/report.json with the chosen arch per scan so
 * scripts/render_arch_eval.py can overlay it for review.
 */
const dir = process.env.ARCH_EVAL_DIR;
describe.skipIf(!dir || !existsSync(dir))('arch evaluation on real scans', () => {
  const metas = dir ? readdirSync(dir).filter((f) => f.endsWith('.json') && f !== 'report.json').sort() : [];
  const report: Record<string, unknown>[] = [];
  for (const file of metas.filter((f) => !process.env.ARCH_EVAL_ONLY || f.includes(process.env.ARCH_EVAL_ONLY))) {
    it(file, async () => {
      const meta = JSON.parse(readFileSync(join(dir!, file), 'utf8'));
      const [nx, ny, nz] = meta.n as number[], [sx, sy, sz] = meta.s as number[];
      const raw = readFileSync(join(dir!, `${meta.name}.raw`));
      const i16 = new Int16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
      const data = new Float32Array(i16.length);
      for (let i = 0; i < i16.length; i++) data[i] = i16[i];
      const v: Volume = { data, nx, ny, nz, sx, sy, sz, origin: [0, 0, 0], direction: [1, 0, 0, 0, 1, 0, 0, 0, 1] };
      const maskPath = join(dir!, `${meta.name}.mask`);
      const withMask = existsSync(maskPath) && !process.env.ARCH_EVAL_NOMASK;
      if (withMask) v.mask = new Uint8Array(readFileSync(maskPath));
      const t0 = performance.now();
      const slice = findOcclusalSlice(v, 20);
      const z = slice >= 0 ? slice : Math.round(nz * 0.55);
      const r = await searchArch(v, z, 6, withMask);
      const ms = Math.round(performance.now() - t0);
      const pick = r.chosen ?? r.best;
      report.push({
        name: meta.name, withMask, n: meta.n, s: meta.s, occlusalSlice: slice, ms, accepted: !!r.chosen, anyEvidence: r.anyEvidence,
        fit: pick?.fit, zc: pick?.zc, support: pick?.ev.supportFraction, gapMm: pick?.ev.longestUnsupportedMm,
        lengthMm: pick?.arch.length, plaus: pick?.plaus, points: pick ? pick.arch.points.filter((_, i) => i % 4 === 0) : [],
      });
      writeFileSync(join(dir!, 'report.json'), JSON.stringify(report));
    }, 600_000);
  }
});
