import { describe, expect, it } from 'vitest';
import { drawRulers, formatMm, rulerTicks, type ViewGeom } from './overlay';

describe('rulerTicks', () => {
  it('splits the extent into equal major intervals starting at 0', () => {
    const { major } = rulerTicks(100.6, 4);
    expect(major).toHaveLength(5);
    [0, 25.15, 50.3, 75.45, 100.6].forEach((mm, i) => expect(major[i]).toBeCloseTo(mm, 6));
  });

  it('adds minorPerMajor - 1 minor ticks inside each interval and never duplicates a major tick', () => {
    const { major, minor } = rulerTicks(100, 4, 5);
    expect(minor).toHaveLength(4 * 4);
    for (const m of minor) expect(major.some((M) => Math.abs(M - m) < 1e-9)).toBe(false);
    expect(minor[0]).toBeCloseTo(5);
  });
});

describe('formatMm', () => {
  it('uses one decimal like the clinical reference', () => {
    expect(formatMm(0)).toBe('0.0');
    expect(formatMm(25.15)).toMatch(/^25\.[12]$/);
    expect(formatMm(100.6)).toBe('100.6');
  });
});

describe('drawRulers', () => {
  it('labels 0 at the bottom of the left ruler and the extent at the top', () => {
    const calls: { text: string; x: number; y: number }[] = [];
    const ctx = new Proxy({ fillText: (text: string, x: number, y: number) => calls.push({ text, x, y }) } as Record<string, unknown>, {
      get: (target, key: string) => (key in target ? target[key] : () => undefined),
      set: (target, key: string, value) => { target[key] = value; return true; },
    }) as unknown as CanvasRenderingContext2D;
    const g: ViewGeom = { left: 50, top: 10, width: 200, height: 100, widthMm: 190.2, heightMm: 100.6 };
    drawRulers(ctx, g, { color: '#f00' });
    const left = calls.filter((c) => c.x < g.left);
    const zero = left.find((c) => c.text === '0.0')!, top = left.find((c) => c.text === '100.6')!;
    expect(zero.y).toBeGreaterThan(top.y);
    expect(calls.some((c) => c.text === '190.2' && c.y > g.top + g.height)).toBe(true);
  });
});
