/** Shared 2D overlay drawing: mm rulers, orientation letters and corner read-outs. */

export interface ViewGeom {
  /** Image rectangle inside its stage, CSS px. */
  left: number; top: number; width: number; height: number;
  /** Physical size shown by that rectangle. */
  widthMm: number; heightMm: number;
}

export interface Pad { left: number; right: number; top: number; bottom: number }
export const RULER_PAD: Pad = { left: 46, right: 10, top: 10, bottom: 24 };

/** Place a canvas inside `stage` at the largest size keeping physical aspect, leaving room for the rulers. */
export function fitInto(canvas: HTMLCanvasElement, stage: HTMLElement, widthMm: number, heightMm: number, pad: Pad = RULER_PAD): ViewGeom {
  const bounds = stage.getBoundingClientRect();
  const availW = Math.max(10, bounds.width - pad.left - pad.right), availH = Math.max(10, bounds.height - pad.top - pad.bottom);
  const scale = Math.min(availW / widthMm, availH / heightMm);
  const width = widthMm * scale, height = heightMm * scale;
  const left = pad.left + (availW - width) / 2, top = pad.top + (availH - height) / 2;
  Object.assign(canvas.style, { position: 'absolute', inset: 'auto', margin: '0', left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` });
  return { left, top, width, height, widthMm, heightMm };
}

/** Size an overlay canvas to its stage (devicePixelRatio-aware) and return a CSS-pixel context. */
export function prepOverlay(canvas: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; width: number; height: number } {
  const stage = canvas.parentElement!, bounds = stage.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(bounds.width)), height = Math.max(1, Math.round(bounds.height));
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  }
  Object.assign(canvas.style, { position: 'absolute', inset: '0', width: `${width}px`, height: `${height}px`, margin: '0' });
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

/**
 * Tick positions in mm. Major ticks split the extent into `divisions` equal parts
 * (labelled 0.0, 25.2, 50.3, 75.5, 100.6 for a 100.6 mm extent, as in the clinical reference);
 * `minorPerMajor` minor ticks subdivide every major interval.
 */
export function rulerTicks(extentMm: number, divisions = 4, minorPerMajor = 5): { major: number[]; minor: number[] } {
  const major: number[] = [], minor: number[] = [];
  const step = extentMm / divisions;
  for (let i = 0; i <= divisions; i++) major.push(i * step);
  for (let i = 0; i < divisions * minorPerMajor; i++) if (i % minorPerMajor) minor.push((i * step) / minorPerMajor);
  return { major, minor };
}

export const formatMm = (value: number): string => value.toFixed(1);

export interface RulerStyle { color: string; divisions?: number; hDivisions?: number; minorPerMajor?: number; font?: string }

/**
 * Left ruler: 0 at the bottom, extent at the top. Bottom ruler: 0 at the left, extent at the right.
 * Both sit outside the image rectangle.
 */
export function drawRulers(ctx: CanvasRenderingContext2D, g: ViewGeom, style: RulerStyle): void {
  const { color, divisions = 4, hDivisions = divisions, minorPerMajor = 5, font = '10px Inter, sans-serif' } = style;
  ctx.save();
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1; ctx.font = font;
  const bottom = g.top + g.height, right = g.left + g.width;

  const v = rulerTicks(g.heightMm, divisions, minorPerMajor);
  ctx.beginPath(); ctx.moveTo(g.left - 4.5, g.top); ctx.lineTo(g.left - 4.5, bottom); ctx.stroke();
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const mm of v.minor) { const y = bottom - (mm / g.heightMm) * g.height; ctx.beginPath(); ctx.moveTo(g.left - 7.5, y); ctx.lineTo(g.left - 4.5, y); ctx.stroke(); }
  for (const mm of v.major) {
    const y = bottom - (mm / g.heightMm) * g.height;
    ctx.beginPath(); ctx.moveTo(g.left - 11.5, y); ctx.lineTo(g.left - 4.5, y); ctx.stroke();
    // keep the end labels inside the stage instead of half outside it
    ctx.fillText(formatMm(mm), g.left - 14, Math.min(Math.max(y, g.top + 5), bottom - 5));
  }

  const h = rulerTicks(g.widthMm, hDivisions, minorPerMajor);
  ctx.beginPath(); ctx.moveTo(g.left, bottom + 4.5); ctx.lineTo(right, bottom + 4.5); ctx.stroke();
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const mm of h.minor) { const x = g.left + (mm / g.widthMm) * g.width; ctx.beginPath(); ctx.moveTo(x, bottom + 4.5); ctx.lineTo(x, bottom + 7.5); ctx.stroke(); }
  h.major.forEach((mm, i) => {
    const x = g.left + (mm / g.widthMm) * g.width;
    ctx.beginPath(); ctx.moveTo(x, bottom + 4.5); ctx.lineTo(x, bottom + 11.5); ctx.stroke();
    ctx.textAlign = i === 0 ? 'left' : i === h.major.length - 1 ? 'right' : 'center';
    ctx.fillText(formatMm(mm), x, bottom + 12);
  });
  ctx.restore();
}

/** Letters at the middle of the left and right image edges (e.g. R … L or B … L). */
export function drawOrientation(ctx: CanvasRenderingContext2D, g: ViewGeom, left: string, right: string, color: string): void {
  ctx.save();
  ctx.fillStyle = color; ctx.font = '600 13px Inter, sans-serif'; ctx.textBaseline = 'middle';
  const y = g.top + g.height / 2;
  ctx.textAlign = 'left'; ctx.fillText(left, g.left + 6, y);
  ctx.textAlign = 'right'; ctx.fillText(right, g.left + g.width - 6, y);
  ctx.restore();
}

export function drawCornerText(ctx: CanvasRenderingContext2D, g: ViewGeom, text: string, color: string, corner: 'tr' | 'tl' | 'top-center', size = 14, alpha = 1): void {
  if (!text) return;
  ctx.save();
  ctx.globalAlpha = alpha; ctx.fillStyle = color; ctx.font = `600 ${size}px Inter, sans-serif`; ctx.textBaseline = 'top';
  if (corner === 'tr') { ctx.textAlign = 'right'; ctx.fillText(text, g.left + g.width - 8, g.top + 6); }
  else if (corner === 'tl') { ctx.textAlign = 'left'; ctx.fillText(text, g.left + 8, g.top + 6); }
  else { ctx.textAlign = 'center'; ctx.fillText(text, g.left + g.width / 2, Math.max(1, g.top - size - 3)); }
  ctx.restore();
}
