import './style.css';
import { splineArch, type ArchResult, type Pt } from './arch/arch';
import { searchArch } from './archDetect';
import { panoramaOffset, panoramaPoint, trackMandible } from './panoramaTracking';
import {
  archFrameAt, findOcclusalSlice, renderCrossAt, renderPanorama, sectionNumber, toCanvas,
} from './cpr';
import { GpuPanoramaRenderer } from './cprGpu';
import { LoadingOverlay, type StageId } from './loading';
import { drawCornerText, drawOrientation, drawRulers, fitInto, prepOverlay, RULER_PAD, type Pad, type ViewGeom } from './overlay';
import { API, ServiceLink, type JobInfo, type ServerEventDetail } from './service';
import { makePhantom, sample, volumeFromResponse, type Volume } from './volume';
import type { Viewer3D } from './viewer3d';

type Jaw = 'lower' | 'upper';
type Source = { kind: 'path'; path: string } | { kind: 'files'; files: File[] };
type Img = { data: Float32Array; w: number; h: number };
class Cancelled extends Error {}
class Restarted extends Error {}

const COLOR = { ax: '#ff5a3c', pano: '#8d96ff', cross: '#35e0a1', level: '#ff9a3c' };
const CROSS_PAD: Pad = { left: 36, right: 6, top: 20, bottom: 22 };
const WL_PRESETS: Record<string, { wc: number; ww: number }> = { bone: { wc: 500, ww: 2000 }, tooth: { wc: 1200, ww: 2500 }, soft: { wc: 40, ww: 400 } };

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const num = (id: string) => Number($<HTMLInputElement>(id).value);
const cvAxial = $<HTMLCanvasElement>('cv-axial'), ovAxial = $<HTMLCanvasElement>('ov-axial'), stageAxial = $<HTMLDivElement>('stage-axial');
const cvPano = $<HTMLCanvasElement>('cv-pano'), cvPanoCpu = $<HTMLCanvasElement>('cv-pano-cpu'), ovPano = $<HTMLCanvasElement>('ov-pano'), panoStage = $<HTMLDivElement>('pano-stage');
const crossStrip = $<HTMLDivElement>('cross-strip');
const loading = new LoadingOverlay($('loading'));
const link = new ServiceLink();

// ---- case state -----------------------------------------------------------------------------
let volume: Volume | null = null;
let activeDatasetId = '';
let currentArch: ArchResult | null = null;
let controlPoints: Pt[] = [];
let selectedJaw: Jaw = 'lower';
const arches = new Map<Jaw, { arch: ArchResult; controls: Pt[] }>();
let centerS = 0; // arc-length (mm) of the middle of the 5 cross-sections
let viewer3d: Viewer3D | null = null;
let lastSource: Source | null = null;
let loadToken = 0;
let activeJob = '';
let skipRequested = false;
let currentXhr: XMLHttpRequest | null = null;
let dragIndex = -1;

// ---- view caches ----------------------------------------------------------------------------
let axialGeom: ViewGeom | null = null;
let panoGeom: ViewGeom | null = null;
const crossGeoms: ViewGeom[] = [];
const crossImgs: (Img | null)[] = [];
const crossCells: { cell: HTMLDivElement; cv: HTMLCanvasElement; ov: HTMLCanvasElement }[] = [];
let hover: { panel: 'axial' | 'pano' | number; hu: number } | null = null;
let lastAxialHu: number | null = null;
const renderer = new GpuPanoramaRenderer(cvPano);

const jaw = (): Jaw => $<HTMLSelectElement>('in-jaw').value as Jaw;
const windowLevel = () => ({ center: num('in-wc'), width: num('in-ww') });
const slab = () => ({ thicknessMm: num('in-thick'), mode: $<HTMLInputElement>('in-max').checked ? 'max' as const : 'mean' as const, trackMandible: $<HTMLInputElement>('in-tracking').checked });
const half = () => num('in-width');

// ---- messages -------------------------------------------------------------------------------
let toastTimer = 0;
function toast(text: string, kind: 'info' | 'warn' | 'error' | 'ok' = 'info', ms = 7000) {
  const el = $('toast');
  el.textContent = text; el.className = `toast ${kind}`; el.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { el.hidden = true; }, ms);
}
function setCaseInfo(text: string) { const el = $('case-info'); el.textContent = text; el.title = text; }

// ---- service light + popover ----------------------------------------------------------------
const CONN_LABEL = { offline: 'Servis kapalı', connecting: 'Yeniden bağlanıyor…', online: 'Çalışıyor', stopping: 'Durduruluyor…', incompatible: 'Sürüm uyumsuz' } as const;
const CONN_HINT = {
  offline: 'Servis kapalı. PineClinic Lokal Servisler uygulamasını açıp Start düğmesine basın.',
  connecting: 'Servise yeniden bağlanılıyor…',
  online: '',
  stopping: 'Servis durduruluyor…',
  incompatible: 'Servis sürümü bu görüntüleyiciyle uyumsuz. Lokal Servisler uygulamasını güncelleyin.',
} as const;
function updateServiceUi() {
  const state = link.state;
  $('service-dot').className = `service-dot ${state}`;
  $('sp-state').textContent = CONN_LABEL[state];
  $('sp-address').textContent = API.replace('http://', '');
  $('sp-engine').textContent = state === 'online' ? (link.provider && link.provider !== 'none' ? link.provider : 'AI motoru yok') : '—';
  $('sp-started').textContent = state === 'online' && link.startedAt ? new Date(link.startedAt * 1000).toLocaleTimeString() : '—';
  $('sp-manager').textContent = state === 'online' ? `${link.viewers} görüntüleyici bağlı` : '—';
  const usable = state === 'online';
  const btnLoad = $<HTMLButtonElement>('btn-load');
  btnLoad.hidden = !usable && !isDemo;
  btnLoad.disabled = !usable && !isDemo;
  $('btn-load-empty').hidden = !usable && !isDemo;
  $('empty-hint').textContent = '';
  if (usable) { localStorage.setItem(INSTALLED_KEY, '1'); launching = false; launchFailed = false; }

  const emptyTitle = $('empty-title');
  const emptyDesc = $('empty-desc');
  if (usable || isDemo) {
    emptyTitle.textContent = 'Başlamak için bir DICOM serisi seçin';
    emptyDesc.textContent = 'Seri klasörünü seçin ya da buraya sürükleyin. Panoramik, kesitler ve 3D görünüm AI segmentasyonu tamamlanınca birlikte açılır.';
  } else {
    const installed = isInstalled() && !launchFailed;
    if (installed) {
      emptyTitle.textContent = 'Lokal Servis Kapalı';
      emptyDesc.textContent = 'DICOM verilerini işlemek ve AI analizi yapabilmek için bilgisayarınızdaki servisi başlatın.';
    } else {
      emptyTitle.textContent = 'PineClinic Lokal Servisler Gerekli';
      emptyDesc.textContent = 'Görüntüleme ve AI modelleri yerel bilgisayarınızda çalışır. Başlamak için uygulamayı indirin ve kurun.';
    }
  }

  renderServiceCta(state);
}
const isDemo = new URLSearchParams(location.search).has('demo');

// ---- install / launch flow --------------------------------------------------------------------
const INSTALLED_KEY = 'pineclinic.service.installed';
const DOWNLOAD_URL = (import.meta.env.VITE_DOWNLOAD_URL as string | undefined) ?? '/downloads/PineClinic-Lokal-Servisler.dmg';
const LAUNCH_URL = 'pineclinic://start';
let launching = false;
let launchFailed = false;
let launchTimer = 0;
const isInstalled = () => {
  const val = localStorage.getItem(INSTALLED_KEY);
  if (val !== null) return val === '1';
  return true; // Default to installed on local machine so existing users see "Servisi Aç"
};

function cancelLaunch() {
  launching = false;
  window.clearTimeout(launchTimer);
  updateServiceUi();
}

function launchLocalService() {
  launching = true;
  launchFailed = false;
  updateServiceUi();
  // Custom-protocol hand-off to the desktop app; an iframe avoids navigating away when no handler exists.
  const frame = document.createElement('iframe');
  frame.style.display = 'none'; frame.src = LAUNCH_URL;
  document.body.appendChild(frame);
  window.setTimeout(() => frame.remove(), 3000);
  window.clearTimeout(launchTimer);
  launchTimer = window.setTimeout(() => {
    if (link.state === 'online') return;
    launching = false;
    toast('Servis henüz bağlanmadı. Tarayıcı uyarısından "Aç"ı onayladığınızdan emin olun veya tekrar deneyin.', 'warn', 7000);
    updateServiceUi();
  }, 10000);
}
const macGuideModal = $('modal-mac-guide');
const btnMacGuideClose = $('btn-mac-guide-close');

function openMacGuide() {
  macGuideModal.hidden = false;
  btnMacGuideClose.focus();
}

function closeMacGuide() {
  macGuideModal.hidden = true;
}

btnMacGuideClose.onclick = () => closeMacGuide();
macGuideModal.addEventListener('click', (event) => {
  if (event.target === macGuideModal) closeMacGuide();
});
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !macGuideModal.hidden) closeMacGuide();
});

function downloadLocalService() {
  window.open(DOWNLOAD_URL, '_blank', 'noopener');
  openMacGuide();
  toast('İndirme başladı. Kurulum ve ilk açılış için ekrandaki rehberi inceleyin.', 'info', 9000);
}

/** Renders every [data-svc-cta] block: install -> open service -> (online) load DICOM. */
function renderServiceCta(state: typeof link.state) {
  const show = state !== 'online' && !isDemo;
  const installed = isInstalled() && !launchFailed;
  document.querySelectorAll<HTMLElement>('[data-svc-cta]').forEach((box) => {
    box.hidden = !show;
    if (!show) return;
    const primary = box.querySelector<HTMLButtonElement>('[data-svc=primary]')!;
    const alt = box.querySelector<HTMLButtonElement>('[data-svc=alt]')!;
    const note = box.querySelector<HTMLElement>('[data-svc=note]')!;
    const macGuide = box.querySelector<HTMLButtonElement>('[data-svc=macguide]');
    if (macGuide) {
      macGuide.onclick = () => openMacGuide();
      macGuide.hidden = false;
    }
    if (state === 'incompatible') {
      primary.textContent = 'Servisi Güncelle'; primary.onclick = downloadLocalService; primary.disabled = false;
      alt.hidden = true; note.textContent = 'Kurulu servis sürümü bu görüntüleyiciyle uyumsuz.';
    } else if (installed) {
      if (launching) {
        primary.textContent = 'Tekrar Gönder';
        primary.disabled = false;
        primary.onclick = () => { launchLocalService(); };
        alt.hidden = false;
        alt.textContent = 'Vazgeç';
        alt.onclick = () => { cancelLaunch(); };
        note.textContent = 'Tarayıcı bildiriminde "PineClinic Lokal Servisler adlı uygulamayı aç" seçeneğine tıklayın.';
      } else {
        primary.textContent = 'Servisi Aç';
        primary.disabled = state === 'stopping';
        primary.onclick = launchLocalService;
        alt.hidden = false;
        alt.textContent = 'Kurulu değil mi? İndir ve Kur';
        alt.onclick = () => { localStorage.setItem(INSTALLED_KEY, '0'); cancelLaunch(); downloadLocalService(); };
        note.textContent = 'Servis kapalı. Buton masaüstü uygulamasını otomatik başlatır.';
      }
    } else {
      primary.textContent = 'Servisi İndir ve Kur'; primary.onclick = downloadLocalService; primary.disabled = false;
      alt.hidden = false; alt.textContent = 'Zaten kurulu mu? Servisi Aç';
      alt.onclick = () => { localStorage.setItem(INSTALLED_KEY, '1'); launchLocalService(); };
      alt.disabled = false;
      note.textContent = 'PineClinic Lokal Servisler bulunamadı. Önce indirip kurun.';
    }
  });
}

// ---- popovers -------------------------------------------------------------------------------
let openPop: HTMLElement | null = null;
let openBtn: HTMLElement | null = null;
function closePop() { if (openPop) openPop.hidden = true; openBtn?.classList.remove('open'); openPop = openBtn = null; }
function togglePop(pop: HTMLElement, anchor: HTMLElement, side: 'right' | 'below') {
  const wasOpen = openPop === pop;
  closePop();
  if (wasOpen) return;
  pop.hidden = false;
  const r = anchor.getBoundingClientRect();
  if (side === 'right') { pop.style.left = `${r.right + 8}px`; pop.style.top = `${Math.max(8, Math.min(r.top, innerHeight - pop.offsetHeight - 8))}px`; }
  anchor.classList.add('open');
  openPop = pop; openBtn = anchor;
}
document.querySelectorAll<HTMLButtonElement>('[data-pop]').forEach((button) => {
  button.addEventListener('click', (event) => { event.stopPropagation(); togglePop($(button.dataset.pop!), button, 'right'); });
});
$('btn-service').addEventListener('click', (event) => {
  event.stopPropagation();
  const pop = $('service-popover'); const wasOpen = !pop.hidden;
  closePop(); pop.hidden = wasOpen;
});
document.addEventListener('click', (event) => {
  const target = event.target as Node;
  if (openPop && !openPop.contains(target)) closePop();
  const sp = $('service-popover');
  if (!sp.hidden && !sp.contains(target)) sp.hidden = true;
});

// ---- geometry helpers -----------------------------------------------------------------------
const centroid = (arch: ArchResult): Pt => {
  let x = 0, y = 0; for (const p of arch.points) { x += p[0]; y += p[1]; }
  return [x / arch.points.length, y / arch.points.length];
};
/** Which side of a cross-section is buccal: the side the normal points away from the arch centre. */
function sideLabels(arch: ArchResult, s: number): [string, string] {
  const f = archFrameAt(arch, s), c = centroid(arch);
  const outward = (f.p[0] - c[0]) * f.n[0] + (f.p[1] - c[1]) * f.n[1] > 0;
  return outward ? ['L', 'B'] : ['B', 'L'];
}
function sections() {
  if (!currentArch) return [];
  const spacing = num('in-spacing');
  return Array.from({ length: 5 }, (_, k) => {
    const s = Math.max(0, Math.min(currentArch!.length, centerS + (k - 2) * spacing));
    return { s, number: sectionNumber(s, spacing) };
  });
}
const clampCenter = () => { if (currentArch) centerS = Math.max(0, Math.min(currentArch.length, centerS)); };
const centerIndex = () => currentArch ? Math.round((centerS / currentArch.length) * (currentArch.points.length - 1)) : 0;

// ---- axial ----------------------------------------------------------------------------------
function drawAxial() {
  if (!volume) return;
  const z = num('in-slice');
  toCanvas(cvAxial, { data: volume.data.subarray(z * volume.nx * volume.ny, (z + 1) * volume.nx * volume.ny), w: volume.nx, h: volume.ny }, windowLevel());
  axialGeom = fitInto(cvAxial, stageAxial, volume.nx * volume.sx, volume.ny * volume.sy);
  drawAxialOverlay();
}

function drawAxialOverlay() {
  const { ctx } = prepOverlay(ovAxial);
  if (!volume || !axialGeom) return;
  const g = axialGeom, X = (mm: number) => g.left + (mm / g.widthMm) * g.width, Y = (mm: number) => g.top + (mm / g.heightMm) * g.height;
  drawRulers(ctx, g, { color: COLOR.ax });
  const flipped = volume.direction[0] < 0; // LPS: image left is the patient's right unless the series is mirrored
  drawOrientation(ctx, g, flipped ? 'L' : 'R', flipped ? 'R' : 'L', '#ff7a5f');
  drawCornerText(ctx, g, `${num('in-slice') + 1}/${volume.nz}`, 'rgba(255,122,95,.85)', 'tl', 12);
  if (hover?.panel === 'axial') drawCornerText(ctx, g, `${Math.round(hover.hu)} HU`, COLOR.ax, 'tr', 15);
  else if (lastAxialHu !== null) drawCornerText(ctx, g, `${Math.round(lastAxialHu)} HU`, COLOR.ax, 'tr', 15, 0.55);
  if (!currentArch) return;
  const line = (pts: Pt[]) => { ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.stroke(); };
  // focal trough borders
  const tracked = slab().trackMandible && !!currentArch.panorama;
  const points = currentArch.points.map((_, i) => panoramaPoint(currentArch!, i, num('in-slice'), tracked));
  const t = slab().thicknessMm / 2;
  if (t > 0) {
    ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(53,224,161,.55)';
    for (const sign of [-1, 1]) line(points.map((p, i) => [p[0] + sign * t * currentArch!.normals[i][0], p[1] + sign * t * currentArch!.normals[i][1]] as Pt));
  }
  if (tracked) { ctx.setLineDash([3, 4]); ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,211,77,.5)'; line(currentArch.points); ctx.setLineDash([]); }
  ctx.lineWidth = 2; ctx.strokeStyle = COLOR.cross; line(points);
  // the 5 cross-section planes; the first one carries its number like the reference
  const secs = sections(), h = half();
  ctx.font = '600 12px Inter, sans-serif'; ctx.textBaseline = 'middle';
  secs.forEach((sec, k) => {
    const f = archFrameAt(currentArch!, sec.s);
    const a: Pt = [f.p[0] - f.n[0] * h, f.p[1] - f.n[1] * h], b: Pt = [f.p[0] + f.n[0] * h, f.p[1] + f.n[1] * h];
    ctx.lineWidth = k === 2 ? 2 : 1; ctx.strokeStyle = k === 2 ? '#7dffd0' : 'rgba(53,224,161,.8)'; line([a, b]);
    if (k === 0) { ctx.fillStyle = COLOR.cross; ctx.textAlign = 'center'; ctx.fillText(String(sec.number), X(b[0]) + f.n[0] * 12, Y(b[1]) + f.n[1] * 12); }
  });
  ctx.fillStyle = '#ffd34d'; ctx.strokeStyle = '#10131a'; ctx.lineWidth = 1.5;
  for (const point of controlPoints) { ctx.beginPath(); ctx.arc(X(point[0]), Y(point[1]), 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
}

// ---- panorama -------------------------------------------------------------------------------
function drawPano(interactive = false) {
  if (!volume || !currentArch) return;
  const usedGpu = renderer.render(volume, currentArch, slab(), windowLevel());
  cvPano.hidden = !usedGpu; cvPanoCpu.hidden = usedGpu;
  if (!usedGpu) toCanvas(cvPanoCpu, renderPanorama(volume, currentArch, slab(), interactive ? 2 : Infinity), windowLevel());
  panoGeom = fitInto(usedGpu ? cvPano : cvPanoCpu, panoStage, currentArch.length, volume.nz * volume.sz);
  fitInto(usedGpu ? cvPanoCpu : cvPano, panoStage, currentArch.length, volume.nz * volume.sz);
  drawPanoOverlay();
}

function drawPanoOverlay() {
  const { ctx } = prepOverlay(ovPano);
  if (!volume || !currentArch || !panoGeom) return;
  const g = panoGeom;
  drawRulers(ctx, g, { color: COLOR.pano });
  // axial level (orange) – rows are z, top = highest
  const y = g.top + g.height * (1 - (num('in-slice') + 0.5) / volume.nz);
  ctx.strokeStyle = COLOR.level; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(g.left, y); ctx.lineTo(g.left + g.width, y); ctx.stroke();
  // the 5 cross-section positions (green) with the first number on top
  sections().forEach((sec, k) => {
    const x = g.left + (sec.s / currentArch!.length) * g.width;
    ctx.strokeStyle = k === 2 ? '#7dffd0' : 'rgba(53,224,161,.8)'; ctx.lineWidth = k === 2 ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(x, g.top); ctx.lineTo(x, g.top + g.height); ctx.stroke();
    if (k === 0) { ctx.fillStyle = COLOR.cross; ctx.font = '600 12px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(String(sec.number), x, g.top + 14); }
  });
  if (hover?.panel === 'pano') drawCornerText(ctx, g, `${Math.round(hover.hu)} HU`, 'rgba(141,150,255,.9)', 'tr', 12);
}

// ---- cross-sections -------------------------------------------------------------------------
function ensureCrossCells() {
  if (crossCells.length) return;
  for (let k = 0; k < 5; k++) {
    const cell = document.createElement('div'); cell.className = 'cross-cell';
    const cv = document.createElement('canvas'), ov = document.createElement('canvas'); ov.className = 'overlay';
    cell.append(cv, ov); crossStrip.append(cell);
    cell.addEventListener('mousemove', (event) => crossHover(k, event));
    cell.addEventListener('mouseleave', () => { if (hover?.panel === k) { hover = null; drawCrossOverlay(k); } });
    cell.addEventListener('wheel', (event) => { event.preventDefault(); stepCenter(event.deltaY > 0 ? 1 : -1); }, { passive: false });
    crossCells.push({ cell, cv, ov });
  }
}

let crossRefineTimer = 0;
function drawCross(resample = true, interactive = false) {
  ensureCrossCells();
  window.clearTimeout(crossRefineTimer);
  const secs = sections();
  let draft = false;
  crossCells.forEach((c, k) => {
    if (!volume || !currentArch) { prepOverlay(c.ov); c.cv.width = 1; c.cv.height = 1; return; }
    const s = slab(), wl = windowLevel();
    if (renderer.renderCross(c.cv, volume, archFrameAt(currentArch, secs[k].s), half(), s, wl)) {
      crossImgs[k] = null;
    } else {
      if (resample || !crossImgs[k]) {
        crossImgs[k] = renderCrossAt(volume, currentArch, secs[k].s, half(), s.thicknessMm, s.mode, interactive ? 2 : Infinity);
        if (interactive) draft = true;
      }
      toCanvas(c.cv, crossImgs[k]!, wl);
    }
    crossGeoms[k] = fitInto(c.cv, c.cell, 2 * half(), volume.nz * volume.sz, CROSS_PAD);
    drawCrossOverlay(k);
  });
  if (draft) crossRefineTimer = window.setTimeout(() => drawCross(true), 200);
}

function drawCrossOverlay(k: number) {
  const c = crossCells[k];
  const { ctx } = prepOverlay(c.ov);
  const g = crossGeoms[k], sec = sections()[k];
  if (!volume || !currentArch || !g || !sec) return;
  drawRulers(ctx, g, { color: COLOR.cross, hDivisions: 2, minorPerMajor: 4, font: '9px Inter, sans-serif' });
  const [left, right] = sideLabels(currentArch, sec.s);
  drawOrientation(ctx, g, left, right, COLOR.cross);
  drawCornerText(ctx, g, String(sec.number), COLOR.cross, 'top-center', k === 2 ? 14 : 12, k === 2 ? 1 : 0.85);
  if (slab().trackMandible && currentArch.panorama) {
    // Cross-sections remain planar. These curves show exactly which part of each
    // plane contributes to the panorama, rather than warping a measurement view.
    ctx.save(); ctx.beginPath(); ctx.rect(g.left, g.top, g.width, g.height); ctx.clip();
    const column = sec.s / currentArch.length * (currentArch.points.length - 1), thickness = slab().thicknessMm;
    for (const sign of [-1, 1]) {
      ctx.beginPath(); ctx.strokeStyle = 'rgba(53,224,161,.65)'; ctx.lineWidth = 1;
      for (let z = 0; z < volume.nz; z++) {
        const offset = panoramaOffset(currentArch, column, z) + sign * thickness / 2;
        const x = g.left + (offset / (2 * half()) + 0.5) * g.width, y = g.top + (1 - (z + 0.5) / volume.nz) * g.height;
        if (z) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }
      ctx.stroke();
    }
    ctx.restore();
  }
  if (hover?.panel === k) drawCornerText(ctx, g, `${Math.round(hover.hu)} HU`, 'rgba(53,224,161,.9)', 'tr', 10);
}

function crossHover(k: number, event: MouseEvent) {
  if (!volume || !currentArch) return;
  const r = crossCells[k].cv.getBoundingClientRect();
  const fx = (event.clientX - r.left) / r.width, fy = (event.clientY - r.top) / r.height;
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;
  const sec = sections()[k], f = archFrameAt(currentArch, sec.s), off = -half() + fx * 2 * half();
  const hu = sample(volume, (f.p[0] + f.n[0] * off) / volume.sx, (f.p[1] + f.n[1] * off) / volume.sy, (1 - fy) * (volume.nz - 1));
  hover = { panel: k, hu }; scheduleOverlay(k);
}

// ---- coordinated refresh --------------------------------------------------------------------
/** Hover (HU readout) redraws are coalesced to one per animation frame instead of one per mousemove event. */
const pendingOverlays = new Set<'axial' | 'pano' | number>();
let overlayFrame = 0;
function scheduleOverlay(which: 'axial' | 'pano' | number) {
  pendingOverlays.add(which);
  if (overlayFrame) return;
  overlayFrame = requestAnimationFrame(() => {
    overlayFrame = 0;
    for (const w of pendingOverlays) { if (w === 'axial') drawAxialOverlay(); else if (w === 'pano') drawPanoOverlay(); else drawCrossOverlay(w); }
    pendingOverlays.clear();
  });
}
let guideTimer = 0;
function schedule3d() {
  window.clearTimeout(guideTimer);
  guideTimer = window.setTimeout(() => {
    if (!viewer3d || !volume) return;
    viewer3d.setGuides(currentArch, num('in-slice') * volume.sz, sections().map((s) => ({ s: s.s, halfWidthMm: half() })), slab().trackMandible);
  }, 250); // VTK rebuilds the guide actors and re-renders the whole volume: only once the user pauses
}
let sectionFrame = 0;
/** The 5-section group moved (or spacing/width changed). */
function scheduleSections() {
  if (sectionFrame) return;
  sectionFrame = requestAnimationFrame(() => { sectionFrame = 0; drawAxialOverlay(); drawPanoOverlay(); drawCross(true, true); schedule3d(); });
}
function stepCenter(direction: number) { if (!currentArch) return; centerS += direction * num('in-spacing'); clampCenter(); scheduleSections(); }
/**
 * Physical aspect is always preserved (mm accuracy). To avoid black side bars, size the two grid columns
 * proportionally to the width each top-row image needs at the available row height.
 */
function adaptColumns() {
  if (!volume) return;
  const grid = $('viewer'), cs = getComputedStyle(grid);
  const gap = parseFloat(cs.rowGap) || 0;
  const rowH = (grid.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom) - gap) / 2;
  const availH = Math.max(50, rowH - RULER_PAD.top - RULER_PAD.bottom - 2);
  const chrome = 34 + 2 + RULER_PAD.left + RULER_PAD.right;
  const aAxial = (volume.nx * volume.sx) / (volume.ny * volume.sy);
  const aPano = (currentArch ? currentArch.length : 1.7 * volume.nz * volume.sz) / (volume.nz * volume.sz);
  const wA = availH * aAxial + chrome, wP = availH * aPano + chrome;
  grid.style.gridTemplateColumns = `minmax(0, ${wA.toFixed(1)}fr) minmax(0, ${wP.toFixed(1)}fr)`;
}
function refresh() { adaptColumns(); drawAxial(); drawPano(); drawCross(true); schedule3d(); }
function refreshWindow() { adaptColumns(); drawAxial(); drawPano(); drawCross(false); }

function clearDerivedViews() {
  for (const canvas of [cvPano, cvPanoCpu]) { canvas.width = 1; canvas.height = 1; }
  cvPano.hidden = true; cvPanoCpu.hidden = false;
  crossImgs.length = 0; crossGeoms.length = 0; panoGeom = null;
  prepOverlay(ovPano); drawCross(true);
}

// ---- arch -----------------------------------------------------------------------------------
function saveJaw(which: Jaw = selectedJaw) { if (currentArch) arches.set(which, { arch: currentArch, controls: controlPoints.map((p) => [...p] as Pt) }); }
function loadJaw(which: Jaw = selectedJaw) {
  const saved = arches.get(which);
  currentArch = saved?.arch ?? null; controlPoints = saved?.controls.map((p) => [...p] as Pt) ?? [];
  centerS = currentArch ? currentArch.length / 2 : 0; crossImgs.length = 0;
  $<HTMLInputElement>('in-tracking').disabled = !currentArch?.panorama;
}
function seedControls() {
  if (!currentArch) return;
  if (currentArch.controls) { controlPoints = currentArch.controls.map(p => [...p] as Pt); return; }
  const count = 9, last = currentArch.points.length - 1;
  controlPoints = Array.from({ length: count }, (_, i) => [...currentArch!.points[Math.round((i * last) / (count - 1))]] as Pt);
}
function autoSlice(): number {
  if (!volume) return 0;
  const detected = findOcclusalSlice(volume, 20);
  return detected >= 0 ? detected : Math.round(volume.nz * .55);
}

/** Heuristic/AI-mask arch detection. Fails closed: a doubtful curve is withheld, never guessed. */
let detectToken = 0;
async function detectArch(): Promise<boolean> {
  const token = ++detectToken;
  if (!volume) return false;
  const v = volume;
  const input = $<HTMLInputElement>('in-slice');
  const z = Math.max(0, Math.min(v.nz - 1, Number.isFinite(Number(input.value)) ? num('in-slice') : autoSlice()));
  // The bundled model is mandible only; never imply the same mask identifies the maxilla.
  const useMandibleMask = !!v.mask && jaw() === 'lower';
  let result;
  toast('Arch aranıyor…', 'info', 2500);
  try { result = await searchArch(v, z, num('in-arch-slab'), useMandibleMask, () => token !== detectToken || volume !== v); }
  catch (error) { toast(String((error as Error).message), 'error'); return false; }
  if (token !== detectToken || volume !== v) return false;
  if (!result.anyEvidence) {
    toast('Güvenli bir otomatik arch için yeterli diş kanıtı yok. Aksiyel kesiti değiştirip "Arch\'ı yeniden tespit et" düğmesine basın.', 'warn');
    return false;
  }
  const chosen = result.chosen;
  if (!chosen) {
    const ev = result.best?.ev;
    toast(`Güvenilir alt çene eğrisi bulunamadı; yanlış panorama üretmemek için otomatik arch reddedildi. En iyi aday eğrinin yalnızca %${((ev?.supportFraction ?? 0) * 100).toFixed(0)}'ini yerel kanıtla destekliyor (en uzun desteksiz bölüm ${(ev?.longestUnsupportedMm ?? 0).toFixed(1)} mm). Başka bir aksiyel seviye deneyin veya kontrol noktalarını elle yerleştirin.`, 'warn', 10000);
    return false;
  }
  currentArch = chosen.arch;
  if (useMandibleMask) trackMandible(v, currentArch, autoSlice());
  $<HTMLInputElement>('in-tracking').disabled = !currentArch.panorama;
  seedControls(); saveJaw(); centerS = currentArch.length / 2; crossImgs.length = 0;
  refresh();
  const label = useMandibleMask ? 'Alt çene arch\'ı (AI maskesi)' : 'Otomatik arch (çene doğrulanmadı)';
  setCaseInfo(`${caseSummary()} · ${label}`);
  return true;
}


function rebuildFromControls() {
  if (!currentArch || controlPoints.length < 3) return;
  currentArch = splineArch(controlPoints, currentArch.points.length);
  if (dragIndex < 0 && volume && selectedJaw === 'lower') trackMandible(volume, currentArch, autoSlice());
  $<HTMLInputElement>('in-tracking').disabled = !currentArch.panorama;
  clampCenter(); saveJaw(); crossImgs.length = 0;
}

const caseSummary = () => volume ? `${volume.nx}×${volume.ny}×${volume.nz} · ${volume.sx.toFixed(2)}×${volume.sy.toFixed(2)}×${volume.sz.toFixed(2)} mm` : '';

// ---- service plumbing -----------------------------------------------------------------------
async function errorText(response: Response): Promise<string> {
  const raw = await response.text().catch(() => response.statusText);
  try { return JSON.parse(raw).detail ?? raw; } catch { return raw || response.statusText; }
}
async function openPath(path: string): Promise<string> {
  const response = await fetch(`${API}/api/datasets/path`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) });
  if (!response.ok) throw new Error(await errorText(response));
  return (await response.json()).datasetId;
}
function uploadFiles(files: File[], onProgress: (fraction: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const form = new FormData(); files.forEach((file) => form.append('files', file));
    const xhr = currentXhr = new XMLHttpRequest();
    xhr.open('POST', `${API}/api/datasets/upload`);
    xhr.upload.onprogress = (event) => { if (event.lengthComputable) onProgress(event.loaded / event.total); };
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(body.datasetId); else reject(new Error(body.detail ?? xhr.statusText));
      } catch { reject(new Error(xhr.statusText || 'Yükleme başarısız')); }
    };
    xhr.onerror = () => reject(new Error('Yerel servise ulaşılamadı'));
    xhr.onabort = () => reject(new Cancelled());
    xhr.send(form);
  });
}
const reopen = (source: Source) => source.kind === 'path' ? openPath(source.path) : uploadFiles(source.files, () => undefined);

async function downloadVolume(datasetId: string, onProgress: (fraction: number) => void): Promise<Volume> {
  const response = await fetch(`${API}/api/datasets/${datasetId}/volume`);
  if (!response.ok) throw new Error(await errorText(response));
  const total = Number(response.headers.get('Content-Length')) || 0;
  if (!response.body || !total) { const buffer = await response.arrayBuffer(); onProgress(1); return volumeFromResponse(response, buffer, datasetId); }
  const reader = response.body.getReader(), out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.set(value, got); got += value.length; onProgress(got / total);
  }
  return volumeFromResponse(response, out.buffer, datasetId);
}

function cancelActiveJob() {
  if (!activeJob) return;
  const job = activeJob; activeJob = '';
  void fetch(`${API}/api/jobs/${job}/cancel`, { method: 'POST' }).catch(() => undefined);
}

/** Wait for the mandible AI result using the live stream, with slow polling as a safety net. */
async function runAi(datasetId: string, token: number): Promise<Uint8Array> {
  const start = await fetch(`${API}/api/datasets/${datasetId}/jobs/segment`, { method: 'POST' });
  if (!start.ok) throw new Error(await errorText(start));
  const jobId: string = (await start.json()).jobId; activeJob = jobId;
  await new Promise<void>((resolve, reject) => {
    let poll = 0, offlineTimer = 0;
    const finish = (settle: () => void) => {
      unsubscribe(); window.clearInterval(poll); window.clearTimeout(offlineTimer);
      link.removeEventListener('state', onState); link.removeEventListener('restarted', onRestart);
      settle();
    };
    const handle = (job: JobInfo) => {
      if (token !== loadToken) return finish(() => reject(new Cancelled()));
      loading.stage('ai', 'active', job.progress, `${job.provider || 'hazırlanıyor'} · ${job.stage}`);
      if (job.state === 'done') finish(resolve);
      else if (job.state === 'failed' || job.state === 'cancelled') finish(() => reject(new Error(job.detail || job.stage || job.state)));
    };
    const onState = () => {
      if (link.state === 'online') { window.clearTimeout(offlineTimer); loading.message(''); }
      else if (link.state === 'stopping') finish(() => reject(new Error('Servis yönetici tarafından durduruldu')));
      else {
        loading.message('Servis bağlantısı koptu, yeniden bağlanılıyor…');
        window.clearTimeout(offlineTimer);
        offlineTimer = window.setTimeout(() => finish(() => reject(new Error('Servis 30 saniye içinde geri gelmedi'))), 30000);
      }
    };
    const onRestart = () => finish(() => reject(new Restarted()));
    const unsubscribe = link.onJob(jobId, handle);
    link.addEventListener('state', onState); link.addEventListener('restarted', onRestart);
    poll = window.setInterval(async () => {
      try {
        const reply = await fetch(`${API}/api/jobs/${jobId}`);
        if (reply.ok) handle(await reply.json()); else if (reply.status === 404) finish(() => reject(new Restarted()));
      } catch { /* the stream/online watchdog reports connectivity */ }
    }, 2000);
  });
  activeJob = '';
  const maskResponse = await fetch(`${API}/api/jobs/${jobId}/mask`);
  if (!maskResponse.ok) throw new Error(await errorText(maskResponse));
  return new Uint8Array(await maskResponse.arrayBuffer());
}

// ---- case lifecycle -------------------------------------------------------------------------
function showViewer() { $('empty-state').hidden = true; $('viewer').hidden = false; $('btn-reset').hidden = false; }
function showEmpty() { $('viewer').hidden = true; $('empty-state').hidden = false; $('btn-reset').hidden = true; }
const checkToken = (token: number) => { if (token !== loadToken) throw new Cancelled(); };

async function openVolume(next: Volume, datasetId: string) {
  const previousDataset = activeDatasetId;
  volume = next; activeDatasetId = datasetId; currentArch = null; controlPoints = []; arches.clear(); hover = null; lastAxialHu = null;
  $<HTMLSelectElement>('in-jaw').value = 'lower'; selectedJaw = 'lower';
  showViewer();
  const slice = $<HTMLInputElement>('in-slice');
  slice.max = String(next.nz - 1); slice.value = String(autoSlice()); $('o-slice').textContent = slice.value;
  setCaseInfo(`${caseSummary()} · ${link.provider && link.provider !== 'none' ? link.provider : 'otomatik arch'}`);
  clearDerivedViews(); drawAxial(); detectArch();
  try {
    viewer3d?.dispose();
    const { Viewer3D } = await import('./viewer3d');
    viewer3d = new Viewer3D($('vtk-stage'));
    await viewer3d.setVolume(next);
    viewer3d.setMask(next.mask);
    applyViewer3dOptions(); schedule3d();
  } catch (error) { console.warn('3D view unavailable', error); toast('3D görünüm başlatılamadı (WebGL2 gerekli). 2D görünümler kullanılabilir.', 'warn'); viewer3d = null; }
  if (previousDataset && previousDataset !== datasetId) void fetch(`${API}/api/datasets/${previousDataset}`, { method: 'DELETE' }).catch(() => undefined);
}

async function loadCase(source: Source) {
  const token = ++loadToken;
  lastSource = source; skipRequested = false;
  let stage: StageId = 'read';
  loading.show();
  try {
    if (link.state !== 'online') { loading.stage('read', 'active', 0, 'Servis bekleniyor…'); await link.waitOnline(60000); checkToken(token); }
    loading.stage('read', 'active', 0, source.kind === 'path' ? 'Seri okunuyor…' : 'Dosyalar servise aktarılıyor…');
    let datasetId = source.kind === 'path' ? await openPath(source.path) : await uploadFiles(source.files, (p) => loading.stage('read', 'active', p, `%${Math.round(p * 100)} aktarıldı`));
    checkToken(token); loading.stage('read', 'done');

    stage = 'transfer'; loading.stage('transfer', 'active', 0);
    const vol = await downloadVolume(datasetId, (p) => loading.stage('transfer', 'active', p, `%${Math.round(p * 100)}`));
    checkToken(token); loading.stage('transfer', 'done');

    stage = 'ai'; loading.stage('ai', 'active', 0, 'Kuyrukta…');
    let aiWarning = '';
    for (let attempt = 0; ; attempt++) {
      try { vol.mask = await runAi(datasetId, token); loading.stage('ai', 'done'); break; }
      catch (error) {
        if (error instanceof Cancelled) throw error;
        if (error instanceof Restarted && attempt < 1) { loading.message('Servis yeniden başladı, vaka yeniden kaydediliyor…'); datasetId = await reopen(source); vol.datasetId = datasetId; loading.message(''); continue; }
        aiWarning = skipRequested ? 'AI atlandı; otomatik arch kullanılıyor.' : `AI segmentasyonu tamamlanamadı (${(error as Error).message}); otomatik arch kullanılıyor.`;
        loading.stage('ai', 'skipped', 1, skipRequested ? 'Atlandı' : 'Başarısız, otomatik arch');
        break;
      }
    }
    checkToken(token);

    stage = 'build'; loading.stage('build', 'active', 0.2, 'Arch, panorama ve 3D hazırlanıyor…');
    await openVolume(vol, datasetId);
    checkToken(token);
    loading.hide();
    if (aiWarning) toast(`${aiWarning} Kontrol noktalarını gözden geçirin.`, 'warn', 10000);
  } catch (error) {
    if (error instanceof Cancelled) return;
    loading.fail(stage, (error as Error).message || 'Bilinmeyen hata');
  }
}

async function onLoadClick() {
  if (isDemo && link.state !== 'online') { void openVolume(makePhantom(), ''); return; }
  if (link.state !== 'online') { toast(CONN_HINT[link.state], 'warn'); return; }
  try {
    // The Electron manager owns the native folder dialog; no volume data is copied through the browser.
    const response = await fetch(`${API}/api/pick-folder`, { method: 'POST' });
    if (response.status === 409) { $<HTMLInputElement>('folder-dicom').click(); return; }
    const body = await response.json();
    if (body.path) await loadCase({ kind: 'path', path: body.path });
  } catch (error) { toast(`Klasör seçilemedi: ${(error as Error).message}`, 'error'); }
}

async function resetCase() {
  loadToken++; cancelActiveJob(); currentXhr?.abort();
  const dataset = activeDatasetId; activeDatasetId = '';
  volume = null; currentArch = null; controlPoints = []; arches.clear(); hover = null; lastAxialHu = null; lastSource = null;
  axialGeom = panoGeom = null; crossImgs.length = 0; crossGeoms.length = 0;
  viewer3d?.dispose(); viewer3d = null; renderer.release();
  crossCells.length = 0; crossStrip.replaceChildren();
  closeResetConfirm(); closePop(); loading.hide(); setCaseInfo(''); showEmpty();
  if (dataset) void fetch(`${API}/api/datasets/${dataset}`, { method: 'DELETE' }).catch(() => undefined);
}

// ---- live service events --------------------------------------------------------------------
link.addEventListener('state', updateServiceUi);
link.addEventListener('event', (event) => {
  const { event: name, data } = (event as CustomEvent<ServerEventDetail>).detail;
  if (name === 'clients') updateServiceUi();
  if (name === 'service.stopping' && volume && !loading.visible) toast('Servis durduruldu. Görüntüler ekranda kalır; AI ve yeniden yükleme için servisi başlatın.', 'warn', 12000);
  if (name === 'dataset.released' && data.datasetId === activeDatasetId && data.reason !== 'viewer') {
    activeDatasetId = ''; toast('Vaka serviste serbest bırakıldı. Görüntüler ekranda kalır.', 'warn');
  }
});
link.addEventListener('restarted', () => {
  updateServiceUi();
  if (!volume || loading.visible || !lastSource) return;
  reopen(lastSource).then((id) => { activeDatasetId = id; if (volume) volume.datasetId = id; toast('Servis yeniden bağlandı; vaka yeniden kaydedildi.', 'ok'); })
    .catch((error) => toast(`Servis geri geldi ama vaka yeniden kaydedilemedi: ${(error as Error).message}`, 'warn'));
});

// ---- UI wiring ------------------------------------------------------------------------------
loading.onSkipAi(() => { skipRequested = true; cancelActiveJob(); });
loading.onCancel(() => { loadToken++; cancelActiveJob(); currentXhr?.abort(); loading.hide(); if (!volume) showEmpty(); });
loading.onRetry(() => { if (lastSource) void loadCase(lastSource); });

const resetModal = $('modal-reset-confirm');
const btnCancelReset = $('btn-reset-cancel');
const btnProceedReset = $('btn-reset-proceed');

function openResetConfirm() {
  resetModal.hidden = false;
  btnCancelReset.focus();
}

function closeResetConfirm() {
  resetModal.hidden = true;
}

for (const id of ['btn-load', 'btn-load-empty']) $(id).onclick = () => void onLoadClick();
$('btn-reset').onclick = () => openResetConfirm();
btnCancelReset.onclick = () => closeResetConfirm();
btnProceedReset.onclick = () => {
  closeResetConfirm();
  void resetCase();
  toast('Vaka ve analiz verileri temizlendi. Başlangıç ekranına dönüldü.', 'info', 4000);
};
resetModal.addEventListener('click', (event) => {
  if (event.target === resetModal) closeResetConfirm();
});
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !resetModal.hidden) closeResetConfirm();
});
$<HTMLInputElement>('folder-dicom').onchange = (event) => {
  const input = event.target as HTMLInputElement, files = Array.from(input.files ?? []);
  input.value = '';
  if (files.length) void loadCase({ kind: 'files', files });
};

// drag-and-drop a folder onto the empty screen
const emptyState = $('empty-state');
emptyState.addEventListener('dragover', (event) => { event.preventDefault(); emptyState.classList.add('drag'); });
emptyState.addEventListener('dragleave', () => emptyState.classList.remove('drag'));
emptyState.addEventListener('drop', async (event) => {
  event.preventDefault(); emptyState.classList.remove('drag');
  const entries = Array.from(event.dataTransfer?.items ?? []).map((item) => item.webkitGetAsEntry?.()).filter(Boolean) as FileSystemEntry[];
  const files: File[] = [];
  const walk = async (entry: FileSystemEntry): Promise<void> => {
    if (entry.isFile) files.push(await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject)));
    else {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
        if (!batch.length) break;
        for (const child of batch) await walk(child);
      }
    }
  };
  for (const entry of entries) await walk(entry);
  if (!files.length) return;
  if (link.state !== 'online') return toast(CONN_HINT[link.state], 'warn');
  void loadCase({ kind: 'files', files });
});

// input bindings
const bind = (id: string, callback: () => void) => $<HTMLInputElement>(id).addEventListener('input', () => {
  const output = document.getElementById(id.replace('in-', 'o-')); if (output) output.textContent = $<HTMLInputElement>(id).value; callback();
});
const windowChanged = () => { $<HTMLSelectElement>('in-wlpreset').value = 'custom'; refreshWindow(); };
bind('in-slice', () => { drawAxial(); drawPanoOverlay(); schedule3d(); });
bind('in-thick', () => { drawPano(); drawCross(true); drawAxialOverlay(); });
bind('in-wc', windowChanged); bind('in-ww', windowChanged);
bind('in-arch-slab', () => undefined);
$<HTMLInputElement>('in-max').addEventListener('change', () => { drawPano(); drawCross(true); });
$<HTMLInputElement>('in-tracking').addEventListener('change', () => { drawPano(); drawAxialOverlay(); drawCrossOverlay(0); for (let k = 1; k < crossCells.length; k++) drawCrossOverlay(k); schedule3d(); });
$<HTMLSelectElement>('in-spacing').addEventListener('change', scheduleSections);
$<HTMLSelectElement>('in-width').addEventListener('change', scheduleSections);
$<HTMLSelectElement>('in-jaw').onchange = () => { const next = jaw(); saveJaw(selectedJaw); selectedJaw = next; loadJaw(next); if (!currentArch) { clearDerivedViews(); drawAxial(); detectArch(); } else refresh(); };
$<HTMLSelectElement>('in-wlpreset').addEventListener('change', (event) => {
  const preset = WL_PRESETS[(event.target as HTMLSelectElement).value]; if (!preset) return;
  $<HTMLInputElement>('in-wc').value = String(preset.wc); $<HTMLInputElement>('in-ww').value = String(preset.ww);
  $('o-wc').textContent = String(preset.wc); $('o-ww').textContent = String(preset.ww); refreshWindow();
});
$('tool-redetect').onclick = () => { if (!volume) return; currentArch = null; controlPoints = []; arches.delete(jaw()); clearDerivedViews(); drawAxial(); detectArch(); };
$('tool-reseed').onclick = () => { if (!currentArch) return; seedControls(); rebuildFromControls(); refresh(); };
$('tool-export').onclick = () => { const canvas = cvPano.hidden ? cvPanoCpu : cvPano; const a = document.createElement('a'); a.href = canvas.toDataURL('image/png'); a.download = `${jaw()}-panoramic.png`; a.click(); };

// 3D controls
function applyViewer3dOptions() {
  if (!viewer3d) return;
  viewer3d.setPreset($<HTMLSelectElement>('in-preset').value as Parameters<Viewer3D['setPreset']>[0]);
  viewer3d.setOpacity(num('in-opacity')); viewer3d.setBrightness(num('in-brightness')); viewer3d.setClip($<HTMLInputElement>('in-clip').checked);
  viewer3d.setGuideVisibility({
    arch: $<HTMLInputElement>('in-g-arch').checked, planes: $<HTMLInputElement>('in-g-planes').checked,
    level: $<HTMLInputElement>('in-g-level').checked, surface: $<HTMLInputElement>('in-g-surface').checked,
  });
}
$<HTMLSelectElement>('in-preset').addEventListener('change', applyViewer3dOptions);
bind('in-opacity', () => viewer3d?.setOpacity(num('in-opacity'))); bind('in-brightness', () => viewer3d?.setBrightness(num('in-brightness')));
for (const id of ['in-clip', 'in-g-arch', 'in-g-planes', 'in-g-level', 'in-g-surface']) $<HTMLInputElement>(id).addEventListener('change', applyViewer3dOptions);
$('tool-3d-reset').onclick = () => viewer3d?.resetCamera();
$('tool-3d-shot').onclick = () => { const url = viewer3d?.screenshot(); if (!url) return; const a = document.createElement('a'); a.href = url; a.download = '3d-view.png'; a.click(); };
document.querySelectorAll<HTMLButtonElement>('.tool.dir').forEach((button) => { button.onclick = () => viewer3d?.view(button.dataset.dir as Parameters<Viewer3D['view']>[0]); });

// axial interaction: drag points, wheel = slice, right-drag = window/level, hover = HU
const axialPoint = (event: MouseEvent): Pt | null => {
  if (!volume || !axialGeom) return null;
  const r = cvAxial.getBoundingClientRect();
  return [((event.clientX - r.left) / r.width) * volume.nx * volume.sx, ((event.clientY - r.top) / r.height) * volume.ny * volume.sy];
};
function hitControl(event: MouseEvent): number {
  const point = axialPoint(event); if (!point || !volume) return -1;
  const r = cvAxial.getBoundingClientRect(), radius = 12 * ((volume.nx * volume.sx) / r.width);
  let hit = -1, distance = radius;
  controlPoints.forEach((c, i) => { const d = Math.hypot(c[0] - point[0], c[1] - point[1]); if (d < distance) { hit = i; distance = d; } });
  return hit;
}
let dragFrame = 0;
let wlDrag: { x: number; y: number; wc: number; ww: number; moved: boolean; deleteIndex: number } | null = null;
stageAxial.addEventListener('contextmenu', (event) => event.preventDefault());
stageAxial.addEventListener('mousedown', (event) => {
  if (!volume) return;
  if (event.button === 0 && currentArch) { dragIndex = hitControl(event); if (dragIndex >= 0) event.preventDefault(); }
  if (event.button === 2) { wlDrag = { x: event.clientX, y: event.clientY, wc: num('in-wc'), ww: num('in-ww'), moved: false, deleteIndex: currentArch ? hitControl(event) : -1 }; event.preventDefault(); }
});
let panoDrag = false;
window.addEventListener('mousemove', (event) => {
  if (wlDrag) {
    const dx = event.clientX - wlDrag.x, dy = event.clientY - wlDrag.y;
    if (Math.hypot(dx, dy) > 3) wlDrag.moved = true;
    if (wlDrag.moved) {
      const ww = Math.max(50, Math.min(4000, wlDrag.ww + dx * 6)), wc = Math.max(-500, Math.min(2500, wlDrag.wc - dy * 6));
      $<HTMLInputElement>('in-wc').value = String(Math.round(wc)); $<HTMLInputElement>('in-ww').value = String(Math.round(ww));
      $('o-wc').textContent = String(Math.round(wc)); $('o-ww').textContent = String(Math.round(ww)); $<HTMLSelectElement>('in-wlpreset').value = 'custom';
      if (!dragFrame) dragFrame = requestAnimationFrame(() => { dragFrame = 0; refreshWindow(); });
    }
    return;
  }
  if (dragIndex >= 0) {
    const point = axialPoint(event); if (!point) return; controlPoints[dragIndex] = point;
    if (!dragFrame) dragFrame = requestAnimationFrame(() => { dragFrame = 0; rebuildFromControls(); drawAxialOverlay(); drawPano(true); drawCross(true, true); });
  }
});
window.addEventListener('mouseup', () => {
  panoDrag = false;
  if (wlDrag) {
    // a right click without movement on a control point removes that point
    if (!wlDrag.moved && wlDrag.deleteIndex >= 0 && controlPoints.length > 3) { controlPoints.splice(wlDrag.deleteIndex, 1); rebuildFromControls(); refresh(); }
    wlDrag = null;
  }
  if (dragIndex >= 0) { dragIndex = -1; rebuildFromControls(); refresh(); }
});
stageAxial.addEventListener('mousemove', (event) => {
  if (!volume || !axialGeom) return;
  cvAxial.style.cursor = hitControl(event) >= 0 || dragIndex >= 0 ? 'grab' : 'crosshair';
  const p = axialPoint(event); if (!p) return;
  const x = Math.round(p[0] / volume.sx), y = Math.round(p[1] / volume.sy);
  if (x < 0 || y < 0 || x >= volume.nx || y >= volume.ny) return;
  const hu = volume.data[x + volume.nx * (y + volume.ny * num('in-slice'))];
  hover = { panel: 'axial', hu }; lastAxialHu = hu; scheduleOverlay('axial');
});
stageAxial.addEventListener('mouseleave', () => { if (hover?.panel === 'axial') { hover = null; drawAxialOverlay(); } });
stageAxial.addEventListener('wheel', (event) => {
  if (!volume) return; event.preventDefault();
  const slice = $<HTMLInputElement>('in-slice'), step = (event.shiftKey ? 5 : 1) * (event.deltaY > 0 ? 1 : -1);
  slice.value = String(Math.max(0, Math.min(volume.nz - 1, Number(slice.value) + step))); $('o-slice').textContent = slice.value;
  drawAxial(); drawPanoOverlay(); schedule3d();
}, { passive: false });
cvAxial.addEventListener('dblclick', (event) => {
  if (!currentArch || hitControl(event) >= 0) return;
  const point = axialPoint(event); if (!point) return;
  let insert = 0, best = Infinity;
  for (let i = 0; i < controlPoints.length - 1; i++) {
    const [a, b] = [controlPoints[i], controlPoints[i + 1]], dx = b[0] - a[0], dy = b[1] - a[1], length = dx * dx + dy * dy || 1;
    const t = Math.max(0, Math.min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length));
    const distance = Math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy));
    if (distance < best) { best = distance; insert = i; }
  }
  controlPoints.splice(insert + 1, 0, point); rebuildFromControls(); refresh();
});

// panorama interaction: click/drag positions the 5-section group, wheel steps it, hover = HU
function setCenterFromPano(event: MouseEvent) {
  if (!currentArch || !panoGeom) return;
  const r = (cvPano.hidden ? cvPanoCpu : cvPano).getBoundingClientRect();
  centerS = ((event.clientX - r.left) / r.width) * currentArch.length; clampCenter(); scheduleSections();
}
panoStage.addEventListener('mousedown', (event) => { if (event.button === 0) { panoDrag = true; setCenterFromPano(event); } });
panoStage.addEventListener('mousemove', (event) => {
  if (panoDrag) setCenterFromPano(event);
  if (!volume || !currentArch || !panoGeom) return;
  const r = (cvPano.hidden ? cvPanoCpu : cvPano).getBoundingClientRect();
  const fx = (event.clientX - r.left) / r.width, fy = (event.clientY - r.top) / r.height;
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;
  const f = archFrameAt(currentArch, fx * currentArch.length);
  const z = (1 - fy) * (volume.nz - 1), offset = panoramaOffset(currentArch, fx * (currentArch.points.length - 1), z, slab().trackMandible);
  hover = { panel: 'pano', hu: sample(volume, (f.p[0] + f.n[0] * offset) / volume.sx, (f.p[1] + f.n[1] * offset) / volume.sy, z) }; scheduleOverlay('pano');
});
panoStage.addEventListener('mouseleave', () => { if (hover?.panel === 'pano') { hover = null; drawPanoOverlay(); } });
panoStage.addEventListener('wheel', (event) => { event.preventDefault(); stepCenter(event.deltaY > 0 ? 1 : -1); }, { passive: false });

window.addEventListener('resize', () => { if (volume) { refresh(); viewer3d?.resize(); } });

// ---- start ----------------------------------------------------------------------------------
$<HTMLSelectElement>('in-wlpreset').value = 'bone';
updateServiceUi();
link.connect();
if (isDemo) void openVolume(makePhantom(), '');
void centerIndex; // kept for 3D/export extensions that need the nearest arch index
