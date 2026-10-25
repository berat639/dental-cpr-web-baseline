/** Full-screen staged loading overlay. Closes only when the caller calls `hide()`. */
export type StageId = 'read' | 'transfer' | 'ai' | 'build';
type StageState = 'pending' | 'active' | 'done' | 'skipped' | 'error';

const LABELS: Record<StageId, string> = {
  read: 'DICOM serisi okunuyor',
  transfer: 'Hacim aktarılıyor',
  ai: 'AI segmentasyonu (alt çene)',
  build: 'Görünümler hazırlanıyor',
};
const ORDER: StageId[] = ['read', 'transfer', 'ai', 'build'];

export class LoadingOverlay {
  private readonly root: HTMLElement;
  private readonly list: HTMLElement;
  private readonly notice: HTMLElement;
  private readonly skipButton: HTMLButtonElement;
  private readonly cancelButton: HTMLButtonElement;
  private readonly retryButton: HTMLButtonElement;
  private skipHandler: () => void = () => undefined;
  private cancelHandler: () => void = () => undefined;
  private retryHandler: () => void = () => undefined;

  constructor(root: HTMLElement) {
    this.root = root;
    root.innerHTML = `
      <div class="loading-card">
        <div class="loading-logo" aria-hidden="true">◖◗</div>
        <h2 id="loading-title">Vaka hazırlanıyor</h2>
        <ol class="loading-stages" id="loading-stages"></ol>
        <p class="loading-notice" id="loading-notice"></p>
        <div class="loading-actions">
          <button class="btn" id="loading-skip" hidden>AI'yı atla</button>
          <button class="btn primary" id="loading-retry" hidden>Tekrar dene</button>
          <button class="btn" id="loading-cancel">İptal</button>
        </div>
      </div>`;
    this.list = root.querySelector('#loading-stages')!;
    this.notice = root.querySelector('#loading-notice')!;
    this.skipButton = root.querySelector('#loading-skip')!;
    this.cancelButton = root.querySelector('#loading-cancel')!;
    this.retryButton = root.querySelector('#loading-retry')!;
    this.skipButton.onclick = () => this.skipHandler();
    this.cancelButton.onclick = () => this.cancelHandler();
    this.retryButton.onclick = () => this.retryHandler();
    this.reset();
  }

  onSkipAi(handler: () => void) { this.skipHandler = handler; }
  onCancel(handler: () => void) { this.cancelHandler = handler; }
  onRetry(handler: () => void) { this.retryHandler = handler; }

  private reset() {
    this.list.innerHTML = ORDER.map((id) => `
      <li data-stage="${id}" data-state="pending">
        <span class="stage-icon"></span>
        <div class="stage-body"><span class="stage-label">${LABELS[id]}</span><span class="stage-detail"></span>
          <div class="stage-bar"><i style="width:0%"></i></div></div>
      </li>`).join('');
    this.notice.textContent = '';
    this.skipButton.hidden = true; this.retryButton.hidden = true; this.cancelButton.hidden = false;
  }

  show() { this.reset(); this.root.hidden = false; }
  hide() { this.root.hidden = true; }
  get visible() { return !this.root.hidden; }

  stage(id: StageId, state: StageState, progress?: number, detail?: string) {
    const item = this.list.querySelector<HTMLElement>(`[data-stage="${id}"]`);
    if (!item) return;
    item.dataset.state = state;
    if (state === 'done' || state === 'skipped') progress = 1;
    if (progress !== undefined) item.querySelector<HTMLElement>('.stage-bar i')!.style.width = `${Math.round(Math.max(0, Math.min(1, progress)) * 100)}%`;
    if (detail !== undefined) item.querySelector('.stage-detail')!.textContent = detail;
    if (id === 'ai') this.skipButton.hidden = state !== 'active';
  }

  message(text: string) { this.notice.textContent = text; }

  /** Show an unrecoverable problem with Retry/Cancel instead of silently closing. */
  fail(id: StageId, text: string) {
    this.stage(id, 'error', undefined, text);
    this.notice.textContent = text;
    this.skipButton.hidden = true; this.retryButton.hidden = false;
  }
}
