/** Live link to the local service (Server-Sent Events). One instance per viewer tab. */
export const API: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://127.0.0.1:8000';
export const API_VERSION = 2;

export type Conn = 'offline' | 'connecting' | 'online' | 'stopping' | 'incompatible';

export interface JobInfo {
  jobId: string; datasetId: string; state: string; stage: string; progress: number;
  detail: string; provider: string; startedAt?: number | null; finishedAt?: number | null;
}
export interface ServerEventDetail { event: string; data: any }

const EVENT_NAMES = [
  'hello', 'heartbeat', 'clients', 'dataset.opened', 'dataset.released',
  'job.queued', 'job.running', 'job.progress', 'job.done', 'job.failed', 'job.cancelled', 'service.stopping',
];
const HEARTBEAT_TIMEOUT_MS = 6000;
const BACKOFF_MS = [500, 1000, 2000, 4000, 5000];

/**
 * Emits:  `state`      – connection state changed (`link.state`)
 *         `event`      – every server event, detail = {event, data}
 *         `restarted`  – the service came back with a different instanceId
 */
export class ServiceLink extends EventTarget {
  state: Conn = 'offline';
  instanceId = '';
  engine = '';
  provider = '';
  startedAt = 0;
  viewers = 0;
  private source?: EventSource;
  private watchdog = 0;
  private retryTimer = 0;
  private attempt = 0;
  private everOnline = false;
  private closed = false;
  private stoppingByUser = false;

  connect(): void {
    this.closed = false;
    window.clearTimeout(this.retryTimer);
    this.source?.close();
    const source = this.source = new EventSource(`${API}/api/events?role=viewer`);
    for (const name of EVENT_NAMES) source.addEventListener(name, (message) => this.onMessage(name, (message as MessageEvent).data));
    source.onerror = () => { if (this.source === source) this.lost(); };
  }

  disconnect(): void { this.closed = true; this.source?.close(); window.clearTimeout(this.watchdog); window.clearTimeout(this.retryTimer); }

  private set(next: Conn): void {
    if (this.state === next) return;
    this.state = next;
    this.dispatchEvent(new Event('state'));
  }

  private arm(): void {
    window.clearTimeout(this.watchdog);
    this.watchdog = window.setTimeout(() => this.lost(), HEARTBEAT_TIMEOUT_MS);
  }

  private onMessage(event: string, raw: string): void {
    this.arm();
    let data: any = {};
    try { data = JSON.parse(raw); } catch { /* heartbeat/keep-alive frames */ }
    if (event === 'hello') {
      const restarted = this.everOnline && this.instanceId && data.instanceId !== this.instanceId;
      this.instanceId = data.instanceId; this.engine = data.engine; this.provider = data.provider; this.startedAt = data.startedAt;
      this.viewers = data.clients?.viewers ?? 0;
      this.attempt = 0;
      this.stoppingByUser = false;
      if (data.apiVersion !== API_VERSION) { this.set('incompatible'); this.source?.close(); window.clearTimeout(this.watchdog); return; }
      this.everOnline = true;
      this.set('online');
      if (restarted) this.dispatchEvent(new Event('restarted'));
    } else if (event === 'clients') {
      this.viewers = data.viewers ?? this.viewers;
    } else if (event === 'service.stopping') {
      this.stoppingByUser = true;
      this.set('stopping');
    }
    this.dispatchEvent(new CustomEvent<ServerEventDetail>('event', { detail: { event, data } }));
  }

  /** Stream broke (service stopped, crashed, or stalled): fall back to retrying with backoff. */
  private lost(): void {
    if (this.closed) return;
    this.source?.close();
    window.clearTimeout(this.watchdog);
    if (this.state === 'incompatible') {
      this.set('incompatible');
    } else if (this.stoppingByUser || this.attempt >= 1) {
      this.set('offline');
    } else {
      this.set(this.everOnline ? 'connecting' : 'offline');
    }
    const delay = BACKOFF_MS[Math.min(this.attempt++, BACKOFF_MS.length - 1)];
    this.retryTimer = window.setTimeout(() => this.connect(), delay);
  }

  waitOnline(timeoutMs: number): Promise<void> {
    if (this.state === 'online') return Promise.resolve();
    return new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => { this.removeEventListener('state', listener); reject(new Error('The local service did not come online')); }, timeoutMs);
      const listener = () => {
        if (this.state === 'online') { window.clearTimeout(timer); this.removeEventListener('state', listener); resolve(); }
        else if (this.state === 'incompatible') { window.clearTimeout(timer); this.removeEventListener('state', listener); reject(new Error('Service version is incompatible')); }
      };
      this.addEventListener('state', listener);
    });
  }

  /** Subscribe to one job's state changes. Returns an unsubscribe function. */
  onJob(jobId: string, callback: (job: JobInfo) => void): () => void {
    const listener = (event: Event) => {
      const { event: name, data } = (event as CustomEvent<ServerEventDetail>).detail;
      if (name.startsWith('job.') && data.jobId === jobId) callback(data as JobInfo);
    };
    this.addEventListener('event', listener);
    return () => this.removeEventListener('event', listener);
  }
}
