import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  onerror: (() => void) | null = null;
  closed = false;
  private listeners = new Map<string, ((message: { data: string }) => void)[]>();
  url: string;
  constructor(url: string) { this.url = url; FakeEventSource.instances.push(this); }
  addEventListener(name: string, fn: (message: { data: string }) => void) { this.listeners.set(name, [...(this.listeners.get(name) ?? []), fn]); }
  close() { this.closed = true; }
  emit(name: string, data: object) { for (const fn of this.listeners.get(name) ?? []) fn({ data: JSON.stringify(data) }); }
  static get last() { return FakeEventSource.instances[FakeEventSource.instances.length - 1]; }
}

const hello = (instanceId: string, apiVersion = 3) => ({ instanceId, apiVersion, engine: 'onnx', provider: 'CPU', startedAt: 1, clients: { viewers: 1 }, datasets: [], jobs: [] });

async function freshLink() {
  vi.resetModules();
  const { ServiceLink } = await import('./service');
  return new ServiceLink();
}

describe('ServiceLink', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeEventSource.instances = [];
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal('window', globalThis);
  });
  afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

  it('goes online on hello and exposes the service identity', async () => {
    const link = await freshLink(); link.connect();
    FakeEventSource.last.emit('hello', hello('a'));
    expect(link.state).toBe('online');
    expect(link.instanceId).toBe('a');
    expect(link.provider).toBe('CPU');
  });

  it('treats a missing heartbeat as lost and reconnects with backoff', async () => {
    const link = await freshLink(); link.connect();
    FakeEventSource.last.emit('hello', hello('a'));
    vi.advanceTimersByTime(6100);
    expect(link.state).toBe('connecting');
    expect(FakeEventSource.instances).toHaveLength(1);
    vi.advanceTimersByTime(600);
    expect(FakeEventSource.instances).toHaveLength(2);
  });

  it('stays offline (not connecting) if the service was never reachable', async () => {
    const link = await freshLink(); link.connect();
    FakeEventSource.last.onerror?.();
    expect(link.state).toBe('offline');
  });

  it('reports a restart when the instanceId changes after being online', async () => {
    const link = await freshLink(); link.connect();
    const restarted = vi.fn(); link.addEventListener('restarted', restarted);
    FakeEventSource.last.emit('hello', hello('a'));
    FakeEventSource.last.onerror?.(); vi.advanceTimersByTime(600);
    FakeEventSource.last.emit('hello', hello('a'));
    expect(restarted).not.toHaveBeenCalled();
    FakeEventSource.last.onerror?.(); vi.advanceTimersByTime(1100);
    FakeEventSource.last.emit('hello', hello('b'));
    expect(restarted).toHaveBeenCalledTimes(1);
  });

  it('flags an incompatible API version and stops reconnecting', async () => {
    const link = await freshLink(); link.connect();
    FakeEventSource.last.emit('hello', hello('a', 1));
    expect(link.state).toBe('incompatible');
    vi.advanceTimersByTime(20000);
    expect(FakeEventSource.instances).toHaveLength(1);
  });

  it('moves to stopping on service.stopping and then offline when the stream closes', async () => {
    const link = await freshLink(); link.connect();
    FakeEventSource.last.emit('hello', hello('a'));
    FakeEventSource.last.emit('service.stopping', { reason: 'user' });
    expect(link.state).toBe('stopping');
    FakeEventSource.last.onerror?.();
    expect(link.state).toBe('offline');
    vi.advanceTimersByTime(600);
    FakeEventSource.last.onerror?.();
    expect(link.state).toBe('offline');
  });

  it('delivers only the requested job and unsubscribes cleanly', async () => {
    const link = await freshLink(); link.connect();
    FakeEventSource.last.emit('hello', hello('a'));
    const seen: string[] = [];
    const stop = link.onJob('job-1', (job) => seen.push(`${job.state}:${job.progress}`));
    FakeEventSource.last.emit('job.progress', { jobId: 'job-2', state: 'running', progress: 0.9 });
    FakeEventSource.last.emit('job.progress', { jobId: 'job-1', state: 'running', progress: 0.5 });
    FakeEventSource.last.emit('job.done', { jobId: 'job-1', state: 'done', progress: 1 });
    stop();
    FakeEventSource.last.emit('job.progress', { jobId: 'job-1', state: 'running', progress: 0.1 });
    expect(seen).toEqual(['running:0.5', 'done:1']);
  });

  it('waitOnline resolves when the service comes up and rejects on timeout', async () => {
    const link = await freshLink(); link.connect();
    const pending = link.waitOnline(5000);
    FakeEventSource.last.emit('hello', hello('a'));
    await expect(pending).resolves.toBeUndefined();
    const other = await freshLink();
    const timeout = other.waitOnline(1000);
    const assertion = expect(timeout).rejects.toThrow();
    vi.advanceTimersByTime(1100);
    await assertion;
  });
});
