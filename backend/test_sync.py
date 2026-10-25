"""Live-synchronisation tests for the local service (stdlib unittest, no pytest).

Run:  backend/.venv/bin/python -m unittest backend/test_sync.py -v
A real uvicorn server is started on a free loopback port so that the SSE stream,
the worker-thread publishing and the graceful shutdown are exercised end to end.
The neural network is replaced by a fast fake engine.
"""
from __future__ import annotations

import json
import os
import queue
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ["PINECLINIC_CONTROL_TOKEN"] = "test-token"

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
import uvicorn  # noqa: E402

import main  # noqa: E402

TOKEN = {"X-Control-Token": "test-token"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SseClient:
    """Collects (event, data) tuples from /api/events in a background thread."""

    def __init__(self, base: str, role: str = "viewer", headers: dict[str, str] | None = None) -> None:
        self.events: queue.Queue[tuple[str, dict]] = queue.Queue()
        self._stop = threading.Event()
        self.status: int | None = None
        self._thread = threading.Thread(target=self._run, args=(base, role, headers or {}), daemon=True)
        self._thread.start()

    def _run(self, base: str, role: str, headers: dict[str, str]) -> None:
        try:
            with httpx.stream("GET", f"{base}/api/events", params={"role": role}, headers=headers, timeout=None) as response:
                self.status = response.status_code
                if response.status_code != 200:
                    self.events.put(("http-error", {"status": response.status_code}))
                    return
                event = ""
                for line in response.iter_lines():
                    if self._stop.is_set():
                        return
                    if line.startswith("event: "):
                        event = line[7:]
                    elif line.startswith("data: "):
                        self.events.put((event, json.loads(line[6:])))
        except httpx.HTTPError:
            self.events.put(("closed", {}))
            return
        self.events.put(("closed", {}))

    def wait_for(self, name: str, timeout: float = 5.0, where=lambda data: True) -> dict:
        deadline = time.time() + timeout
        seen = []
        while time.time() < deadline:
            try:
                event, data = self.events.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                break
            seen.append(event)
            if event == name and where(data):
                return data
        raise AssertionError(f"event {name!r} not received; saw {seen}")

    def close(self) -> None:
        self._stop.set()


def fake_segment(image, log, on_progress, is_cancelled):
    nx, ny, nz = image.GetSize()
    for tile in range(1, 5):
        time.sleep(0.15)
        if is_cancelled():
            raise main.onnx_infer.SegmentationCancelled()
        on_progress(tile, 4, 0.0)
    return np.ones((nz, ny, nx), dtype=np.uint8)


class SyncTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        main.CONTROL_TOKEN = "test-token"
        main.selected_engine = lambda: ("onnx", "Fake EP")
        main.onnx_infer.segment_mandible = fake_segment
        cls.port = free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=cls.port, log_level="warning", timeout_graceful_shutdown=2))
        main.SERVER = cls.server
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        for _ in range(100):
            try:
                if httpx.get(f"{cls.base}/api/health").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.should_exit = True
        cls.thread.join(timeout=5)

    def register(self, fingerprint: str) -> str:
        image = sitk.Image(8, 8, 4, sitk.sitkInt16)
        image.SetSpacing((0.5, 0.5, 0.5))
        dataset_id, _ = main.register_dataset(image, fingerprint, f"synthetic {fingerprint}")
        return dataset_id

    def test_00_same_source_is_a_fresh_case(self) -> None:
        self.assertNotEqual(self.register("same-source"), self.register("same-source"))

    def test_00_each_job_runs_inference(self) -> None:
        from unittest.mock import patch
        dataset_id = self.register("fresh-inference")
        with patch.object(main.onnx_infer, "segment_mandible", wraps=fake_segment) as infer:
            for index in range(2):
                job_id = f"fresh-job-{index}"
                main.JOBS[job_id] = main.Job(dataset_id=dataset_id)
                main.run_segmentation(job_id)
                self.assertEqual(main.JOBS[job_id].state, "done")
                self.assertEqual(main.JOBS[job_id].stage, "complete")
            self.assertEqual(infer.call_count, 2)

    def test_01_hello_and_health_identity(self) -> None:
        client = SseClient(self.base)
        hello = client.wait_for("hello")
        health = httpx.get(f"{self.base}/api/health").json()
        self.assertEqual(hello["instanceId"], health["instanceId"])
        self.assertEqual(hello["apiVersion"], main.API_VERSION)
        self.assertEqual(health["apiVersion"], main.API_VERSION)
        self.assertFalse(health["resultCacheEnabled"])
        self.assertEqual(httpx.get(f"{self.base}/api/health").headers["cache-control"], "no-store")
        client.wait_for("clients", where=lambda d: d["viewers"] >= 1)
        client.close()

    def test_02_job_lifecycle_is_streamed(self) -> None:
        client = SseClient(self.base)
        client.wait_for("hello")
        dataset_id = self.register("job-lifecycle")
        client.wait_for("dataset.opened", where=lambda d: d["datasetId"] == dataset_id)
        job_id = httpx.post(f"{self.base}/api/datasets/{dataset_id}/jobs/segment").json()["jobId"]
        mine = lambda d: d["jobId"] == job_id
        client.wait_for("job.queued", where=mine)
        client.wait_for("job.running", where=mine)
        progress = client.wait_for("job.progress", where=mine)
        self.assertGreater(progress["progress"], 0)
        done = client.wait_for("job.done", where=mine)
        self.assertEqual(done["progress"], 1.0)
        mask = httpx.get(f"{self.base}/api/jobs/{job_id}/mask")
        self.assertEqual(mask.status_code, 200)
        self.assertEqual(len(mask.content), 8 * 8 * 4)
        client.close()

    def test_03_release_dataset_cancels_and_announces(self) -> None:
        client = SseClient(self.base)
        client.wait_for("hello")
        dataset_id = self.register("release")
        job_id = httpx.post(f"{self.base}/api/datasets/{dataset_id}/jobs/segment").json()["jobId"]
        client.wait_for("job.running", where=lambda d: d["jobId"] == job_id)
        self.assertTrue(httpx.delete(f"{self.base}/api/datasets/{dataset_id}").json()["released"])
        client.wait_for("job.cancelled", where=lambda d: d["jobId"] == job_id)
        client.wait_for("dataset.released", where=lambda d: d["datasetId"] == dataset_id)
        self.assertEqual(httpx.get(f"{self.base}/api/datasets/{dataset_id}/volume").status_code, 404)
        client.close()

    def test_04_control_endpoints_require_token(self) -> None:
        self.assertEqual(httpx.post(f"{self.base}/api/control/shutdown").status_code, 403)
        self.assertEqual(httpx.post(f"{self.base}/api/control/shutdown", headers={"X-Control-Token": "wrong"}).status_code, 403)
        manager = SseClient(self.base, role="manager")
        self.assertEqual(manager.wait_for("http-error")["status"], 403)

    def test_05_pick_folder_round_trip(self) -> None:
        response = httpx.post(f"{self.base}/api/pick-folder")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["reason"], "no-manager")
        manager = SseClient(self.base, role="manager", headers=TOKEN)
        manager.wait_for("hello")
        result: dict = {}
        viewer_call = threading.Thread(target=lambda: result.update(httpx.post(f"{self.base}/api/pick-folder", timeout=10).json()))
        viewer_call.start()
        request = manager.wait_for("pick-folder.request")
        delivered = httpx.post(f"{self.base}/api/control/pick-folder/{request['requestId']}", json={"path": "/tmp/series"}, headers=TOKEN).json()
        self.assertTrue(delivered["delivered"])
        viewer_call.join(timeout=5)
        self.assertEqual(result, {"path": "/tmp/series"})
        manager.close()

    def test_99_graceful_shutdown_notifies_clients(self) -> None:
        viewer = SseClient(self.base)
        viewer.wait_for("hello")
        manager = SseClient(self.base, role="manager", headers=TOKEN)
        manager.wait_for("hello")
        self.assertEqual(httpx.post(f"{self.base}/api/control/shutdown", headers=TOKEN).status_code, 200)
        self.assertEqual(viewer.wait_for("service.stopping")["reason"], "user")
        manager.wait_for("service.stopping")
        self.thread.join(timeout=5)
        self.assertFalse(self.thread.is_alive(), "server did not exit after shutdown")


if __name__ == "__main__":
    unittest.main()
