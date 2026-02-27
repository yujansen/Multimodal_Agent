"""Tests for ResourceMonitor and ParallelExecutor."""

from __future__ import annotations

import platform
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from pinocchio.utils.resource_monitor import GPUInfo, ResourceMonitor, ResourceSnapshot
from pinocchio.utils.parallel_executor import ParallelExecutor


# =====================================================================
# GPUInfo
# =====================================================================

class TestGPUInfo:
    def test_utilisation_pct(self) -> None:
        gpu = GPUInfo(name="Test GPU", vram_total_mb=8000, vram_free_mb=2000, backend="cuda")
        assert gpu.utilisation_pct == pytest.approx(75.0)

    def test_utilisation_pct_zero_vram(self) -> None:
        gpu = GPUInfo(name="No GPU", vram_total_mb=0, vram_free_mb=0, backend="none")
        assert gpu.utilisation_pct == 0.0


# =====================================================================
# ResourceSnapshot
# =====================================================================

class TestResourceSnapshot:
    def test_has_gpu_true(self) -> None:
        snap = ResourceSnapshot(gpus=[GPUInfo(backend="cuda", vram_total_mb=8000)])
        assert snap.has_gpu is True

    def test_has_gpu_false(self) -> None:
        snap = ResourceSnapshot(gpus=[])
        assert snap.has_gpu is False

    def test_gpu_backend(self) -> None:
        snap = ResourceSnapshot(gpus=[GPUInfo(backend="mps", vram_total_mb=16000)])
        assert snap.gpu_backend == "mps"

    def test_gpu_backend_none(self) -> None:
        snap = ResourceSnapshot()
        assert snap.gpu_backend == "none"

    def test_total_vram(self) -> None:
        snap = ResourceSnapshot(gpus=[
            GPUInfo(vram_total_mb=8000, backend="cuda"),
            GPUInfo(vram_total_mb=8000, backend="cuda"),
        ])
        assert snap.total_vram_mb == 16000

    def test_recommended_workers_gpu_large(self) -> None:
        snap = ResourceSnapshot(
            cpu_count_physical=8,
            gpus=[GPUInfo(vram_total_mb=16000, vram_free_mb=12000, backend="cuda")],
        )
        assert snap.recommended_workers >= 2

    def test_recommended_workers_no_gpu_low_ram(self) -> None:
        snap = ResourceSnapshot(
            cpu_count_physical=4,
            ram_available_mb=4000,
        )
        assert snap.recommended_workers == 1

    def test_recommended_workers_no_gpu_high_ram(self) -> None:
        snap = ResourceSnapshot(
            cpu_count_physical=8,
            ram_available_mb=32000,
        )
        assert snap.recommended_workers == 2

    def test_to_dict(self) -> None:
        snap = ResourceSnapshot(
            cpu_count=8,
            cpu_count_physical=4,
            ram_total_mb=16000,
            ram_available_mb=8000,
            gpus=[GPUInfo(name="Test", vram_total_mb=8000, vram_free_mb=4000, backend="cuda")],
        )
        d = snap.to_dict()
        assert d["cpu_count"] == 8
        assert d["has_gpu"] is True
        assert d["gpu_backend"] == "cuda"
        assert len(d["gpus"]) == 1
        assert d["gpus"][0]["utilisation_pct"] == 50.0


# =====================================================================
# ResourceMonitor
# =====================================================================

class TestResourceMonitor:
    def test_snapshot_returns_data(self) -> None:
        mon = ResourceMonitor()
        snap = mon.snapshot()
        assert snap.cpu_count >= 1
        assert snap.ram_total_mb > 0

    def test_snapshot_caching(self) -> None:
        mon = ResourceMonitor()
        s1 = mon.snapshot()
        s2 = mon.snapshot()
        assert s1 is s2  # same object (cached)

    def test_snapshot_refresh(self) -> None:
        mon = ResourceMonitor()
        s1 = mon.snapshot()
        s2 = mon.snapshot(refresh=True)
        assert s1 is not s2  # new object after refresh

    def test_detects_cpu_cores(self) -> None:
        mon = ResourceMonitor()
        snap = mon.snapshot(refresh=True)
        assert snap.cpu_count_physical >= 1
        assert snap.cpu_count >= snap.cpu_count_physical


# =====================================================================
# ParallelExecutor
# =====================================================================

class TestParallelExecutor:
    def test_map_empty(self) -> None:
        exe = ParallelExecutor(max_workers=2)
        assert exe.map(lambda x: x * 2, []) == []

    def test_map_single_item(self) -> None:
        exe = ParallelExecutor(max_workers=2)
        assert exe.map(lambda x: x * 2, [5]) == [10]

    def test_map_parallel(self) -> None:
        exe = ParallelExecutor(max_workers=4)
        items = [1, 2, 3, 4, 5]
        results = exe.map(lambda x: x ** 2, items)
        assert results == [1, 4, 9, 16, 25]

    def test_map_sequential_fallback(self) -> None:
        exe = ParallelExecutor(max_workers=1)
        results = exe.map(lambda x: x + 1, [10, 20])
        assert results == [11, 21]

    def test_run_parallel_heterogeneous(self) -> None:
        exe = ParallelExecutor(max_workers=4)
        tasks = [
            (lambda a, b: a + b, (1, 2), {}),
            (lambda a, b: a * b, (3, 4), {}),
            (lambda a, b: a - b, (10, 5), {}),
        ]
        results = exe.run_parallel(tasks)
        assert results == [3, 12, 5]

    def test_run_parallel_empty(self) -> None:
        exe = ParallelExecutor(max_workers=4)
        assert exe.run_parallel([]) == []

    def test_run_parallel_sequential_fallback(self) -> None:
        exe = ParallelExecutor(max_workers=1)
        tasks = [
            (lambda: "a", (), {}),
            (lambda: "b", (), {}),
        ]
        results = exe.run_parallel(tasks)
        assert results == ["a", "b"]

    def test_submit_and_gather(self) -> None:
        exe = ParallelExecutor(max_workers=2)
        futures = [exe.submit(lambda x: x * 3, i) for i in range(5)]
        results = exe.gather(futures)
        assert results == [0, 3, 6, 9, 12]
        exe.shutdown()

    def test_context_manager(self) -> None:
        with ParallelExecutor(max_workers=2) as exe:
            result = exe.map(lambda x: x, [1, 2, 3])
            assert result == [1, 2, 3]

    def test_max_workers_property(self) -> None:
        exe = ParallelExecutor(max_workers=3)
        assert exe.max_workers == 3

    def test_refresh_resources(self) -> None:
        exe = ParallelExecutor(max_workers=2)
        snap = exe.refresh_resources()
        assert snap.cpu_count >= 1

    def test_parallel_actually_concurrent(self) -> None:
        """Verify tasks actually run in parallel by timing them."""
        exe = ParallelExecutor(max_workers=4)

        def slow_fn(x: int) -> int:
            time.sleep(0.1)
            return x

        start = time.monotonic()
        results = exe.map(slow_fn, [1, 2, 3, 4])
        elapsed = time.monotonic() - start

        assert results == [1, 2, 3, 4]
        # 4 tasks × 0.1s each; parallel should finish in ~0.1-0.2s, not 0.4s
        assert elapsed < 0.35, f"Tasks took {elapsed:.2f}s — not parallel enough"
        exe.shutdown()
