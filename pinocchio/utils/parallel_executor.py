"""Parallel Executor — resource-aware concurrent task runner.

Provides a high-level API for running multiple I/O-bound tasks (LLM calls,
multimodal processing) in parallel while respecting the detected hardware
limits from ``ResourceMonitor``.

Key features:
- Dynamically sizes the thread pool based on GPU / RAM availability
- Supports both ``map`` (homogeneous) and ``submit`` (heterogeneous) patterns
- Graceful degradation: falls back to sequential execution on single-core
- Progress logging via PinocchioLogger
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, TypeVar

from pinocchio.utils.resource_monitor import ResourceMonitor, ResourceSnapshot

T = TypeVar("T")


class ParallelExecutor:
    """Resource-aware parallel task executor.

    Usage
    -----
    >>> executor = ParallelExecutor()  # auto-detects resources
    >>> results = executor.map(process_fn, items)
    >>> # or heterogeneous tasks:
    >>> futures = [executor.submit(fn, arg) for fn, arg in tasks]
    >>> results = executor.gather(futures)
    """

    def __init__(
        self,
        max_workers: int | None = None,
        resource_monitor: ResourceMonitor | None = None,
    ) -> None:
        self._monitor = resource_monitor or ResourceMonitor()
        self._snapshot = self._monitor.snapshot()
        self._max_workers = max_workers or self._snapshot.recommended_workers
        self._pool: ThreadPoolExecutor | None = None

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def resources(self) -> ResourceSnapshot:
        return self._snapshot

    def refresh_resources(self) -> ResourceSnapshot:
        """Re-detect resources and resize the worker pool."""
        self._snapshot = self._monitor.snapshot(refresh=True)
        self._max_workers = self._snapshot.recommended_workers
        # Recreate pool on next use
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        return self._snapshot

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def _get_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="pinocchio-worker",
            )
        return self._pool

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the thread pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=wait)
            self._pool = None

    # ------------------------------------------------------------------
    # Execution API
    # ------------------------------------------------------------------

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        """Submit a single task to the pool."""
        return self._get_pool().submit(fn, *args, **kwargs)

    def map(
        self,
        fn: Callable[..., T],
        items: Iterable[Any],
        *,
        timeout: float | None = None,
    ) -> list[T]:
        """Apply *fn* to each item in parallel, return results in order.

        Falls back to sequential execution when ``max_workers == 1``.
        """
        item_list = list(items)
        if not item_list:
            return []

        # Sequential fast-path
        if self._max_workers <= 1 or len(item_list) == 1:
            return [fn(item) for item in item_list]

        pool = self._get_pool()
        futures = {pool.submit(fn, item): idx for idx, item in enumerate(item_list)}
        results: list[Any] = [None] * len(item_list)

        for future in as_completed(futures, timeout=timeout):
            idx = futures[future]
            results[idx] = future.result()

        return results

    def run_parallel(
        self,
        tasks: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]],
        *,
        timeout: float | None = None,
    ) -> list[Any]:
        """Run a list of heterogeneous ``(fn, args, kwargs)`` tasks in parallel.

        Returns results in the same order as the input tasks.
        """
        if not tasks:
            return []

        if self._max_workers <= 1 or len(tasks) == 1:
            return [fn(*args, **kw) for fn, args, kw in tasks]

        pool = self._get_pool()
        futures = {
            pool.submit(fn, *args, **kw): idx
            for idx, (fn, args, kw) in enumerate(tasks)
        }
        results: list[Any] = [None] * len(tasks)

        for future in as_completed(futures, timeout=timeout):
            idx = futures[future]
            results[idx] = future.result()

        return results

    def gather(self, futures: list[Future[T]], *, timeout: float | None = None) -> list[T]:
        """Wait for all futures and return results in order."""
        return [f.result(timeout=timeout) for f in futures]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ParallelExecutor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown(wait=True)
