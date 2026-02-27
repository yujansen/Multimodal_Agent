"""Resource Monitor — dynamic detection and tracking of local compute resources.

Detects CPU, GPU (CUDA / Apple MPS / ROCm), RAM, and VRAM availability at
runtime so the orchestrator can make intelligent scheduling decisions:

- How many modality processors to run in parallel
- Whether to use GPU-accelerated inference
- When to throttle concurrency to avoid OOM
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GPUInfo:
    """Snapshot of a single GPU device."""

    name: str = "unknown"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    backend: str = "none"  # cuda | mps | rocm | none

    @property
    def utilisation_pct(self) -> float:
        if self.vram_total_mb == 0:
            return 0.0
        return (1 - self.vram_free_mb / self.vram_total_mb) * 100


@dataclass
class ResourceSnapshot:
    """A point-in-time snapshot of available compute resources."""

    cpu_count: int = 1
    cpu_count_physical: int = 1
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    gpus: list[GPUInfo] = field(default_factory=list)
    ollama_running: bool = False
    ollama_gpu_layers: int | None = None

    # -- derived helpers --------------------------------------------------

    @property
    def has_gpu(self) -> bool:
        return any(g.backend != "none" for g in self.gpus)

    @property
    def gpu_backend(self) -> str:
        """Return the primary GPU backend name."""
        for g in self.gpus:
            if g.backend != "none":
                return g.backend
        return "none"

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_total_mb for g in self.gpus)

    @property
    def free_vram_mb(self) -> int:
        return sum(g.vram_free_mb for g in self.gpus)

    @property
    def recommended_workers(self) -> int:
        """Recommend the number of parallel workers for I/O-bound LLM calls.

        Heuristic:
        - GPU present & ≥8 GB VRAM  → up to 4 concurrent requests
        - GPU present & <8 GB       → 2
        - CPU-only with ≥16 GB RAM  → 2
        - else                       → 1 (sequential)
        """
        if self.has_gpu:
            vram = self.total_vram_mb
            if vram >= 16_000:
                return min(4, self.cpu_count_physical)
            if vram >= 8_000:
                return min(4, self.cpu_count_physical)
            return 2
        if self.ram_available_mb >= 16_000:
            return min(2, self.cpu_count_physical)
        return 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_count": self.cpu_count,
            "cpu_count_physical": self.cpu_count_physical,
            "ram_total_mb": self.ram_total_mb,
            "ram_available_mb": self.ram_available_mb,
            "gpus": [
                {
                    "name": g.name,
                    "vram_total_mb": g.vram_total_mb,
                    "vram_free_mb": g.vram_free_mb,
                    "backend": g.backend,
                    "utilisation_pct": round(g.utilisation_pct, 1),
                }
                for g in self.gpus
            ],
            "has_gpu": self.has_gpu,
            "gpu_backend": self.gpu_backend,
            "recommended_workers": self.recommended_workers,
            "ollama_running": self.ollama_running,
        }


class ResourceMonitor:
    """Detect and periodically refresh local compute resource information."""

    def __init__(self) -> None:
        self._last: ResourceSnapshot | None = None

    def snapshot(self, *, refresh: bool = False) -> ResourceSnapshot:
        """Return the current resource snapshot; cache unless *refresh* is True."""
        if self._last is not None and not refresh:
            return self._last
        self._last = self._detect()
        return self._last

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect(self) -> ResourceSnapshot:
        snap = ResourceSnapshot(
            cpu_count=os.cpu_count() or 1,
            cpu_count_physical=self._physical_cores(),
        )
        self._detect_ram(snap)
        self._detect_gpu(snap)
        self._detect_ollama(snap)
        return snap

    # -- CPU / RAM -------------------------------------------------------

    @staticmethod
    def _physical_cores() -> int:
        try:
            import multiprocessing
            # On macOS, os.cpu_count() returns logical; try sysctl
            if platform.system() == "Darwin":
                out = subprocess.check_output(
                    ["sysctl", "-n", "hw.physicalcpu"], text=True,
                ).strip()
                return int(out)
            # Linux: look for physical cores in /proc/cpuinfo
            if platform.system() == "Linux":
                out = subprocess.check_output(
                    ["nproc", "--all"], text=True,
                ).strip()
                return int(out)
        except Exception:
            pass
        return os.cpu_count() or 1

    @staticmethod
    def _detect_ram(snap: ResourceSnapshot) -> None:
        """Fill RAM info using platform-specific methods (no psutil needed)."""
        system = platform.system()
        try:
            if system == "Darwin":
                # Total RAM via sysctl
                total = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True,
                ).strip()
                snap.ram_total_mb = int(total) // (1024 * 1024)
                # Available = use vm_stat
                vm = subprocess.check_output(["vm_stat"], text=True)
                free_pages = 0
                for line in vm.splitlines():
                    if "Pages free" in line or "Pages inactive" in line:
                        parts = line.split(":")
                        if len(parts) == 2:
                            free_pages += int(parts[1].strip().rstrip("."))
                snap.ram_available_mb = (free_pages * 4096) // (1024 * 1024)
            elif system == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            snap.ram_total_mb = int(line.split()[1]) // 1024
                        elif line.startswith("MemAvailable:"):
                            snap.ram_available_mb = int(line.split()[1]) // 1024
        except Exception:
            snap.ram_total_mb = 8_000  # conservative fallback
            snap.ram_available_mb = 4_000

    # -- GPU detection ---------------------------------------------------

    @staticmethod
    def _detect_gpu(snap: ResourceSnapshot) -> None:
        """Detect CUDA, Apple MPS, or ROCm GPUs."""
        # 1) Try nvidia-smi for CUDA GPUs
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total,memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        snap.gpus.append(
                            GPUInfo(
                                name=parts[0],
                                vram_total_mb=int(float(parts[1])),
                                vram_free_mb=int(float(parts[2])),
                                backend="cuda",
                            )
                        )
                if snap.gpus:
                    return
            except Exception:
                pass

        # 2) Apple MPS (Metal Performance Shaders) on macOS
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            # Apple Silicon — unified memory, report total RAM as VRAM
            gpu_name = "Apple Silicon (MPS)"
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                ).strip()
                gpu_name = f"{out} (MPS)"
            except Exception:
                pass
            snap.gpus.append(
                GPUInfo(
                    name=gpu_name,
                    vram_total_mb=snap.ram_total_mb,  # unified memory
                    vram_free_mb=snap.ram_available_mb,
                    backend="mps",
                )
            )
            return

        # 3) ROCm (AMD)
        if shutil.which("rocm-smi"):
            try:
                out = subprocess.check_output(
                    ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                    text=True,
                    timeout=5,
                )
                # Simplified parsing
                snap.gpus.append(
                    GPUInfo(name="AMD ROCm GPU", vram_total_mb=0, backend="rocm")
                )
            except Exception:
                pass

    # -- Ollama status ---------------------------------------------------

    @staticmethod
    def _detect_ollama(snap: ResourceSnapshot) -> None:
        """Check if Ollama is running and query GPU layer info."""
        if not shutil.which("ollama"):
            return
        try:
            out = subprocess.check_output(
                ["ollama", "ps"], text=True, timeout=5,
            )
            snap.ollama_running = bool(out.strip())
        except Exception:
            snap.ollama_running = False
