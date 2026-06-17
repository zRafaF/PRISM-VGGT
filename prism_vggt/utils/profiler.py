import threading

import torch


class VRAMProfiler:
    """Tracks PyTorch and overall system VRAM usage.

    A background thread samples total GPU usage between ``start()`` and ``stop()`` so
    that transient spikes (e.g. nvblox's block-hash doubling, which briefly allocates
    a 2x buffer) are captured as a true peak rather than missed by a single end-of-
    window reading.
    """

    def __init__(self, device: int = 0, sample_hz: float = 50.0):
        self.device = device
        self._interval = 1.0 / max(sample_hz, 1.0)
        self._thread: threading.Thread | None = None
        self._stop_evt: threading.Event | None = None
        self._peak_used_bytes: int = 0
        self._lock = threading.Lock()

    def _sample_loop(self) -> None:
        assert self._stop_evt is not None
        while not self._stop_evt.is_set():
            try:
                free, total = torch.cuda.mem_get_info(self.device)
                used = total - free
                with self._lock:
                    if used > self._peak_used_bytes:
                        self._peak_used_bytes = used
            except Exception:  # pragma: no cover - driver hiccup
                pass
            self._stop_evt.wait(self._interval)

    def start(self) -> None:
        if not torch.cuda.is_available():
            return
        torch.cuda.reset_peak_memory_stats(self.device)
        with self._lock:
            self._peak_used_bytes = 0
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, float, float, float, float]:
        """Stop sampling and return GB:
        (pytorch_alloc_peak, pytorch_reserved_peak, system_used_now,
         system_total, system_used_peak).
        """
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0, 0.0, 0.0

        if self._stop_evt is not None:
            self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

        gib = 1024 ** 3
        pt_alloc = torch.cuda.max_memory_allocated(self.device) / gib
        pt_res = torch.cuda.max_memory_reserved(self.device) / gib
        free, total = torch.cuda.mem_get_info(self.device)
        sys_used = (total - free) / gib
        sys_total = total / gib
        with self._lock:
            sys_peak = max(self._peak_used_bytes, total - free) / gib
        return pt_alloc, pt_res, sys_used, sys_total, sys_peak
