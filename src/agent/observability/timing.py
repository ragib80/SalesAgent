import time


class Timer:
    """Context manager for measuring elapsed wall-clock time in milliseconds."""

    def __init__(self) -> None:
        self._start: float | None = None
        self.elapsed_ms: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000  # type: ignore[operator]
