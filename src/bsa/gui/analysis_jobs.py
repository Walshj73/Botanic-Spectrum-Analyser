"""Main-thread queue dispatcher for a background hyperspectral analysis."""

from __future__ import annotations

import queue
import threading
from typing import Callable

from bsa.analysis.background import AnalysisCancelled, AnalysisSnapshot, run_analysis


class AnalysisJobController:
    def __init__(self, root, runner=run_analysis):
        self.root = root
        self.runner = runner
        self.active: dict | None = None
        self.closing = False

    def start(
        self,
        snapshot: AnalysisSnapshot,
        on_event: Callable[[str, object], None],
        on_finish: Callable[[str, object], None],
    ) -> bool:
        if self.closing or self.active is not None:
            return False
        job = {
            "cancelled": threading.Event(),
            "events": queue.Queue(),
            "after_id": None,
            "thread": None,
        }
        self.active = job
        runner = self.runner

        def worker():
            try:
                result = runner(
                    snapshot, job["cancelled"],
                    lambda event, value: job["events"].put((event, value)),
                )
                outcome = ("done", result)
            except AnalysisCancelled:
                outcome = ("cancelled", None)
            except Exception as error:
                outcome = ("error", error)
            job["events"].put(outcome)

        job["thread"] = threading.Thread(target=worker, name="BSA analyser", daemon=True)
        job["thread"].start()
        job["after_id"] = self.root.after(30, lambda: self._poll(job, on_event, on_finish))
        return True

    def _poll(self, job, on_event, on_finish) -> None:
        if self.closing or self.active is not job:
            return
        while True:
            try:
                event, value = job["events"].get_nowait()
            except queue.Empty:
                break
            if event in ("done", "cancelled", "error"):
                self.active = None
                on_finish(event, value)
                return
            on_event(event, value)
        job["after_id"] = self.root.after(30, lambda: self._poll(job, on_event, on_finish))

    def cancel(self) -> None:
        if self.active is not None:
            self.active["cancelled"].set()

    def close(self) -> None:
        self.closing = True
        self.cancel()
        if self.active is not None and self.active["after_id"] is not None:
            self.root.after_cancel(self.active["after_id"])
        self.active = None
