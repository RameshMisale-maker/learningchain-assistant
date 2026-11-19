"""Background ops loop: alert checks every ALERT_CHECK_MINUTES (ALERTS_ENABLED) and the nightly retrieval eval at
NIGHTLY_EVAL_HOUR UTC (NIGHTLY_EVAL_ENABLED). One daemon thread per API process; nothing runs when both are off."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from config.settings import get_settings

logger = logging.getLogger(__name__)
_thread: "OpsScheduler | None" = None


class OpsScheduler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="ops-scheduler", daemon=True)
        self._stop_event = threading.Event()
        self.last_alert_check = 0.0
        self.last_eval_day = ""

    def stop(self) -> None:
        self._stop_event.set()

    def tick(self) -> None:
        s = get_settings()
        if s.alerts_enabled and time.monotonic() - self.last_alert_check >= s.alert_check_minutes * 60:
            self.last_alert_check = time.monotonic()
            from backend.observability.alerts import evaluate_alerts

            evaluate_alerts()
        now = datetime.now(timezone.utc)
        day = now.strftime("%Y-%m-%d")
        if s.nightly_eval_enabled and now.hour == s.nightly_eval_hour and self.last_eval_day != day:
            self.last_eval_day = day
            from backend.observability.alerts import nightly_eval

            out = nightly_eval()
            logger.info("Nightly retrieval eval saved to %s", out.get("file"))

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("Ops scheduler error")
            self._stop_event.wait(30)


def start() -> OpsScheduler | None:
    global _thread
    s = get_settings()
    if not (s.alerts_enabled or s.nightly_eval_enabled):
        return None
    if _thread is None or not _thread.is_alive():
        _thread = OpsScheduler()
        _thread.start()
    return _thread


def stop() -> None:
    if _thread is not None:
        _thread.stop()
        _thread.join(timeout=5)
