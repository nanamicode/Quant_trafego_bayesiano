from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any

import pandas as pd


@dataclass
class RunTelemetry:
    """In-process observability for one quantitative run."""

    started_perf: float = field(default_factory=time.perf_counter)
    started_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    events: list[dict[str, Any]] = field(default_factory=list)
    _stage_started: dict[str, float] = field(default_factory=dict)
    _stage_finished: dict[str, float] = field(default_factory=dict)

    def emit(
        self,
        stage: str,
        event: str,
        *,
        message: str = "",
        progress: float | None = None,
        **meta: Any,
    ) -> dict[str, Any]:
        now = time.perf_counter()
        elapsed = now - self.started_perf

        if event in {"start", "stage_start"} and stage not in self._stage_started:
            self._stage_started[stage] = now
        if stage not in self._stage_started:
            self._stage_started[stage] = now
        if event in {"done", "stage_done", "complete"}:
            self._stage_finished[stage] = now

        stage_elapsed = now - self._stage_started[stage]
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": float(elapsed),
            "stage": str(stage),
            "event": str(event),
            "message": str(message),
            "progress": (
                None if progress is None else float(progress)
            ),
            "stage_elapsed_seconds": float(stage_elapsed),
            **meta,
        }
        self.events.append(row)
        return row

    def finish_stage(
        self,
        stage: str,
        *,
        message: str = "",
        **meta: Any,
    ) -> dict[str, Any]:
        return self.emit(
            stage,
            "stage_done",
            message=message,
            progress=1.0,
            **meta,
        )

    @property
    def elapsed_seconds(self) -> float:
        return float(time.perf_counter() - self.started_perf)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)

    def stage_summary(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        stages = []
        for event in self.events:
            stage = event["stage"]
            if stage not in stages:
                stages.append(stage)

        now = time.perf_counter()
        for stage in stages:
            started = self._stage_started.get(stage)
            if started is None:
                continue
            finished = self._stage_finished.get(stage)
            duration = (finished or now) - started
            stage_events = [
                event
                for event in self.events
                if event["stage"] == stage
            ]
            last = stage_events[-1] if stage_events else {}
            rows.append(
                {
                    "stage": stage,
                    "duration_seconds": float(duration),
                    "finished": finished is not None,
                    "last_event": last.get("event"),
                    "last_message": last.get("message"),
                    "last_progress": last.get("progress"),
                    "event_count": len(stage_events),
                }
            )
        return pd.DataFrame(rows)

    def developer_snapshot(self) -> dict[str, Any]:
        summary = self.stage_summary()
        bottleneck = None
        if not summary.empty:
            idx = summary["duration_seconds"].idxmax()
            bottleneck = str(summary.loc[idx, "stage"])

        return {
            "started_utc": self.started_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "event_count": len(self.events),
            "bottleneck_stage_so_far": bottleneck,
            "stages": (
                summary.to_dict(orient="records")
                if not summary.empty
                else []
            ),
        }
