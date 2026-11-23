import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from sme_bridge.models import KnowledgeGap

logger = logging.getLogger(__name__)


class GapLedger(ABC):
    @abstractmethod
    def record(self, gap: KnowledgeGap) -> None: ...

    @abstractmethod
    def get(self, gap_id: str) -> KnowledgeGap | None: ...

    @abstractmethod
    def list_open(self) -> list[KnowledgeGap]: ...

    @abstractmethod
    def list_all(self) -> list[KnowledgeGap]: ...

    @abstractmethod
    def mark_answered(self, gap_id: str, answer: str, by: str) -> None: ...

    @abstractmethod
    def mark_indexed(self, gap_id: str) -> None: ...

    @abstractmethod
    def statistics(self) -> dict: ...


class InMemoryLedger(GapLedger):
    def __init__(self):
        self.gaps: dict[str, KnowledgeGap] = {}

    def record(self, gap: KnowledgeGap) -> None:
        self.gaps[gap.gap_id] = gap

    def get(self, gap_id: str) -> KnowledgeGap | None:
        return self.gaps.get(gap_id)

    def list_open(self) -> list[KnowledgeGap]:
        return [g for g in self.gaps.values() if g.status == "open"]

    def list_all(self) -> list[KnowledgeGap]:
        return list(self.gaps.values())

    def mark_answered(self, gap_id: str, answer: str, by: str) -> None:
        gap = self.gaps.get(gap_id)
        if gap:
            gap.sme_answer = answer
            gap.answered_at = datetime.now()
            gap.answered_by = by
            gap.status = "answered"

    def mark_indexed(self, gap_id: str) -> None:
        gap = self.gaps.get(gap_id)
        if gap:
            gap.indexed_at = datetime.now()
            gap.status = "indexed"

    def statistics(self) -> dict:
        all_gaps = list(self.gaps.values())
        open_count = sum(1 for g in all_gaps if g.status == "open")
        answered = [g for g in all_gaps if g.answered_at]
        if answered:
            durations = [(g.answered_at - g.asked_at).total_seconds() for g in answered]
            mean_time = sum(durations) / len(durations)
        else:
            mean_time = 0.0
        resolution_rate = len(answered) / len(all_gaps) if all_gaps else 0.0
        return {
            "total": len(all_gaps),
            "open": open_count,
            "mean_time_to_answer_seconds": mean_time,
            "resolution_rate": resolution_rate,
        }


class JSONLLedger(GapLedger):
    def __init__(self, path: str):
        self.path = Path(path)
        self.lock = threading.Lock()
        self._cache: dict[str, KnowledgeGap] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                gap = KnowledgeGap(**json.loads(line))
                self._cache[gap.gap_id] = gap

    def _append(self, gap: KnowledgeGap) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(gap.model_dump(mode="json")) + "\n")
            self._cache[gap.gap_id] = gap

    def record(self, gap: KnowledgeGap) -> None:
        self._append(gap)

    def get(self, gap_id: str) -> KnowledgeGap | None:
        return self._cache.get(gap_id)

    def list_open(self) -> list[KnowledgeGap]:
        return [g for g in self._cache.values() if g.status == "open"]

    def list_all(self) -> list[KnowledgeGap]:
        return list(self._cache.values())

    def mark_answered(self, gap_id: str, answer: str, by: str) -> None:
        gap = self._cache.get(gap_id)
        if gap:
            gap.sme_answer = answer
            gap.answered_at = datetime.now()
            gap.answered_by = by
            gap.status = "answered"
            self._append(gap)

    def mark_indexed(self, gap_id: str) -> None:
        gap = self._cache.get(gap_id)
        if gap:
            gap.indexed_at = datetime.now()
            gap.status = "indexed"
            self._append(gap)

    def statistics(self) -> dict:
        all_gaps = list(self._cache.values())
        open_count = sum(1 for g in all_gaps if g.status == "open")
        answered = [g for g in all_gaps if g.answered_at]
        if answered:
            durations = [(g.answered_at - g.asked_at).total_seconds() for g in answered]
            mean_time = sum(durations) / len(durations)
        else:
            mean_time = 0.0
        resolution_rate = len(answered) / len(all_gaps) if all_gaps else 0.0
        return {
            "total": len(all_gaps),
            "open": open_count,
            "mean_time_to_answer_seconds": mean_time,
            "resolution_rate": resolution_rate,
        }


LEDGERS = {
    "memory": InMemoryLedger,
    "jsonl": JSONLLedger,
}


def get_ledger(name: str, **kwargs) -> GapLedger:
    cls = LEDGERS.get(name)
    if not cls:
        available = ", ".join(LEDGERS.keys())
        raise ValueError(f"Unknown ledger backend: {name}. Available: {available}")
    return cls(**kwargs)


__all__ = [
    "GapLedger",
    "InMemoryLedger",
    "JSONLLedger",
    "get_ledger",
]
