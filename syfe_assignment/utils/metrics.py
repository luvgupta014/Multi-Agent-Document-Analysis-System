import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class QueryMetric:
    question: str
    specialist_used: str
    retrieval_time: float
    synthesis_time: float
    total_time: float
    results_count: int
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:

    def __init__(self):
        self._queries: List[QueryMetric] = []
        self._start_time: Optional[float] = None

    def start_timer(self):
        self._start_time = time.time()
        return self._start_time

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def record_query(self, metric: QueryMetric):
        self._queries.append(metric)

    def get_summary(self) -> Dict:
        if not self._queries:
            return {"total_queries": 0}

        successful = [q for q in self._queries if q.success]
        failed = [q for q in self._queries if not q.success]

        avg_total = sum(q.total_time for q in successful) / len(successful) if successful else 0
        avg_retrieval = sum(q.retrieval_time for q in successful) / len(successful) if successful else 0
        avg_synthesis = sum(q.synthesis_time for q in successful) / len(successful) if successful else 0

        return {
            "total_queries": len(self._queries),
            "successful": len(successful),
            "failed": len(failed),
            "avg_total_time_ms": round(avg_total * 1000, 2),
            "avg_retrieval_time_ms": round(avg_retrieval * 1000, 2),
            "avg_synthesis_time_ms": round(avg_synthesis * 1000, 2),
            "success_rate": round(len(successful) / len(self._queries) * 100, 1)
        }

    def reset(self):
        self._queries.clear()
        self._start_time = None
