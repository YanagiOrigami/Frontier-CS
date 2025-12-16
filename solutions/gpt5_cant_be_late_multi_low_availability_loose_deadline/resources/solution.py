import sys
from typing import Any, Dict, Union

class Solution:
    def solve(self, spec_path: str = None) -> Union[str, Dict[str, str]]:
        code = '''
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class SlackConservativeV1(MultiRegionStrategy):
    NAME = "slack_conservative_v1"

    def __init__(self, args):
        super().__init__(args)
        # Efficient progress tracking to avoid O(n) summation each step
        self._progress_sum = 0.0
        self._progress_idx = 0

    def _get_progress(self) -> float:
        lst = getattr(self, "task_done_time", None)
        if not lst:
            return self._progress_sum
        n = len(lst)
        if n > self._progress_idx:
            # Accumulate only new entries
            for i in range(self._progress_idx, n):
                self._progress_sum += lst[i]
            self._progress_idx = n
        return self._progress_sum

    def _is_behind_schedule(self) -> bool:
        progress = self._get_progress()
        total = getattr(self, "task_duration", getattr(self.env, "task_duration", 0.0))
        work_left = max(0.0, total - progress)

        now = getattr(self.env, "elapsed_seconds", 0.0)
        deadline = getattr(self, "deadline", getattr(self.env, "deadline", 0.0))
        time_left = deadline - now

        gap = getattr(self.env, "gap_seconds", getattr(self, "gap_seconds", 3600.0))
        overhead = getattr(self, "restart_overhead", getattr(self.env, "restart_overhead", 0.0))

        # Safety buffer for discrete step rounding and one restart overhead.
        buffer_time = gap + overhead

        # If time left is insufficient to complete remaining work with buffer, we are behind.
        return time_left <= work_left + buffer_time

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Simple policy:
        # - If behind schedule: run on On-Demand to guarantee progress.
        # - Else if Spot is available: use Spot.
        # - Else: wait (NONE) to save cost; plenty of slack exists.
        if self._is_behind_schedule():
            return ClusterType.ON_DEMAND
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
'''
        return code
