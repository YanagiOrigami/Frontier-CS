import argparse

class Solution:
    def solve(self, spec_path: str = None):
        code = r'''
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

class SimpleAdaptiveStrategy(MultiRegionStrategy):
    NAME = "simple_adaptive_strategy_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self._stay_on_demand = False
        # Margin before we consider ourselves behind schedule (3 time steps)
        self._margin_steps = 3

    def _is_behind_schedule(self) -> bool:
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        time_left = self.deadline - self.env.elapsed_seconds

        safety_margin = self._margin_steps * self.env.gap_seconds
        return (time_left - remaining_work) < safety_margin

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If we've switched to on-demand, stay there until completion
        if self._stay_on_demand:
            return ClusterType.ON_DEMAND

        if has_spot:
            # Prefer spot when available
            return ClusterType.SPOT

        # Spot not available
        if self._is_behind_schedule():
            # Fall back to on-demand to guarantee completion
            self._stay_on_demand = True
            return ClusterType.ON_DEMAND

        # Otherwise, pause and wait for spot to return
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
'''
        return {"code": code}
