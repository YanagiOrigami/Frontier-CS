import argparse

class Solution:
    def solve(self, spec_path: str = None):
        strategy_code = r"""
import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class CheapThenSafeStrategy(Strategy):
    NAME = "cheap_then_safe_v1"

    def __init__(self, args):
        super().__init__(args)
        self._init_done = False
        # configurable parameters
        self.wait_factor = 2           # steps to wait for spot before falling back
        self.risk_multiplier = 3       # multiplier to derive risk threshold

    # internal helper to lazily initialize params that depend on env
    def _lazy_init(self):
        if self._init_done:
            return
        self.gap = getattr(self.env, "gap_seconds", 3600.0)
        self.risk_time = max(self.risk_multiplier * self.gap,
                             self.risk_multiplier * self.restart_overhead)
        self._init_done = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()

        # remaining work in seconds
        work_done = sum(self.task_done_time)
        remaining_task = self.task_duration - work_done

        # finished all work
        if remaining_task <= 0:
            return ClusterType.NONE

        # time left until deadline
        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed

        # if no time left, run on-demand to salvage
        if remaining_time <= 0:
            return ClusterType.ON_DEMAND

        # slack after reserving time for required work and a possible restart
        safe_slack = remaining_time - remaining_task - self.restart_overhead

        # if slack is too small, secure on-demand immediately
        if safe_slack <= self.risk_time:
            return ClusterType.ON_DEMAND

        # plenty of slack: prefer spot if available
        if has_spot:
            return ClusterType.SPOT

        # spot unavailable: decide whether to wait or switch
        if safe_slack > self.wait_factor * self.gap:
            return ClusterType.NONE
        else:
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
"""
        return {"code": strategy_code}
