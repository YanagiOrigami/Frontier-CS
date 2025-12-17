"""
Example solution for cant-be-late problem.

Solution interface:
    class Solution(Strategy):
        def solve(self, spec_path: str) -> "Solution":
            # Optional: read spec for configuration
            return self

        def _step(self, last_cluster_type, has_spot) -> ClusterType:
            # Decision logic at each simulation step
            ...
"""
import argparse
import math

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    """Greedy strategy: stay on spot until deadline pressure dictates on-demand."""

    NAME = "greedy_safety"

    def solve(self, spec_path: str) -> "Solution":
        """Initialize the solution. Can read spec for configuration if needed."""
        # For this simple strategy, no configuration needed
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Make decision at each simulation step."""
        env = self.env
        gap = env.gap_seconds

        work_left = self.task_duration - sum(self.task_done_time)
        if work_left <= 1e-9:
            return ClusterType.NONE

        left_ticks = max(0, math.floor((self.deadline - env.elapsed_seconds) / gap))
        need1d = math.ceil((work_left + self.restart_overhead) / gap)
        need2d = math.ceil((work_left + 2 * self.restart_overhead) / gap)

        # Must switch to on-demand if we can't afford any more preemptions
        if need1d >= left_ticks:
            return ClusterType.ON_DEMAND

        # Should be cautious if we can only afford one more preemption
        if need2d >= left_ticks:
            if env.cluster_type == ClusterType.SPOT and has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Otherwise, prefer spot if available
        return ClusterType.SPOT if has_spot else ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser) -> "Solution":
        parser.add_argument_group("Solution")
        args, _ = parser.parse_known_args()
        return cls(args)
