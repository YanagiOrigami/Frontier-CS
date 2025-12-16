import argparse
from typing import Union


class Solution:
    def solve(self, spec_path: str = None) -> Union[str, dict]:
        code = '''
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class CantBeLateThresholdV6(Strategy):
    NAME = "cant_be_late_threshold_v6"

    def __init__(self, args):
        super().__init__(args)
        # Tunable safety parameters
        self._safety_overhead_mult = getattr(args, "safety_overhead_mult", 2.0)
        self._safety_gap_mult = getattr(args, "safety_gap_mult", 1.5)
        # Internal state
        self._od_committed = False
        self._done_sum_cache = 0.0
        self._done_len_cache = 0

    def _update_done_sum(self):
        cur_len = len(self.task_done_time)
        if cur_len == self._done_len_cache:
            return self._done_sum_cache
        if cur_len > self._done_len_cache:
            # Add only the new segments
            self._done_sum_cache += sum(self.task_done_time[self._done_len_cache:cur_len])
            self._done_len_cache = cur_len
        else:
            # Fallback (should not happen), recompute from scratch
            self._done_sum_cache = sum(self.task_done_time)
            self._done_len_cache = cur_len
        return self._done_sum_cache

    def _safety_time(self):
        g = self.env.gap_seconds
        H = self.restart_overhead
        # Conservative buffer: account for potential double-count of overhead and discretization
        return self._safety_overhead_mult * H + self._safety_gap_mult * g

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If we've already finished, do nothing
        done = self._update_done_sum()
        remaining_work = self.task_duration - done
        if remaining_work <= 0:
            self._od_committed = True
            return ClusterType.NONE

        # If we've already committed to on-demand, stay there to avoid extra overhead switches
        if self._od_committed or last_cluster_type == ClusterType.ON_DEMAND:
            self._od_committed = True
            return ClusterType.ON_DEMAND

        elapsed = self.env.elapsed_seconds
        time_remaining = self.deadline - elapsed
        g = self.env.gap_seconds
        safety = self._safety_time()

        # If we are close enough to the deadline that we must guarantee completion via OD, commit now
        if time_remaining <= remaining_work + safety:
            self._od_committed = True
            return ClusterType.ON_DEMAND

        # Prefer spot when available while we have sufficient slack
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable: decide whether we can afford to wait one step
        # If after waiting one step we still have time to finish via OD with safety, we wait
        if (time_remaining - g) >= (remaining_work + safety):
            return ClusterType.NONE

        # Otherwise, we must switch to OD now to guarantee completion
        self._od_committed = True
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        # Allow tuning via CLI, but use safe defaults
        parser.add_argument("--safety_overhead_mult", type=float, default=2.0)
        parser.add_argument("--safety_gap_mult", type=float, default=1.5)
        args, _ = parser.parse_known_args()
        return cls(args)
'''
        return {"code": code}
