import os

class Solution:
    def solve(self, spec_path: str = None):
        code = '''
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class SafeWaitODFallback(Strategy):
    NAME = "safe_wait_od_fallback_v1"

    def __init__(self, args):
        super().__init__(args)
        self.lock_to_od = False
        self.progress = 0.0
        self._seen_done_len = 0
        self._ct_none = None

    def _get_ct_none(self):
        if self._ct_none is None:
            self._ct_none = getattr(ClusterType, "NONE", getattr(ClusterType, "None"))
        return self._ct_none

    def _update_progress(self):
        l = len(self.task_done_time)
        if l > self._seen_done_len:
            inc = 0.0
            for i in range(self._seen_done_len, l):
                v = self.task_done_time[i]
                if v:
                    inc += v
            self.progress += inc
            self._seen_done_len = l

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update internal progress efficiently
        self._update_progress()

        # If finished, avoid any further cost
        remaining = self.task_duration - self.progress
        if remaining <= 0.0:
            return self._get_ct_none()

        # If we already committed to On-Demand, stick with it
        if self.lock_to_od:
            return ClusterType.ON_DEMAND

        # Time accounting
        t = float(self.env.elapsed_seconds)
        L = float(self.deadline) - t  # time left to deadline
        g = float(self.env.gap_seconds)
        # Small safety fudge to avoid edge effects (clip to <= 60s)
        fudge = min(60.0, 2.0 * g)

        # Cost of switching to On-Demand now (conservative: full restart overhead if not already OD)
        need_overhead = self.restart_overhead if (self.env.cluster_type != ClusterType.ON_DEMAND) else 0.0

        # Slack if we were to switch to OD now and run to completion
        spare_time = L - (remaining + need_overhead + fudge)

        # If slack is depleted (or nearly so), commit to On-Demand to guarantee finish
        if spare_time <= 0.0 or (not has_spot and spare_time <= g):
            self.lock_to_od = True
            return ClusterType.ON_DEMAND

        # Otherwise, prefer Spot when available, else wait (NONE) to save cost
        if has_spot:
            return ClusterType.SPOT
        return self._get_ct_none()

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
'''
        return {"code": code}
