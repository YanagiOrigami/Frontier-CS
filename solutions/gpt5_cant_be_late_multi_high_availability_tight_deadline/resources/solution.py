import sys
from typing import Union, Dict

class Solution:
    def solve(self, spec_path: str = None) -> Union[str, Dict[str, str]]:
        code = '''
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

class CantBeLateSafeV2(MultiRegionStrategy):
    NAME = "cant_be_late_safe_v2"

    def __init__(self, args=None):
        super().__init__(args)
        self._force_od = False
        self._progress_sum = 0.0
        self._last_seen_segments = 0
        self._no_spot_streak = 0
        # Optional args hook for scenarios that pass custom region counts
        self._regions_count = getattr(args, "regions_count", 9) if args is not None else 9

    # ----------------- Helpers -----------------
    def _update_progress(self):
        segments = getattr(self, "task_done_time", None)
        if segments is None:
            return
        n = len(segments)
        if n > self._last_seen_segments:
            # Incremental sum to keep O(1) amortized per step
            self._progress_sum += sum(segments[self._last_seen_segments:n])
            self._last_seen_segments = n

    def _ro(self):
        ro = getattr(self, "restart_overhead", None)
        if ro is None:
            ro = getattr(self.env, "restart_overhead", 0.0)
        return float(ro)

    def _gap(self):
        g = getattr(self.env, "gap_seconds", 60.0)
        return float(g)

    def _work_remaining(self):
        return max(0.0, float(self.task_duration) - float(self._progress_sum))

    def _time_remaining(self):
        return float(self.deadline) - float(self.env.elapsed_seconds)

    def _required_time_on_demand(self, last_cluster_type):
        extra = 0.0
        if last_cluster_type != ClusterType.ON_DEMAND:
            extra = self._ro()
        return self._work_remaining() + extra

    def _slack_time(self, last_cluster_type):
        return self._time_remaining() - self._required_time_on_demand(last_cluster_type)

    def _panic_margin(self):
        # Safety margin accounting for one restart overhead and a couple of steps granularity
        return 2.0 * self._ro() + 2.0 * self._gap()

    def _avoid_pause_margin(self):
        # Additional headroom to avoid pausing when near schedule pressure
        return self._panic_margin() + 2.0 * self._gap()

    def _consec_no_spot_limit(self):
        # After this many consecutive no-spot steps, fallback to on-demand
        return 3

    def _regions_total(self):
        # Default to 9 regions if unknown
        if isinstance(self._regions_count, int) and self._regions_count > 0:
            return self._regions_count
        return 9

    def _next_region(self):
        total = self._regions_total()
        cur = self.env.get_current_region()
        return (cur + 1) % total

    # ----------------- Required API -----------------
    def _is_behind_schedule(self) -> bool:
        self._update_progress()
        last_ct = getattr(self.env, "cluster_type", None)
        if last_ct is None:
            last_ct = ClusterType.NONE
        slack = self._slack_time(last_ct)
        return slack <= self._panic_margin()

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_progress()

        # If out of time, attempt guaranteed progress
        if self._time_remaining() <= 0:
            self._force_od = True
            return ClusterType.ON_DEMAND

        # Engage forced on-demand if behind
        if not self._force_od and self._is_behind_schedule():
            self._force_od = True

        if self._force_od:
            self._no_spot_streak = 0
            return ClusterType.ON_DEMAND

        if has_spot:
            self._no_spot_streak = 0
            return ClusterType.SPOT

        # Spot not available here this step
        self._no_spot_streak += 1
        slack = self._slack_time(last_cluster_type)

        # Use on-demand if slack getting thin or spot unavailable repeatedly
        if slack <= self._avoid_pause_margin() or self._no_spot_streak >= self._consec_no_spot_limit():
            return ClusterType.ON_DEMAND

        # Otherwise, rotate region and wait for spot
        try:
            self.env.switch_region(self._next_region())
        except Exception:
            pass
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
'''
        return {"code": code}
