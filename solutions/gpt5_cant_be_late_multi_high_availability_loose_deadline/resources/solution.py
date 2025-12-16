import argparse
from typing import Union, Dict


class Solution:
    def solve(self, spec_path: str = None) -> Union[str, Dict[str, str]]:
        code = r'''
from argparse import ArgumentParser
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class CantBeLateBufferedSpot(MultiRegionStrategy):
    NAME = "cant_be_late_buffered_spot_v1"

    def __init__(self, args=None):
        super().__init__(args)
        # Strategy tunables (can be overridden by CLI args)
        self.buffer_steps = getattr(args, "buffer_steps", 2)
        self.buffer_overhead_mult = getattr(args, "buffer_overhead_mult", 2.0)
        self.lock_on_od = getattr(args, "lock_on_od", True)
        self.panic_lookahead_steps = getattr(args, "panic_lookahead_steps", 1)

        # Internal state
        self._od_lock = False
        self._cached_done_sum = 0.0
        self._cached_done_len = 0

    # -------- Utility getters to be robust to attribute placement --------
    def _get_attr(self, name, default=None):
        if hasattr(self, name):
            return getattr(self, name)
        if hasattr(self.env, name):
            return getattr(self.env, name)
        return default

    def _gap(self):
        g = self._get_attr("gap_seconds", 3600.0)
        return float(g if g is not None else 3600.0)

    def _restart_overhead(self):
        ro = self._get_attr("restart_overhead", 0.0)
        return float(ro if ro is not None else 0.0)

    def _deadline(self):
        d = self._get_attr("deadline", 0.0)
        return float(d if d is not None else 0.0)

    def _task_duration(self):
        td = self._get_attr("task_duration", 0.0)
        return float(td if td is not None else 0.0)

    def _task_done_list(self):
        lst = getattr(self, "task_done_time", None)
        if lst is None:
            lst = getattr(self.env, "task_done_time", [])
        return lst if lst is not None else []

    def _work_done(self):
        # Incremental sum to avoid O(n^2) over many steps
        lst = self._task_done_list()
        n = len(lst)
        if n > self._cached_done_len:
            # Sum only newly appended segments
            new_sum = 0.0
            for i in range(self._cached_done_len, n):
                new_sum += float(lst[i])
            self._cached_done_sum += new_sum
            self._cached_done_len = n
        return self._cached_done_sum

    def _buffer_time(self):
        # Dynamic buffer to protect against last-minute risks.
        # Combines a few timesteps of slack plus some multiple of restart overhead.
        gap = self._gap()
        ro = self._restart_overhead()
        base = self.buffer_steps * gap
        overhead_guard = self.buffer_overhead_mult * ro
        # Ensure at least one timestep + overhead worth of buffer
        min_guard = (1.0 * gap) + ro
        return max(base + overhead_guard, min_guard)

    def _time_left(self):
        elapsed = float(self._get_attr("elapsed_seconds", 0.0))
        return self._deadline() - elapsed

    def _remaining_work(self):
        return max(self._task_duration() - self._work_done(), 0.0)

    def _od_overhead_if_switch_now(self, last_cluster_type=None):
        if last_cluster_type is None:
            last_cluster_type = getattr(self.env, "cluster_type", None)
        ro = self._restart_overhead()
        return 0.0 if last_cluster_type == ClusterType.ON_DEMAND else ro

    # --------- Required API methods ----------
    def _is_behind_schedule(self) -> bool:
        # Determine if slack to switch to on-demand is below a protective buffer.
        time_left = self._time_left()
        remain = self._remaining_work()
        last_ct = getattr(self.env, "cluster_type", None)
        od_overhead = self._od_overhead_if_switch_now(last_ct)
        slack_if_switch_now = time_left - (remain + od_overhead)
        return slack_if_switch_now <= self._buffer_time()

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If we have already committed to OD, stick with it to avoid extra overhead.
        if self._od_lock or (self.lock_on_od and last_cluster_type == ClusterType.ON_DEMAND):
            self._od_lock = True
            return ClusterType.ON_DEMAND

        gap = self._gap()
        ro = self._restart_overhead()
        remain = self._remaining_work()
        time_left = self._time_left()

        # If already done, do nothing
        if remain <= 0.0:
            return ClusterType.NONE

        # If SPOT is currently unavailable, decide to wait or pivot to OD.
        if not has_spot:
            # Panic lookahead: if waiting just one step would make OD completion impossible, pivot now.
            od_overhead = ro if last_cluster_type != ClusterType.ON_DEMAND else 0.0
            lookahead = self.panic_lookahead_steps * gap
            if (time_left - lookahead) <= (remain + od_overhead):
                self._od_lock = True
                return ClusterType.ON_DEMAND
            # Otherwise, wait for SPOT to reappear
            return ClusterType.NONE

        # SPOT is available now. If we're behind schedule (not enough buffer), pivot to OD.
        if self._is_behind_schedule():
            self._od_lock = True
            return ClusterType.ON_DEMAND

        # Otherwise, use SPOT to save cost.
        return ClusterType.SPOT

    @classmethod
    def _from_args(cls, parser):
        # Optional tunables
        if not any(a.option_strings == ["--buffer-steps"] for a in parser._actions):
            parser.add_argument("--buffer-steps", type=int, default=2, dest="buffer_steps")
        if not any(a.option_strings == ["--buffer-overhead-mult"] for a in parser._actions):
            parser.add_argument("--buffer-overhead-mult", type=float, default=2.0, dest="buffer_overhead_mult")
        if not any(a.option_strings == ["--lock-on-od"] for a in parser._actions):
            parser.add_argument("--lock-on-od", action="store_true", default=True, dest="lock_on_od")
        if not any(a.option_strings == ["--panic-lookahead-steps"] for a in parser._actions):
            parser.add_argument("--panic-lookahead-steps", type=int, default=1, dest="panic_lookahead_steps")
        args, _ = parser.parse_known_args()
        return cls(args)
'''
        return {"code": code}
