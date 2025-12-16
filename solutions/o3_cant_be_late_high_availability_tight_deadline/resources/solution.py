import textwrap

class Solution:
    def solve(self, spec_path: str = None):
        code = textwrap.dedent("""
            from sky_spot.strategies.strategy import Strategy
            from sky_spot.utils import ClusterType


            class DeadlineAwareStrategy(Strategy):
                NAME = "deadline_aware_v1"

                def __init__(self, args):
                    super().__init__(args)
                    self.locked_on_demand = False
                    self.accrued_done = 0.0
                    self.last_task_done_len = 0

                def _update_progress(self):
                    if len(self.task_done_time) > self.last_task_done_len:
                        new_segments = self.task_done_time[self.last_task_done_len:]
                        self.accrued_done += sum(new_segments)
                        self.last_task_done_len = len(self.task_done_time)

                def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
                    self._update_progress()

                    remaining_work = self.task_duration - self.accrued_done
                    if remaining_work <= 0:
                        return ClusterType.NONE

                    time_left = self.deadline - self.env.elapsed_seconds

                    # If previously locked to on-demand, stay there.
                    if self.locked_on_demand:
                        return ClusterType.ON_DEMAND

                    # Time needed to finish if we switch to OD now
                    time_needed_if_od_now = remaining_work
                    if last_cluster_type != ClusterType.ON_DEMAND:
                        time_needed_if_od_now += self.restart_overhead

                    # Safety margin equals at least one gap step or restart overhead
                    safety_margin = max(self.restart_overhead, self.env.gap_seconds)

                    # Enter critical window: must switch/lock to on-demand
                    if time_left <= time_needed_if_od_now + safety_margin:
                        self.locked_on_demand = True
                        return ClusterType.ON_DEMAND

                    # Prefer spot when available
                    if has_spot:
                        return ClusterType.SPOT

                    # Spot unavailable: decide between waiting and switching to OD
                    slack_time = time_left - (remaining_work + self.restart_overhead)
                    wait_threshold = max(self.restart_overhead, self.env.gap_seconds)

                    if slack_time > wait_threshold:
                        return ClusterType.NONE
                    else:
                        return ClusterType.ON_DEMAND

                @classmethod
                def _from_args(cls, parser):
                    args, _ = parser.parse_known_args()
                    return cls(args)
        """)
        return {"code": code}
