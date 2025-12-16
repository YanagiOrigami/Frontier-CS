class Solution:
    def solve(self, spec_path: str = None) -> str | dict:
        strategy_code = '''
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class SimpleSlackStrategy(MultiRegionStrategy):
    NAME = "simple_slack_strategy"

    def __init__(self, args=None):
        super().__init__(args)
        # Threshold for deciding if behind schedule (remaining work / remaining time)
        self._behind_thresh = 0.97

    def _is_behind_schedule(self) -> bool:
        # Calculate remaining work and time
        done = sum(getattr(self, "task_done_time", []))
        remaining_work = max(self.task_duration - done, 0.0)
        remaining_time = max(self.deadline - self.env.elapsed_seconds, 1e-9)
        return (remaining_work / remaining_time) > self._behind_thresh

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        done = sum(getattr(self, "task_done_time", []))
        if done >= self.task_duration:
            # Task finished
            return ClusterType.NONE

        behind = self._is_behind_schedule()

        # If we are behind schedule, prefer guaranteed progress with On-Demand
        if behind:
            return ClusterType.ON_DEMAND

        # If not behind schedule:
        if has_spot:
            # Spot is available and we have slack
            return ClusterType.SPOT
        else:
            # Spot not available. Decide to wait or switch to On-Demand.
            gap = getattr(self.env, "gap_seconds", 0)
            remaining_work = self.task_duration - done
            remaining_time_after_wait = max(self.deadline - (self.env.elapsed_seconds + gap), 1e-9)
            # Check if waiting one gap keeps us on schedule
            if (remaining_work / remaining_time_after_wait) <= self._behind_thresh:
                return ClusterType.NONE
            else:
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
'''
        return strategy_code
