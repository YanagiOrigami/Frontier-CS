import textwrap

class Solution:
    def solve(self, spec_path: str = None):
        strategy_code = textwrap.dedent("""
            from sky_spot.strategies.multi_strategy import MultiRegionStrategy
            from sky_spot.utils import ClusterType
            import argparse

            class SimpleSpotStrategy(MultiRegionStrategy):
                NAME = "simple_spot_strategy_v1"

                def __init__(self, args=None):
                    super().__init__(args)

                def _is_behind_schedule(self) -> bool:
                    try:
                        progress = sum(self.task_done_time)
                    except AttributeError:
                        progress = getattr(self, "task_done", 0.0)
                    remaining_work = self.task_duration - progress
                    remaining_time = self.deadline - self.env.elapsed_seconds
                    if remaining_time <= 0:
                        return True
                    ratio_needed = remaining_work / remaining_time
                    slack = remaining_time - remaining_work
                    min_slack = 2 * self.env.gap_seconds
                    return ratio_needed >= 0.85 or slack <= min_slack

                def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
                    behind = self._is_behind_schedule()
                    if has_spot and not behind:
                        return ClusterType.SPOT
                    if behind:
                        return ClusterType.ON_DEMAND
                    return ClusterType.NONE

                @classmethod
                def _from_args(cls, parser):
                    args, _ = parser.parse_known_args()
                    return cls(args)
        """)
        return {"code": strategy_code}
