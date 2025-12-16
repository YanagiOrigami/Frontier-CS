import textwrap

class Solution:
    def solve(self, spec_path: str = None):
        code = textwrap.dedent("""
            from sky_spot.strategies.strategy import Strategy
            from sky_spot.utils import ClusterType


            class RobustSpotStrategy(Strategy):
                NAME = "robust_spot_heuristic_v1"

                def __init__(self, args):
                    super().__init__(args)
                    # Whether we've decided to stick with on-demand until completion
                    self._commit_on_demand = False
                    # Consecutive timesteps without spot availability
                    self._miss_count = 0

                def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
                    # Remaining work (seconds)
                    remaining_work = self.task_duration - sum(self.task_done_time)

                    # If task already finished, stop all compute
                    if remaining_work <= 0:
                        return ClusterType.NONE

                    # Time left until hard deadline (seconds)
                    time_left = self.deadline - self.env.elapsed_seconds
                    # Slack time (seconds)
                    slack = time_left - remaining_work

                    # If we've already committed to on-demand, keep using it
                    if self._commit_on_demand:
                        return ClusterType.ON_DEMAND

                    # Decide whether to commit to on-demand
                    # Dynamic threshold based on gap size and fixed cushion
                    commit_threshold = max(2 * 3600, 5 * self.env.gap_seconds)  # ≥2h or 5 gaps
                    # Also compare slack to a fraction of remaining work
                    if slack <= max(commit_threshold, 0.4 * remaining_work):
                        self._commit_on_demand = True
                        return ClusterType.ON_DEMAND

                    # If spot is available, use it
                    if has_spot:
                        self._miss_count = 0
                        return ClusterType.SPOT

                    # Spot unavailable – decide to wait or switch to on-demand
                    self._miss_count += 1
                    wait_steps_allowed = 2  # how many gaps we're willing to wait

                    # If we still have healthy slack, we can wait a bit
                    projected_slack = slack - self._miss_count * self.env.gap_seconds
                    if self._miss_count <= wait_steps_allowed and projected_slack > commit_threshold:
                        return ClusterType.NONE

                    # Otherwise, switch to on-demand for safety
                    self._commit_on_demand = True
                    return ClusterType.ON_DEMAND

                @classmethod
                def _from_args(cls, parser):
                    args, _ = parser.parse_known_args()
                    return cls(args)
        """)
        return code
