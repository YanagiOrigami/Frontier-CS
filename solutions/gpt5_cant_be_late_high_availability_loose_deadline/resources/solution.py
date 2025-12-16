import textwrap

class Solution:
    def solve(self, spec_path: str = None) -> str | dict:
        code = textwrap.dedent("""
            from sky_spot.strategies.strategy import Strategy
            from sky_spot.utils import ClusterType

            class CantBeLateGuardrailStrategy(Strategy):
                NAME = "cant_be_late_guardrail_v1"

                def __init__(self, args):
                    super().__init__(args)
                    self._committed_to_od = False
                    self._eps = 1e-9

                def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
                    # If already finished, do nothing.
                    remaining_work = self.task_duration - sum(self.task_done_time)
                    if remaining_work <= 0:
                        return ClusterType.NONE

                    # If already on-demand (committed), keep running on-demand.
                    if self.env.cluster_type == ClusterType.ON_DEMAND or self._committed_to_od:
                        self._committed_to_od = True
                        return ClusterType.ON_DEMAND

                    # Time and scheduling parameters
                    gap = self.env.gap_seconds
                    time_remaining = self.deadline - self.env.elapsed_seconds

                    # If time is already past deadline, best effort is to use OD.
                    if time_remaining <= 0:
                        self._committed_to_od = True
                        return ClusterType.ON_DEMAND

                    # Overhead when switching to ON_DEMAND from non-OD
                    overhead_to_start_od = self.restart_overhead

                    # Guardrail policy:
                    # Safe to wait one more step without progress iff:
                    #   time_remaining - gap >= remaining_work + overhead_to_start_od
                    # If not safe, commit to ON_DEMAND now.
                    safe_to_wait = (time_remaining - gap) >= (remaining_work + overhead_to_start_od - self._eps)

                    if not safe_to_wait:
                        # Commit to on-demand now to guarantee completion
                        self._committed_to_od = True
                        return ClusterType.ON_DEMAND

                    # Otherwise, we can safely keep waiting for spot.
                    if has_spot:
                        return ClusterType.SPOT
                    else:
                        return ClusterType.NONE

                @classmethod
                def _from_args(cls, parser):
                    args, _ = parser.parse_known_args()
                    return cls(args)
        """).strip()
        return {"code": code}
