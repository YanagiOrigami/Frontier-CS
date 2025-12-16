import textwrap

class Solution:
    def solve(self, spec_path: str = None):
        code = textwrap.dedent("""
        from sky_spot.strategies.multi_strategy import MultiRegionStrategy
        from sky_spot.utils import ClusterType

        class SafeScanStrategy(MultiRegionStrategy):
            NAME = "safe_scan_threshold"

            def __init__(self, args=None):
                super().__init__(args)
                self.num_regions = 9
                self.scan_attempts_since_last_spot = 0
                self.force_on_demand = False
                self.region_scan_limit = 3
                self.behind_schedule_slack_seconds = 2 * 3600  # 2 hours

            def _is_behind_schedule(self) -> bool:
                elapsed = self.env.elapsed_seconds
                done_work = sum(self.task_done_time)
                remaining_work = self.task_duration - done_work
                remaining_time = self.deadline - elapsed
                slack = remaining_time - remaining_work
                return slack <= self.behind_schedule_slack_seconds

            def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
                # If already committed to on-demand or behind schedule, stay on-demand
                if self.force_on_demand or self._is_behind_schedule():
                    self.force_on_demand = True
                    return ClusterType.ON_DEMAND

                # Prefer spot if available
                if has_spot:
                    self.scan_attempts_since_last_spot = 0
                    return ClusterType.SPOT

                # Spot not available in current region
                self.scan_attempts_since_last_spot += 1
                if self.scan_attempts_since_last_spot >= self.region_scan_limit:
                    self.force_on_demand = True
                    return ClusterType.ON_DEMAND

                # Try another region
                next_region = (self.env.get_current_region() + 1) % self.num_regions
                self.env.switch_region(next_region)
                return ClusterType.NONE

            @classmethod
            def _from_args(cls, parser):
                args, _ = parser.parse_known_args()
                return cls(args)
        """)
        return {"code": code}
