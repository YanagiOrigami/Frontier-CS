from typing import Union
import argparse

class Solution:
    def solve(self, spec_path: str = None) -> Union[str, dict]:
        strategy_code = '''
import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class SlackAwareStrategy(Strategy):
    NAME = "slack_aware_v1"
    RESERVED_SLACK_HOURS = 1.5   # Threshold to switch to On-Demand
    WAIT_SLACK_HOURS = 2.5       # Threshold to wait when Spot is unavailable

    def __init__(self, args):
        super().__init__(args)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate remaining work (seconds)
        remaining_work = self.task_duration - sum(self.task_done_time)
        if remaining_work <= 0:
            return ClusterType.NONE  # Task complete

        # Remaining wall-clock time until deadline (seconds)
        remaining_time = self.deadline - self.env.elapsed_seconds
        if remaining_time <= 0:
            return ClusterType.ON_DEMAND  # Deadline passed; finish ASAP

        # Slack time (hours)
        slack_hours = (remaining_time - remaining_work) / 3600.0

        # Not enough slack: use On-Demand
        if slack_hours <= self.RESERVED_SLACK_HOURS:
            return ClusterType.ON_DEMAND

        # Enough slack and Spot available: use Spot
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable: decide to wait or use On-Demand
        if slack_hours >= self.WAIT_SLACK_HOURS:
            return ClusterType.NONE  # Wait for Spot to return
        return ClusterType.ON_DEMAND  # Need progress

    @classmethod
    def _from_args(cls, parser):
        if parser is None:
            parser = argparse.ArgumentParser()
        args, _ = parser.parse_known_args()
        return cls(args)
'''
        return {"code": strategy_code}
