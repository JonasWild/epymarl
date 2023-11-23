from custom.lbforaging.behaviours.BalanceLevelRewardAdaption import BalanceLevelRewardAdaption
from custom.lbforaging.behaviours.CoordinatedPairMovement import CoordinatedPairMovement
from custom.lbforaging.behaviours.RewardClosestFruit import RewardClosestFruit
from custom.lbforaging.behaviours.SplitInGroups import SplitInGroups

BEHAVIOUR_MAP = {
    "splitInGroups": SplitInGroups,
    "coordinatedPairMovement": CoordinatedPairMovement,
    "rewardClosestFruit": RewardClosestFruit,
    "balanceLevelRewardAdaption": BalanceLevelRewardAdaption,
}
