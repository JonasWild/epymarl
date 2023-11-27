from custom.CustomStarCraftEnv.Metrics.DistanceToEnemiesPerEpisode import DistanceToEnemiesPerEpisode
from custom.CustomStarCraftEnv.Metrics.StepsTillFirstAttacked import StepsTillFirstAttacked
from custom.CustomStarCraftEnv.Metrics.StepsTillAllAttacked import StepsTillAllAttacked

METRIC_MAP = {
    'distanceToEnemies': DistanceToEnemiesPerEpisode,
    'stepsTillFirstAttacked': StepsTillFirstAttacked,
    'stepsTillAllAttacked': StepsTillAllAttacked,
}
