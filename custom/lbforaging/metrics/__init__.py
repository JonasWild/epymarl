from .metrics import (
    HighestLevelFoodMetric,
    HighestAgentLevelMetric,
    LowestAgentLevelMetric,
    AgentsSummedLevelMetric,
    AgentsLevelDeviationMetric,
    AgentsScoreMetric,
    AgentsRewardMetric,
    FirstAgentRewardMetric,
    FirstAgentCoupleRewardMetric,
    StepsTillAllAgentsCollect,
    StepsSpendWaitingAtFruit,
    CoupledCollectedFood,
)

METRIC_MAP = {
    'highestLevelFood': HighestLevelFoodMetric,
    'highestAgentLevel': HighestAgentLevelMetric,
    'lowestAgentLevel': LowestAgentLevelMetric,
    'agentsSummedLevel': AgentsSummedLevelMetric,
    'agentsDeviationLevel': AgentsLevelDeviationMetric,
    'firstAgentReward': FirstAgentRewardMetric,
    'firstAgentCoupleReward': FirstAgentCoupleRewardMetric,
    'stepsTillAllAgentsCollect': StepsTillAllAgentsCollect,
    'stepsSpendWaitingAtFruit': StepsSpendWaitingAtFruit,
    'coupledCollectedFood': CoupledCollectedFood,
}
