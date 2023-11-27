import numpy as np
from custom.MetricBase import MetricBase


class HighestLevelFoodMetric(MetricBase):
    def add_data(self, env, actions, obs):
        pass

    def evaluate_episode(self, env):
        highest = np.max(env.field)
        self.total += highest


class HighestAgentLevelMetric(MetricBase):
    def add_data(self, env, actions, obs):
        pass

    def evaluate_episode(self, env):
        highest = np.max([player.level if player.level is not None else 0 for player in env.players])
        self.total += highest


class LowestAgentLevelMetric(MetricBase):
    def add_data(self, env, actions, obs):
        pass

    def evaluate_episode(self, env):
        highest = np.min([player.level if player.level is not None else 0 for player in env.players])
        self.total += highest


class AgentsSummedLevelMetric(MetricBase):
    def add_data(self, env, actions, obs):
        pass

    def evaluate_episode(self, env):
        players_level = np.sum([player.level if player.level is not None else 0 for player in env.players])
        self.total += players_level


class AgentsLevelDeviationMetric(MetricBase):
    def add_data(self, env, actions, obs):
        pass

    def evaluate_episode(self, env):
        players_level = np.std([player.level if player.level is not None else 0 for player in env.players])
        self.total += players_level


class AgentsScoreMetric(MetricBase):
    def add_data(self, env, actions, obs):
        pass

    def evaluate_episode(self, env):
        players_score = np.sum([player.score if player.score is not None else 0 for player in env.players])
        self.total += players_score


class AgentsRewardMetric(MetricBase):
    def add_data(self, env, actions, obs):
        pass

    def evaluate_episode(self, env):
        players_reward = np.sum([player.reward if player.reward is not None else 0 for player in env.players])
        self.total += players_reward


class FirstAgentRewardMetric(MetricBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_at_step = 150

    def add_data(self, env, actions, obs):
        if self.first_at_step < 150:
            return

        reward = sum(player.reward > 0 for player in env.players)
        if reward > 0:
            self.first_at_step = env.current_step

    def evaluate_episode(self, env):
        self.total += self.first_at_step
        self.first_at_step = 150


class FirstAgentCoupleRewardMetric(MetricBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_at_step = 150

    def add_data(self, env, actions, obs):
        if self.first_at_step < 150:
            return

        num_players_with_reward = sum(1 for player in env.players if player.reward > 0)
        if num_players_with_reward > 1:
            self.first_at_step = env.current_step

    def evaluate_episode(self, env):
        self.total += self.first_at_step
        self.first_at_step = 150


class StepsTillAllAgentsCollect(MetricBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_at_step = 150

    def add_data(self, env, actions, obs):
        if self.first_at_step < 150:
            return

        num_players_with_reward = sum(1 for player in env.players if player.reward > 0)
        if num_players_with_reward == 4:
            self.first_at_step = env.current_step

    def evaluate_episode(self, env):
        self.total += self.first_at_step
        self.first_at_step = 150


class StepsSpendWaitingAtFruit(MetricBase):

    def add_data(self, env, actions, obs):
        pass

    def evaluate_episode(self, env):
        self.total += env.loading_failed_in_episode
        env.loading_failed_in_episode = 0


class CoupledCollectedFood(MetricBase):

    def add_data(self, env, actions, obs):
        pass

    def evaluate_episode(self, env):
        self.total += env.coupled_collected_food_episode
        env.coupled_collected_food_episode = 0
