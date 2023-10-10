from abc import abstractmethod

from CustomStarCraftEnv import CustomStarCraftEnv


class MetricBase:
    UNSET_TOTAL_DEFAULT = 0

    def __init__(self, config, n_agents):
        self.enabled = config['enabled']
        self.n_agents = n_agents
        self.total = MetricBase.UNSET_TOTAL_DEFAULT

    def get_and_reset_total(self):
        total = self.total
        self.total = MetricBase.UNSET_TOTAL_DEFAULT
        return total

    @abstractmethod
    def add_data(self, env: CustomStarCraftEnv, actions, obs):
        pass

    @abstractmethod
    def evaluate_episode(self):
        pass
