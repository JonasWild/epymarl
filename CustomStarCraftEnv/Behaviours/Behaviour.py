from abc import abstractmethod

from CustomStarCraftEnv import CustomStarCraftEnv


class Behaviour:
    def __init__(self, config, n_agents):
        self.enabled = config['enabled']
        self.reward = config['reward']
        self.n_agents = n_agents

    @abstractmethod
    def get_history_data(self):
        pass

    @abstractmethod
    def get_heatmaps(self):
        pass

    @abstractmethod
    def try_add_data(self, index, row, col, value):
        pass

    @abstractmethod
    def process_episode_finished(self):
        pass

    @abstractmethod
    def reset_history(self):
        pass

    @abstractmethod
    def get_stats(self):
        pass

    @abstractmethod
    def reset_stats(self):
        pass

    @abstractmethod
    def reset_heatmap_registry(self):
        pass

    @abstractmethod
    def evaluate(self, env: CustomStarCraftEnv, actions, obs):
        pass

    @abstractmethod
    def reset_episode(self):
        pass

