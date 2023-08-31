import sys
from abc import abstractmethod

from CustomStarCraftEnv import CustomStarCraftEnv
from CustomStarCraftEnv.Behaviours.Behaviour import Behaviour
from CustomStarCraftEnv.Behaviours.HeatmapRegistry import HeatmapRegistry
from myutils.KickingQueue import ConditionalResetHistoryQueue
from queue import Queue


class SequentialBehaviour(Behaviour):
    reset_code = -sys.maxsize - 1

    def __init__(self, config, n_agents):
        super().__init__(config, n_agents)
        self.heatmaps = HeatmapRegistry(self.__class__.__name__, n_agents)
        self.queues = [Queue() for _ in range(n_agents)]
        self.episode_history = [[] for _ in range(n_agents)]
        self.history = []
        self.stats = {}

    def get_history_data(self):
        return [self.history]

    def get_heatmaps(self):
        return [self.heatmaps.get_summed_agent_heatmap()]

    @abstractmethod
    def is_reset(self, prev, new):
        pass

    def try_add_data(self, index, row, col, value):
        if value == SequentialBehaviour.reset_code:
            self.queues[index].queue.clear()
            return False

        if not self.queues[index].empty():
            last_item = self.queues[index].queue[-1]  # Get the last item in the queue
            if self.is_reset(last_item, value):
                self.queues[index].queue.clear()
                return False

        self.queues[index].put(value)

        if self.queues[index].qsize() < 2:
            return False

        self.episode_history[index].append(value)
        self.heatmaps.add_data(index, row, col, value)

        return True

    def process_episode_finished(self):
        for queue in self.queues:
            queue.queue.clear()
        self.history.append(self.episode_history)
        self.episode_history = [[] for _ in range(self.n_agents)]

    def reset_history(self):
        self.history.clear()
        self.heatmaps.reset_heatmaps()

    def reset_heatmap_registry(self):
        self.heatmaps.reset_heatmaps()

    def get_stats(self):
        return self.stats

    def reset_stats(self):
        for k, v in self.stats.items():
            self.stats[k] = 0

    def reset_episode(self):
        pass
