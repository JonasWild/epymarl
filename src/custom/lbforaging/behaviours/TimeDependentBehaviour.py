from abc import ABC
from collections import deque

from custom.Behaviour import Behaviour
from custom.lbforaging.behaviour_utils import extract_agents_positions_and_level, \
    extract_foods_positions


class TimeDependentBehaviour(Behaviour, ABC):
    def __init__(self, config, n_agents):
        super().__init__(config, n_agents)
        self.position_history = deque(maxlen=2)

    def get_history_data(self):
        # Implement according to your requirements
        pass

    def get_heatmaps(self):
        return []

    def try_add_data(self, index, row, col, value):
        # Implement according to your requirements
        pass

    def process_episode_finished(self):
        # Implement according to your requirements
        pass

    def reset_history(self):
        # Implement according to your requirements
        pass

    def get_stats(self):
        return {}

    def reset_stats(self):
        # Implement according to your requirements
        pass

    def reset_heatmap_registry(self):
        # Implement according to your requirements
        pass

    def get_env_info(self, env):
        agents_positions_and_levels = extract_agents_positions_and_level(env)
        self.position_history.append([agent_info[0] for agent_info in agents_positions_and_levels])

        foods_positions_and_levels = extract_foods_positions(env)
        return agents_positions_and_levels, foods_positions_and_levels

    def reset_episode(self):
        self.position_history = []
