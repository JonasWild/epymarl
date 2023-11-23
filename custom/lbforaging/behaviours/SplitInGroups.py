from collections import deque

from custom.lbforaging.behaviour_utils import calculate_distance, extract_agents_positions_and_level, \
    extract_foods_positions, detect_pairs_heading_for_food, are_moving_towards_food
from custom.lbforaging.behaviours.TimeDependentBehaviour import TimeDependentBehaviour


class SplitInGroups(TimeDependentBehaviour):
    def __init__(self, config, n_agents):
        super().__init__(config, n_agents)
        self.intermediate_rewards_enabled = config["intermediate_rewards_enabled"]

    def evaluate(self, env, actions, obs):
        reward = 0
        agents_positions_and_level, foods_positions_and_level = self.get_env_info(env)

        pairs = detect_pairs_heading_for_food(agents_positions_and_level, foods_positions_and_level)

        if len(pairs) < 2:
            return 0

        both_moving_towards = True
        for (pair, food_position) in pairs:
            if not are_moving_towards_food(pair, food_position, self.position_history):
                both_moving_towards = False
                break
            else:
                # reward += self.reward * 0.1
                continue

        if both_moving_towards:
            reward += self.reward

        return reward  # return the calculated rewards for each agent
