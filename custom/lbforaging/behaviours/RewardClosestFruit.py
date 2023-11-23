from collections import deque

from custom.Behaviour import Behaviour
from custom.lbforaging.behaviour_utils import extract_agents_positions_and_level, extract_foods_positions, \
    calculate_distance, find_closest_pickable_fruit, is_moving_closer
from custom.lbforaging.behaviours.TimeDependentBehaviour import TimeDependentBehaviour


class RewardClosestFruit(TimeDependentBehaviour):

    def evaluate(self, env, actions, obs):
        reward = 0
        agents_positions_and_levels, foods_positions_and_level = self.get_env_info(env)

        for idx, (agent_position, agent_level) in enumerate(agents_positions_and_levels):
            closest_fruit = find_closest_pickable_fruit(agent_position, agent_level, agents_positions_and_levels)
            if closest_fruit and is_moving_closer(agent_idx=idx, target_pos=closest_fruit, position_history=self.position_history):
                reward += self.reward

        return reward
