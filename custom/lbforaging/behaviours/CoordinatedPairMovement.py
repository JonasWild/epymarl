from collections import deque

from custom.Behaviour import Behaviour
from custom.lbforaging.behaviour_utils import calculate_distance, extract_agents_positions_and_level, \
    extract_foods_positions, are_moving_towards_food
from custom.lbforaging.behaviours.TimeDependentBehaviour import TimeDependentBehaviour


class CoordinatedPairMovement(TimeDependentBehaviour):

    def evaluate(self, env, actions, obs):
        reward = 0
        agents_positions_and_level, foods_positions_and_level = self.get_env_info(env)

        # Check for each pair of agents#
        for food_position, food_level in foods_positions_and_level:
            possible_pairs_per_food = []
            for i in range(len(agents_positions_and_level)):
                for j in range(i + 1, len(agents_positions_and_level)):
                    combined_level = agents_positions_and_level[i][1] + agents_positions_and_level[j][1]

                    # Identify a fruit that requires the combined level of both agents
                    if combined_level >= food_level:
                        possible_pairs_per_food.append((i, j))

            # only possibility
            if len(possible_pairs_per_food) == 1:
                if are_moving_towards_food([possible_pairs_per_food[0][0], possible_pairs_per_food[0][1]], food_position, self.position_history):
                    reward += self.reward

        return reward
