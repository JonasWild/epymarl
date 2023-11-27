from custom.lbforaging.behaviour_utils import are_moving_towards_food, is_moving_closer
from custom.lbforaging.behaviours.TimeDependentBehaviour import TimeDependentBehaviour


class WaitAtUnpickableFruit(TimeDependentBehaviour):
    def evaluate(self, env, actions, obs):
        reward = 0
        agents_positions_and_level, foods_positions_and_level = self.get_env_info(env)

        # Identify the agent with the lowest level
        lowest_level_agent_level = min(agents_positions_and_level, key=lambda x: x[1])

        if env.loading_players:  # Check if the list is not empty
            lowest_picking_level_agent = min(env.loading_players, key=lambda x: x.level)
        else:
            return 0

        if lowest_picking_level_agent and lowest_picking_level_agent.level == lowest_level_agent_level:
            food_row, food_col = env.adjacent_food_location(*lowest_picking_level_agent.position)
            food = env.field[food_row, food_col]

            for agent_idx, p in enumerate(env.players):
                if p is not lowest_picking_level_agent:
                    combined_level = lowest_picking_level_agent.level + p.level

                    # Identify a fruit that requires the combined level of both agents
                    if combined_level >= food and is_moving_closer(agent_idx, (food_row, food_col), self.position_history):
                        reward += self.reward

        return reward
