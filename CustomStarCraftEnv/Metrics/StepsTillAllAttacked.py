from CustomStarCraftEnv import CustomStarCraftEnv
from CustomStarCraftEnv.Metrics.MetricBase import MetricBase


class StepsTillAllAttacked(MetricBase):
    default_all_attacked = -1

    def __init__(self, config, n_agents):
        self.steps_to_all_attacked = StepsTillAllAttacked.default_all_attacked
        self.total_steps = 0
        self.agent_attacked = [False] * n_agents
        super().__init__(config, n_agents)

    def add_data(self, env: CustomStarCraftEnv, actions, obs):
        self.total_steps += 1

        if self.steps_to_all_attacked != StepsTillAllAttacked.default_all_attacked:
            return

        agent_attacked_enemy_indices = [-1] * self.n_agents
        for index, action in enumerate(actions.squeeze().tolist()):
            if env.is_attack_action(action):
                agent_attacked_enemy_indices[index] = action - (env.n_actions - len(env.enemies.items()))

        delta_enemy = 0
        for enemy_index, (e_id, e_unit) in enumerate(env.enemies.items()):
            if not env.death_tracker_enemy[e_id]:
                prev_health = (
                        env.previous_enemy_units[e_id].health
                        + env.previous_enemy_units[e_id].shield
                )

                delta_enemy += prev_health - e_unit.health - e_unit.shield
                if delta_enemy > 0 and enemy_index in agent_attacked_enemy_indices:
                    indices = [i for i, x in enumerate(agent_attacked_enemy_indices) if x == enemy_index]
                    for index in indices:
                        self.agent_attacked[index] = True

        if all(self.agent_attacked):
            self.steps_to_all_attacked = self.total_steps

    def evaluate_episode(self):
        steps_to_all_attacked = self.steps_to_all_attacked
        self.steps_to_all_attacked = StepsTillAllAttacked.default_all_attacked
        self.total_steps = 0
        self.agent_attacked = [False] * self.n_agents
        self.total += steps_to_all_attacked
