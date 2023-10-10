from CustomStarCraftEnv import CustomStarCraftEnv
from CustomStarCraftEnv.Metrics.MetricBase import MetricBase


class StepsTillFirstAttacked(MetricBase):
    default_first_attack = -1

    def __init__(self, config, n_agents):
        self.steps_to_first_attack = StepsTillFirstAttacked.default_first_attack
        self.total_steps = 0
        super().__init__(config, n_agents)

    def add_data(self, env: CustomStarCraftEnv, actions, obs):
        self.total_steps += 1

        if self.steps_to_first_attack != StepsTillFirstAttacked.default_first_attack:
            return

        delta_enemy = 0
        for e_id, e_unit in env.enemies.items():
            if not env.death_tracker_enemy[e_id]:
                prev_health = (
                        env.previous_enemy_units[e_id].health
                        + env.previous_enemy_units[e_id].shield
                )

                delta_enemy += prev_health - e_unit.health - e_unit.shield

        if delta_enemy != 0:
            self.steps_to_first_attack = self.total_steps

    def evaluate_episode(self):
        steps_to_first_attack = self.steps_to_first_attack
        self.steps_to_first_attack = StepsTillFirstAttacked.default_first_attack
        self.total_steps = 0
        self.total += steps_to_first_attack
