
from CustomStarCraftEnv import CustomStarCraftEnv
from CustomStarCraftEnv.Behaviours.SequentialBehaviour import \
    SequentialBehaviour


class TeamUp(SequentialBehaviour):
    min_delta = 0
    enemy_in_distance = 0

    def __init__(self, config, n_agents):
        super().__init__(config, n_agents)

    def is_reset(self, prev, new):
        is_moving_towards = prev < new
        in_great_steps = abs(prev - new) < TeamUp.min_delta

        is_kicked = is_moving_towards or in_great_steps

        return is_kicked

    def get_heatmaps(self):
        return [self.heatmaps.get_summed_agent_heatmap()] \
            # + self.heatmaps.heatmaps

    def evaluate(self, env: CustomStarCraftEnv, actions, obs):
        reward = 0

        enemy_distance_list = [(index, feature_name) for index, feature_name in
                               enumerate(env.get_obs_feature_names())
                               if "enemy_distance_" in feature_name]
        enemy_distances_features = {k: v for k, v in enemy_distance_list}

        for index_agent_1, (id_agent_1, agent_1) in enumerate(env.agents.items()):
            distances_feature_list = [(index, feature_name) for index, feature_name in
                                      enumerate(env.get_obs_feature_names())
                                      if "ally_distance_" in feature_name and int(
                    feature_name.replace("ally_distance_", "")) > index_agent_1]
            distances_features = {k: v for k, v in distances_feature_list}

            for index, (feature_index, feature) in enumerate(distances_features.items()):

                index_agent_2 = len(env.agents) - len(distances_features) + index
                agent_2 = env.agents[index_agent_2]
                if agent_1.health == 0 or agent_2.health == 0:
                    continue

                distance = obs[index_agent_1][feature_index]

                enemy_alive = any(enemy.health > 0 for enemy in env.enemies.values())
                enemy_in_distance = any(0 < obs[index_agent_1][enemy_feature_index] < TeamUp.enemy_in_distance for enemy_feature_index in
                                        enemy_distances_features.keys())

                if enemy_alive and enemy_in_distance:
                    if self.try_add_data(index_agent_1, round(agent_1.pos.x), round(agent_1.pos.y), distance) and self.try_add_data(index_agent_2, round(agent_2.pos.x), round(agent_2.pos.y), distance):
                        reward += self.reward

        return reward
