from custom.CustomStarCraftEnv import CustomStarCraftEnv
from custom.MetricBase import MetricBase


class DistanceToEnemiesPerEpisode(MetricBase):
    def __init__(self, config, n_agents):
        self.data = []
        super().__init__(config, n_agents)

    def add_data(self, env: CustomStarCraftEnv, actions, obs):
        enemy_distance_list = [(index, feature_name) for index, feature_name in
                               enumerate(env.get_obs_feature_names())
                               if "enemy_distance_" in feature_name]
        enemy_distances_features = {k: v for k, v in enemy_distance_list}

        group_distance = 0
        for index_agent_1, _ in enumerate(env.agents.items()):
            for enemy_index, enemy_feature_index in enumerate(enemy_distances_features.keys()):
                if list(env.enemies.values())[enemy_index].health > 0:
                    curr_dist = obs[index_agent_1][enemy_feature_index]
                    group_distance += curr_dist

        self.data.append(group_distance)

    def evaluate_episode(self):
        mean_distance = sum(self.data) / len(self.data) if len(self.data) > 0 else 0
        self.data.clear()
        self.total += mean_distance
