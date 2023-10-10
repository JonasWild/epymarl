import sys

import numpy as np

from CustomStarCraftEnv import CustomStarCraftEnv
from CustomStarCraftEnv.Behaviours.SequentialBehaviour import \
    SequentialBehaviour


class StayTogether(SequentialBehaviour):
    min_delta = 0
    enemy_in_distance = 0

    def __init__(self, config, n_agents):
        super().__init__(config, n_agents)

        self.previous_group_distance = 0
        self.total_group_distance = 0
        self.total_group_distance_delta = 0

        self.stats["total_group_distance"] = 0
        self.stats["total_group_distance_delta"] = 0

    def is_reset(self, prev, new):
        return abs(prev - new) < StayTogether.min_delta and prev < new

    def get_heatmaps(self):
        return [self.heatmaps.get_summed_agent_heatmap()]

    def reset_episode(self):
        self.previous_group_distance = 0

    def evaluate(self, env: CustomStarCraftEnv, actions, obs):
        reward = 0

        group_distance = 0

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
                group_distance += distance

        # not just one ally standing there
        if group_distance > 0:
            enemy_alive = any(enemy.health > 0 for enemy in env.enemies.values())
            enemy_in_distance = any(0 < obs[index_agent_1][enemy_feature_index] < StayTogether.enemy_in_distance for enemy_feature_index in
                                    enemy_distances_features.keys() for index_agent_1 in range(len(env.agents)))

            if enemy_alive and enemy_in_distance:
                for index_agent_1, (id_agent_1, agent_1) in enumerate(env.agents.items()):
                    self.try_add_data(index_agent_1, round(agent_1.pos.x), round(agent_1.pos.y), group_distance)

            self.total_group_distance += group_distance
            self.total_group_distance_delta += self.previous_group_distance - group_distance
            self.stats["total_group_distance"] += self.total_group_distance
            self.stats["total_group_distance_delta"] += self.total_group_distance_delta

            if group_distance < self.previous_group_distance and self.previous_group_distance - group_distance > StayTogether.min_delta:
                reward += self.reward

            self.previous_group_distance = group_distance

        return reward
