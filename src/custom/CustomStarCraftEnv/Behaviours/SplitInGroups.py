from custom.CustomStarCraftEnv import CustomStarCraftEnv
from custom.CustomStarCraftEnv.Behaviours.SequentialBehaviour import \
    SequentialBehaviour


class SplitInGroups(SequentialBehaviour):
    min_delta = 0
    enemy_in_distance = 0
    min_ally_distant_distance = 0.123
    max_ally_close_distance = 0.08
    min_distance_moving_towards_enemies = 0.3

    def __init__(self, config, n_agents):
        super().__init__(config, n_agents)
        self.previous_enemy_distances = [0] * n_agents
        self.stats["exactly_one_ally_close"] = 0
        self.stats["moving_towards_enemies"] = 0
        self.stats["correct_grouping"] = 0
        self.stats["moving_towards_enemies"] = 0

    def is_reset(self, prev, new):
        is_moving_towards = prev < new
        in_great_steps = abs(prev - new) < SplitInGroups.min_delta

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

        correct_grouping = True
        moving_towards_enemies = True
        for index_agent_1, (id_agent_1, agent_1) in enumerate(env.agents.items()):
            distances_feature_list = [(index, feature_name) for index, feature_name in
                                      enumerate(env.get_obs_feature_names())
                                      if "ally_distance_" in feature_name]
            distances_features = {k: v for k, v in distances_feature_list}

            enemy_alive = any(enemy.health > 0 for enemy in env.enemies.values())
            ally_moving_towards_enemy = False
            if enemy_alive:
                for enemy_feature_index in enemy_distances_features.keys():
                    prev_dist = self.previous_enemy_distances[index_agent_1]
                    curr_dist = obs[index_agent_1][enemy_feature_index]
                    self.previous_enemy_distances[index_agent_1] = curr_dist
                    if curr_dist > 0 and curr_dist + SplitInGroups.min_distance_moving_towards_enemies < prev_dist:
                        ally_moving_towards_enemy = True
                        break

            moving_towards_enemies = moving_towards_enemies and ally_moving_towards_enemy
            if moving_towards_enemies:
                self.stats["moving_towards_enemies"] += 1
            # else:
            #     break

            ally_distances = []
            for index, (feature_index, feature) in enumerate(distances_features.items()):
                index_agent_2 = len(env.agents) - len(distances_features) + index
                agent_2 = env.agents[index_agent_2]
                if agent_1.health == 0 or agent_2.health == 0:
                    correct_grouping = False
                    break

                distance = obs[index_agent_1][feature_index]

                if distance == 0:
                    correct_grouping = False
                    break
                ally_distances.append(distance)

            min_one_ally_close = any(value < SplitInGroups.max_ally_close_distance for value in ally_distances)
            other_allys_distant = sum(1 for value in ally_distances if value > SplitInGroups.min_ally_distant_distance) == len(ally_distances) - 1

            exactly_one_ally_close = min_one_ally_close and other_allys_distant

            if exactly_one_ally_close:
                self.stats["exactly_one_ally_close"] += 1

            correct_grouping = correct_grouping and exactly_one_ally_close
            if not correct_grouping:
                break

        if correct_grouping:
            self.stats["correct_grouping"] += 1

        if moving_towards_enemies:
            self.stats["moving_towards_enemies"] += 1

        if correct_grouping and moving_towards_enemies:
            reward += self.reward
            for index_agent_1, (id_agent_1, agent_1) in enumerate(env.agents.items()):
                self.try_add_data(index_agent_1, round(agent_1.pos.x), round(agent_1.pos.y), 1)

        # enemy_alive = any(enemy.health > 0 for enemy in env.enemies.values())
        # enemy_in_distance = any(0 < obs[index_agent_1][enemy_feature_index] < SplitInGroups.enemy_in_distance for enemy_feature_index in
        #                         enemy_distances_features.keys())
        #
        # if enemy_alive and enemy_in_distance:
        #     if self.try_add_data(index_agent_1, round(agent_1.pos.x), round(agent_1.pos.y), distance) and self.try_add_data(index_agent_2, round(agent_2.pos.x), round(agent_2.pos.y), distance):
        #         reward += self.reward

        return reward
