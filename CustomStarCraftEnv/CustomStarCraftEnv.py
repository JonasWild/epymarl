from smacv2.smacv2.env import MultiAgentEnv
from smacv2.smacv2.env.starcraft2.starcraft2 import StarCraft2Env
import numpy as np

import myutils.myutils
from myutils.HeatSC2map import HeatSC2map as Heatmap
from itertools import chain

from myutils.KickingQueue import KickingQueue, ConditionalResetKickingQueue


class CustomStarCraftEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.custom_env_args = kwargs["custom_env_args"]
        self.env = StarCraft2Env(**kwargs)

        self.previous_delta_ally = []
        self.egoistic_behaviour_logger = []

        # queues cause for every agent
        self.agent_actions_queues = []
        self.agents_attacking_same_enemy_queues = []

        self.agents_running_from_enemies_queues = []
        self.agents_running_to_enemies_queues = []

        self.agents_running_from_ally_queues = []
        self.agents_running_to_ally_queues = []

        self.agents_running_to_block_corridor_queues = []
        self.agents_blocking_corridor_queues = []

        self.agents_not_attacking_queues = []
        self.agents_delta_health_queues = []
        self.is_queues_initialized = False

        self.agents_attacking_same_enemy_heatmaps = []
        self.agents_running_from_enemies_heatmaps = []
        self.agents_running_to_enemies_heatmaps = []
        self.agents_running_from_ally_heatmaps = []
        self.agents_running_to_ally_heatmaps = []
        self.agents_only_not_attacking_heatmaps = []
        self.agents_loosing_health_heatmaps = []

        self.is_heatmaps_initialized = False

    def _is_attack_action(self, action):
        return action >= self.env.n_actions - len(self.enemies.items())

    def get_heatmaps(self):
        all_heatmaps = []
        for i in range(len(self.env.agents)):
            all_heatmaps.append(self.agents_attacking_same_enemy_heatmaps[i])
            all_heatmaps.append(self.agents_running_from_enemies_heatmaps[i])
            all_heatmaps.append(self.agents_running_to_enemies_heatmaps[i])
            all_heatmaps.append(self.agents_running_from_ally_heatmaps[i])
            all_heatmaps.append(self.agents_running_to_ally_heatmaps[i])
            all_heatmaps.append(self.agents_only_not_attacking_heatmaps[i])
            all_heatmaps.append(self.agents_loosing_health_heatmaps[i])

        summed_up_heatmap = Heatmap("summary")
        for x, row in enumerate(summed_up_heatmap.data):
            for y, _ in enumerate(row):
                for h in all_heatmaps:
                    summed_up_heatmap.add_value(x, y, h.get_value(x, y))
        all_heatmaps.append(summed_up_heatmap)

        for heatmaps in self._get_all_agents_heatmaps():
            new_heatmap = Heatmap(heatmaps[0].name[:-1] + "s_summed")
            for x, row in enumerate(new_heatmap.data):
                for y, _ in enumerate(row):
                    for h in heatmaps:
                        new_heatmap.add_value(x, y, h.get_value(x, y))
            all_heatmaps.append(new_heatmap)

        return all_heatmaps

    def evaluate_actions(self, actions, obs):
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        # ally_attacked = [True if self._is_attack_action(actions[index]) else False for index in range(len(self.agents.items()))]
        delta_allies = [0] * len(self.agents.items())
        for index, (al_id, al_unit) in enumerate(self.agents.items()):
            self.agent_actions_queues[index].put(actions[index])

            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                        self.previous_ally_units[al_id].health
                        + self.previous_ally_units[al_id].shield
                )

                if al_unit.health == 0:
                    # just died
                    delta_deaths -= self.reward_death_value
                    current_delta_ally = prev_health
                else:
                    # still alive
                    current_delta_ally = prev_health - al_unit.health - al_unit.shield

                delta_ally += (
                    current_delta_ally
                )
                self.agents_delta_health_queues[index].put(current_delta_ally)
                delta_allies[index] = current_delta_ally

        # consecutive ???

        agents = [i for i in self.agents.values()]
        enemies = [i for i in self.enemies.values()]

        self._evaluate_distances(obs, agents, agents, "ally_distance_", self.agents_running_from_ally_queues,
                                 self.agents_running_from_ally_heatmaps, self.agents_running_to_ally_queues,
                                 self.agents_running_to_ally_heatmaps)
        self._evaluate_distances(obs, agents, enemies, "enemy_distance_",  self.agents_running_from_enemies_queues,
                                 self.agents_running_from_enemies_heatmaps, self.agents_running_to_enemies_queues,
                                 self.agents_running_to_enemies_heatmaps)

        # only ally that is not attacking => egoistic behaviour
        index_only_ally_not_being_attacked = myutils.myutils.get_zero_index_from_num_array(delta_allies)
        if index_only_ally_not_being_attacked != -1:
            self.agents_not_attacking_queues[index_only_ally_not_being_attacked].put(True)
            egoistic_value = self.agents_not_attacking_queues[index_only_ally_not_being_attacked].qsize() * 0.1
            self.agents_only_not_attacking_heatmaps[index_only_ally_not_being_attacked].add_value(
                round(agents[index_only_ally_not_being_attacked].pos.x),
                round(agents[index_only_ally_not_being_attacked].pos.y),
                -egoistic_value)
        for i in range(len(agents)):
            if i is not index_only_ally_not_being_attacked:
                self.agents_not_attacking_queues[i].reset()

        # allies attacking the same enemy => cooperative behaviour
        same_attacking_agents = list(chain.from_iterable(
            [same_attacking_agents_indices for action, same_attacking_agents_indices in
             myutils.myutils.find_same_indices(actions).items() if self._is_attack_action(action)]))
        for i in range(len(self.agents)):
            if i in same_attacking_agents:
                self.agents_attacking_same_enemy_queues[i].put(True)
                self.agents_attacking_same_enemy_heatmaps[i].add_value(
                    round(agents[i].pos.x), round(
                        agents[i].pos.y), self.agents_attacking_same_enemy_queues[i].qsize() * 0.1)
            else:
                self.agents_attacking_same_enemy_queues[i].reset()

        for index, (e_id, e_unit) in enumerate(self.enemies.items()):
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                        self.previous_enemy_units[e_id].health
                        + self.previous_enemy_units[e_id].shield
                )

                if e_unit.health == 0:
                    delta_deaths += self.reward_death_value
                    current_enemy_delta = prev_health
                else:
                    current_enemy_delta = prev_health - e_unit.health - e_unit.shield

                delta_enemy += current_enemy_delta
                # previous_delta_enemies[index] = current_enemy_delta

    def reset(self):
        env = self.env.reset()

        if not self.is_heatmaps_initialized:
            self.init_heatmaps()

        if not self.is_queues_initialized:
            self.init_queues()

        return env

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError

    def get_obs(self):
        return self.env.get_obs()

    def get_obs_feature_names(self):
        return self.env.get_obs_feature_names()

    def get_state(self):
        return self.env.get_state()

    def get_state_feature_names(self):
        return self.env.get_state_feature_names()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_total_actions(self):
        return self.env.get_total_actions()

    def get_capabilities(self):
        return self.env.get_capabilities()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def render(self):
        return self.env.render()

    def step(self, actions):
        return self.env.step(actions)

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        return self.env.close()

    def _evaluate_distances(self, obs, from_distance_array, to_distance_array, distance_feature_substring,
                            from_queues, from_heatmaps, to_queues, to_heatmaps):
        distances_features = [(index, feature_name) for index, feature_name in
                              enumerate(self.env.obs_feature_names)
                              if distance_feature_substring in feature_name]

        # get distances to allies
        for from_index, from_unit in enumerate(from_distance_array):
            for to_index, to_unit in enumerate(to_distance_array):
                if from_unit != to_unit and from_index != to_index:
                    # because in the observation the self agent is not visible
                    wanted_feature_index = distances_features[to_index - 1][0]
                    arr = obs[from_index]
                    distance = arr[wanted_feature_index]

                    from_queues[from_index].put(distance)
                    from_heatmaps[from_index].add_value(round(from_unit.pos.x),
                                                        round(from_unit.pos.y),
                                                        -(from_queues[from_index].qsize() * 0.001))

                    to_queues[from_index].put(distance)
                    to_heatmaps[from_index].add_value(round(from_unit.pos.x),
                                                      round(from_unit.pos.y),
                                                      to_queues[from_index].qsize() * 0.001)

    def init_queues(self):
        self.is_queues_initialized = True
        for i in range(len(self.env.agents)):
            self.agent_actions_queues.append(KickingQueue(10))
            self.agents_attacking_same_enemy_queues.append(KickingQueue(10))
            self.agents_running_from_enemies_queues.append(ConditionalResetKickingQueue(100,
                                                                                        reset_condition=lambda
                                                                                            old_distance,
                                                                                            new_distance: old_distance > new_distance))
            self.agents_running_to_enemies_queues.append(ConditionalResetKickingQueue(100,
                                                                                      reset_condition=lambda
                                                                                          old_distance,
                                                                                          new_distance: old_distance < new_distance))
            self.agents_running_from_ally_queues.append(ConditionalResetKickingQueue(100,
                                                                                     reset_condition=lambda
                                                                                         old_distance,
                                                                                         new_distance: old_distance > new_distance))
            self.agents_running_to_ally_queues.append(ConditionalResetKickingQueue(100,
                                                                                   reset_condition=lambda old_distance,
                                                                                                          new_distance: old_distance < new_distance))
            self.agents_running_to_block_corridor_queues.append(KickingQueue(10))
            self.agents_blocking_corridor_queues.append(KickingQueue(10))
            self.agents_not_attacking_queues.append(KickingQueue(100))
            self.agents_delta_health_queues.append(KickingQueue(10))

    def init_heatmaps(self):
        self.is_heatmaps_initialized = True
        Heatmap.init_map_size(self.env.map_play_area_max.x, self.env.map_play_area_max.y)
        for i in range(len(self.env.agents)):
            self.agents_attacking_same_enemy_heatmaps.append(Heatmap(f"attacking_same_enemy_agent{i}"))
            self.agents_running_from_enemies_heatmaps.append(Heatmap(f"running_from_enemies_agent{i}"))
            self.agents_running_to_enemies_heatmaps.append(Heatmap(f"running_towards_enemies_agent{i}"))
            self.agents_running_from_ally_heatmaps.append(Heatmap(f"running_from_ally_agent{i}"))
            self.agents_running_to_ally_heatmaps.append(Heatmap(f"running_to_ally_agent{i}"))
            self.agents_only_not_attacking_heatmaps.append(Heatmap(f"only_not_attacking_agent{i}"))
            self.agents_loosing_health_heatmaps.append(Heatmap(f"loosing_health_agent{i}"))

    def _get_all_agents_heatmaps(self):
        return [
            self.agents_attacking_same_enemy_heatmaps,
            self.agents_running_from_enemies_heatmaps,
            self.agents_running_to_enemies_heatmaps,
            self.agents_running_from_ally_heatmaps,
            self.agents_running_to_ally_heatmaps,
            self.agents_only_not_attacking_heatmaps,
            self.agents_loosing_health_heatmaps]
