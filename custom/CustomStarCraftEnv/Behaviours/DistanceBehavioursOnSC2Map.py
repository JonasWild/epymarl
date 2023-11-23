from custom.Behaviour import BehaviourBase
from custom.CustomStarCraftEnv.Behaviours.SequentialBehaviour import \
    ConsecutiveCooperativeBehaviourOnSC2Map


class DistanceBehavioursOnSC2Map(BehaviourBase):
    def add_data(self, index, row, col, value):
        raise NotImplementedError()

    def __init__(self, n_agents, get_obs_features: callable, get_enemies: callable, metric_multiplier=1):
        self.behaviours = [ConsecutiveCooperativeBehaviourOnSC2Map(f"{reset_c[0]}_{target}", n_agents, reset_c[1], metric_multiplier)
                           for target in ["ally", "enemy"]
                           for reset_c in [("running_to", lambda o, n: o <= n), ("running_from", lambda o, n: o >= n)]]
        self.get_obs_features = get_obs_features
        self.get_enemies = get_enemies
        self.name = "distance_behaviour"

    def evaluate(self, agents: [], obs: [], actions: []):
        pos_reward = 0
        neg_reward = 0
        stats = {}

        assert(len(self.behaviours) == 4)

        for agent_behaviour in self.behaviours:
            from_units = None
            to_units = None
            obs_feature_substring = None
            if "ally" in agent_behaviour.name:
                from_units = agents
                to_units = agents
                obs_feature_substring = "ally_distance_"
            elif "enemy" in agent_behaviour.name:
                from_units = agents
                to_units = self.get_enemies()
                obs_feature_substring = "enemy_distance_"

            if obs_feature_substring is None or from_units is None or to_units is None:
                raise Exception("obs_feature_substring cannot be None.")

            distances_features = [(index, feature_name) for index, feature_name in
                                  enumerate(self.get_obs_features())
                                  if obs_feature_substring in feature_name]

            # get distances to allies/enemies
            for from_index, (_, from_unit) in enumerate(from_units.items()):
                for to_index, (_, to_unit) in enumerate(to_units.items()):
                    if from_unit != to_unit and from_index != to_index:
                        if from_unit.health == 0 or to_unit.health == 0:
                            continue
                        
                        # because in the observation the self agent is not visible
                        wanted_feature_index = distances_features[to_index - 1][0]
                        arr = obs[from_index]

                        distance = arr[wanted_feature_index]
                        reward = agent_behaviour.metric_multiplier * distance * (agent_behaviour.queues[from_index].qsize() + 1)

                        agent_behaviour.add_data(from_index, round(from_unit.pos.x),
                                                 round(from_unit.pos.y),
                                                 reward)

                        cur_pos_reward = 0
                        cur_neg_reward = 0
                        if "running_to_agent" in agent_behaviour.name or "running_to_enemy" in agent_behaviour.name:
                            cur_pos_reward = reward
                        elif "running_from_agent" in agent_behaviour.name or "running_from_enemy" in agent_behaviour.name:
                            cur_neg_reward = reward
                        stats[agent_behaviour.name] = (cur_pos_reward, cur_neg_reward)
                        pos_reward += agent_behaviour.queues[from_index].qsize()
                        neg_reward -= agent_behaviour.queues[from_index].qsize()

        return (pos_reward, neg_reward), stats, {}
