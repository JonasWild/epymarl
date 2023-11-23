from custom.CustomStarCraftEnv.Behaviours import TeamUp, AttackingSame, NotAttacking, StayTogether
from custom import Behaviour
from custom.CustomStarCraftEnv.Behaviours.SplitInGroups import SplitInGroups
from custom.myutils.HeatSC2map import HeatSC2map


def _get_count_string(behaviour: Behaviour):
    return behaviour.__class__.__name__ + "_count"


def _get_rewards_string(behaviour: Behaviour):
    return behaviour.__class__.__name__ + "_reward"


class BehaviourRegistry:
    def __init__(self, config, n_agents, behaviour_mapping):
        self.config = config
        self.n_agents = n_agents
        self.behaviours = []
        self.reward_counts = {}
        self.rewards = {}
        self.current_t = 0
        self.behaviour_mapping = behaviour_mapping

    def initialize(self):
        # Check if self.config is not None and that 'behaviours' exists and is a dictionary
        if self.config and 'behaviours' in self.config and isinstance(self.config['behaviours'], dict):
            # Register all behaviors based on the config
            for behaviour_name, behaviour_config in self.config['behaviours'].items():
                behaviour_class = self.behaviour_mapping.get(behaviour_name)
                if behaviour_class:
                    behaviour = behaviour_class(behaviour_config, self.n_agents)
                    self.behaviours.append(behaviour)
                    self.reward_counts[_get_count_string(behaviour)] = 0
                    self.rewards[_get_rewards_string(behaviour)] = 0
        else:
            print("Config is None, or behaviours not found or not a dictionary")

    def reset_heatmaps(self):
        for behaviour in self.behaviours:
            behaviour.reset_heatmap_registry()

    def get_behaviour_stats(self):
        behaviour_stats = {}
        for behaviour in self.behaviours:
            stats = {} if behaviour.get_stats() is None else behaviour.get_stats().items()
            for k, v in stats:
                behaviour_stats[behaviour.__class__.__name__ + "_" + k] = v

        for k, v in self.reward_counts.items():
            behaviour_stats[k] = float(v) / float(max(self.current_t, 1))

        for k, v in self.rewards.items():
            behaviour_stats[k] = float(v) / float(max(self.current_t, 1))

        return behaviour_stats

    def reset_episode(self):
        for behaviour in self.behaviours:
            behaviour.reset_episode()

    def reset_stats(self):
        self.current_t = 0
        for behaviour in self.behaviours:
            behaviour.reset_stats()

        for k, v in self.reward_counts.items():
            self.reward_counts[k] = 0

        for k, v in self.rewards.items():
            self.rewards[k] = 0

    def get_heatmaps(self):
        all_heatmaps = [heatmap for behaviour in self.behaviours for heatmap in behaviour.get_heatmaps()]

        if len(all_heatmaps) > 0:
            summed_up_heatmap = HeatSC2map("cooperation_summary")

            for x, row in enumerate(summed_up_heatmap.data):
                for y, _ in enumerate(row):
                    for h in all_heatmaps:
                        summed_up_heatmap.add_value(x, y, h.get_value(x, y))

            all_heatmaps.append(summed_up_heatmap)
        return all_heatmaps

    def evaluate_behaviors(self, env, actions, obs):
        self.current_t += 1

        behaviour_reward = 0
        for index, behaviour in enumerate(self.behaviours):
            reward = behaviour.evaluate(env, actions, obs)

            if reward != 0:
                self.reward_counts[_get_count_string(behaviour)] += 1
                self.rewards[_get_rewards_string(behaviour)] += reward

            if behaviour.enabled:
                behaviour_reward += reward

        return behaviour_reward
