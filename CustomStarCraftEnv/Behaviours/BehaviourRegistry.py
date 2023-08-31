from CustomStarCraftEnv.Behaviours import Behaviour, StayTogether, TeamUp, NotAttacking, AttackingSame
from myutils.HeatSC2map import HeatSC2map


def _get_count_string(behaviour: Behaviour):
    return behaviour.__class__.__name__ + "_count"


def _get_rewards_string(behaviour: Behaviour):
    return behaviour.__class__.__name__ + "_reward"


class BehaviourRegistry:
    def __init__(self, config, n_agents):
        self.config = config
        self.n_agents = n_agents
        self.behaviours = []
        self.reward_counts = {}
        self.rewards = {}

        self.behaviour_mapping = {
            'stayTogether': StayTogether.StayTogether,
            'teamUp': TeamUp.TeamUp,
            'notAttacking': NotAttacking.NotAttacking,
            'attackingSame': AttackingSame.AttackingSame
        }

    def initialize(self):
        # Register all behaviors based on the config
        for behaviour_name, behaviour_config in self.config['behaviours'].items():
            behaviour_class = self.behaviour_mapping.get(behaviour_name)
            if behaviour_class:
                behaviour = behaviour_class(behaviour_config, self.n_agents)
                self.behaviours.append(behaviour)
                self.reward_counts[_get_count_string(behaviour)] = 0
                self.rewards[_get_rewards_string(behaviour)] = 0

    def reset_heatmaps(self):
        for behaviour in self.behaviours:
            behaviour.reset_heatmap_registry()

    def get_behaviour_stats(self):
        behaviour_stats = {}
        for behaviour in self.behaviours:
            for k, v in behaviour.get_stats().items():
                behaviour_stats[behaviour.__class__.__name__ + "_" + k] = v

        for k, v in self.reward_counts.items():
            behaviour_stats[k] = v

        for k, v in self.rewards.items():
            behaviour_stats[k] = v

        return behaviour_stats

    def reset_episode(self):
        for behaviour in self.behaviours:
            behaviour.reset_episode()

    def reset_stats(self):
        for behaviour in self.behaviours:
            behaviour.reset_stats()

        for k, v in self.reward_counts.items():
            self.reward_counts[k] = 0

        for k, v in self.rewards.items():
            self.rewards[k] = 0

    def get_heatmaps(self):
        all_heatmaps = [heatmap for behaviour in self.behaviours for heatmap in behaviour.get_heatmaps()]
        summed_up_heatmap = HeatSC2map("cooperation_summary")

        for x, row in enumerate(summed_up_heatmap.data):
            for y, _ in enumerate(row):
                for h in all_heatmaps:
                    summed_up_heatmap.add_value(x, y, h.get_value(x, y))

        all_heatmaps.append(summed_up_heatmap)
        return all_heatmaps

    def evaluate_behaviors(self, env, actions, obs):
        behaviour_reward = 0

        for index, behaviour in enumerate(self.behaviours):
            reward = behaviour.evaluate(env, actions, obs)

            if reward != 0:
                self.reward_counts[_get_count_string(behaviour)] += 1
                self.rewards[_get_rewards_string(behaviour)] += reward

            if behaviour.enabled:
                behaviour_reward += reward

        return behaviour_reward
