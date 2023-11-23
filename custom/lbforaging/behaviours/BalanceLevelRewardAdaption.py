from custom.lbforaging.behaviours.TimeDependentBehaviour import TimeDependentBehaviour


class BalanceLevelRewardAdaption(TimeDependentBehaviour):
    def __init__(self, config, n_agents):
        super().__init__(config, n_agents)
        self.multiplier = config["multiplier"]

    def evaluate(self, env, actions, obs):
        return env.level_balanced_reward * self.multiplier
