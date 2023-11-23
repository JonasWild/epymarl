from itertools import chain

from custom.CustomStarCraftEnv import CustomStarCraftEnv
from custom.CustomStarCraftEnv.Behaviours.SequentialBehaviour import SequentialBehaviour
from custom.myutils.myutils import find_same_indices


class AttackingSame(SequentialBehaviour):
    def __init__(self, config, n_agents):
        super().__init__(config, n_agents)

    def is_reset(self, prev, new):
        return new == AttackingSame.reset_code

    def evaluate(self, env: CustomStarCraftEnv, actions, obs):
        reward = 0

        agent_ids = []
        for action, same_attacking_agents_indices in find_same_indices(actions.squeeze().tolist()).items():
            if env.is_attack_action(action):
                agent_ids.append(same_attacking_agents_indices)

        same_attacking_agents = list(chain.from_iterable(agent_ids))

        for i in range(self.n_agents):
            if i in same_attacking_agents:
                if self.try_add_data(i, round(env.agents[i].pos.x), round(env.agents[i].pos.y), self.reward):
                    reward += self.reward
            else:
                self.try_add_data(i, round(env.agents[i].pos.x), round(env.agents[i].pos.y),
                                  SequentialBehaviour.reset_code)

        return reward
