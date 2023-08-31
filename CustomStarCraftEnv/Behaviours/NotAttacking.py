import sys

from CustomStarCraftEnv import CustomStarCraftEnv
from CustomStarCraftEnv.Behaviours.SequentialBehaviour import SequentialBehaviour
from myutils.myutils import get_zero_index_from_num_array


class NotAttacking(SequentialBehaviour):
    reset_code = -sys.maxsize - 1

    def __init__(self, config, n_agents):
        super().__init__(config, n_agents)

    def is_reset(self, prev, new):
        return new == NotAttacking.reset_code

    def evaluate(self, env: CustomStarCraftEnv, actions, obs):
        reward = 0
        previous_ally_units = env.previous_ally_units
        death_tracker_ally = env.death_tracker_ally

        delta_allies = [0] * len(env.agents.items())
        for index, (al_id, al_unit) in enumerate(env.agents.items()):
            prev_health = previous_ally_units[al_id].health + previous_ally_units[al_id].shield
            current_delta_ally = prev_health if al_unit.health == 0 else prev_health - al_unit.health - al_unit.shield
            delta_allies[index] = current_delta_ally

        index_only_ally_not_being_attacked = get_zero_index_from_num_array(delta_allies)

        all_except_index_dead = True
        for index, death in enumerate(death_tracker_ally):
            if index != index_only_ally_not_being_attacked and death == 0:
                all_except_index_dead = False

        if index_only_ally_not_being_attacked != -1 and not all_except_index_dead:
            if self.try_add_data(index_only_ally_not_being_attacked,
                                 round(env.agents[index_only_ally_not_being_attacked].pos.x),
                                 round(env.agents[index_only_ally_not_being_attacked].pos.y),
                                 self.reward
                                 ):
                reward += self.reward

        for index, (al_id, al_unit) in enumerate(env.agents.items()):
            if index is not index_only_ally_not_being_attacked:
                self.try_add_data(index, round(al_unit.pos.x),
                                  round(al_unit.pos.y),
                                  NotAttacking.reset_code)

        return reward
