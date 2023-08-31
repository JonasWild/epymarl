from smacv2.smacv2.env.starcraft2.starcraft2 import StarCraft2Env
from myutils.HeatSC2map import HeatSC2map


class CustomStarCraftEnv(StarCraft2Env):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialized = False

    def is_attack_action(self, action):
        return action >= self.n_actions - len(self.enemies.items())

    def reset(self, episode_config=None):
        if episode_config is None:
            episode_config = {}
        env = super().reset()

        if not self.initialized:
            self.initialized = True
            HeatSC2map.init_map_size(self.map_play_area_max.x, self.map_play_area_max.y, "../assets/bg_corridor.png")

        return env
