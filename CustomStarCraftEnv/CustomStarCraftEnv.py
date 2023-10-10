from smacv2.smacv2.env.starcraft2.starcraft2 import StarCraft2Env
from myutils.HeatSC2map import HeatSC2map


class CustomStarCraftEnv(StarCraft2Env):
    save_replay_after_ep = 500

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialized = False
        self.save_replay_ep = -1

    def is_attack_action(self, action):
        return action >= self.n_actions - len(self.enemies.items())

    def reset(self, episode_config=None):
        self.save_replay_ep += 1
        if self.save_replay_ep == CustomStarCraftEnv.save_replay_after_ep:
            self.save_replay()
            self.save_replay_ep = -1

        if episode_config is None:
            episode_config = {}
        env = super().reset()

        if not self.initialized:
            self.initialized = True
            HeatSC2map.init_map_size(self.map_play_area_max.x, self.map_play_area_max.y,
                                     f"../assets/{self.map_params['bg_img']}")

        return env
