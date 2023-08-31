from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np

import CustomStarCraftEnv.CustomStarCraftEnv
from CustomStarCraftEnv.Behaviours.BehaviourRegistry import BehaviourRegistry
from CustomStarCraftEnv.Behaviours.StayTogether import StayTogether
from CustomStarCraftEnv.Behaviours.TeamUp import TeamUp

from myutils.HeatSC2map import MyHeatmap
from myutils.myutils import pad_second_dim


class ReturnMetric:
    def __init__(self, name: str, returns: []):
        self.name = name
        self.returns = returns


class EpisodeRunner:
    heatmap_size = 100

    def init_behaviour_classes(self):
        StayTogether.min_delta = self.args.env_args["behaviour_stayTogether_min_delta"]
        StayTogether.enemy_in_distance = self.args.env_args["enemy_in_distance"]
        TeamUp.min_delta = self.args.env_args["behaviour_teamUp_min_delta"]
        TeamUp.enemy_in_distance = self.args.env_args["enemy_in_distance"]

        del self.args.env_args["behaviour_stayTogether_min_delta"]
        del self.args.env_args["behaviour_teamUp_min_delta"]
        del self.args.env_args["enemy_in_distance"]

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.init_behaviour_classes()

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.behaviour_registry = BehaviourRegistry(self.args.env_args["custom_env_args"], self.env.n_agents)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_total_returns = []
        self.test_total_returns = []
        self.train_only_custom_returns = []
        self.test_only_custom_returns = []
        self.train_only_built_in_returns = []
        self.test_only_built_in_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000
        self.log_heatmaps_t = 0

        self.t_env_test = 0

        self.obs_per_episode = []
        self.actions_per_episode = []

        self.n_obs = self.get_env_info()["obs_shape"]
        self.n_actions = self.get_env_info()["n_actions"]


    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

        if self.t_env == 0:
            self.behaviour_registry.initialize()

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_total_return = 0
        episode_only_built_in_return = 0
        episode_only_custom_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        self.behaviour_registry.reset_episode()

        stats_episode = []
        obs_episode = []
        actions_episode = []
        custom_scalars = {}
        while not terminated:

            obs = self.env.get_obs()
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [obs]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            obs_episode.append(obs)

            actions_sample = [[1 if action == i else 0 for i in range(self.n_actions)] for action in actions[0]]
            actions_episode.append(actions_sample)

            reward, terminated, env_info = self.env.step(actions[0])

            custom_reward = self.behaviour_registry.evaluate_behaviors(self.env, actions, obs)

            if test_mode and self.args.render:
                self.env.render()

            episode_only_built_in_return += reward
            episode_only_custom_return += custom_reward

            reward += custom_reward
            episode_total_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }

        self.obs_per_episode.append(obs_episode)
        self.actions_per_episode.append(actions_episode)

        if test_mode and self.args.render:
            print(f"Episode return: {episode_total_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_total_returns = self.test_total_returns if test_mode else self.train_total_returns
        cur_only_custom_returns = self.test_only_custom_returns if test_mode else self.train_only_custom_returns
        cur_only_built_in_returns = self.test_only_built_in_returns if test_mode else self.train_only_built_in_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
        else:
            self.t_env_test += self.t

        cur_total_returns.append(episode_total_return)
        cur_only_custom_returns.append(episode_only_custom_return)
        cur_only_built_in_returns.append(episode_only_built_in_return)

        return_metrics = [
            ReturnMetric("summed_", cur_total_returns),
            ReturnMetric("only_custom_", cur_only_custom_returns),
            ReturnMetric("only_built_in_", cur_only_built_in_returns),
        ]

        if test_mode and (len(self.test_total_returns) == self.args.test_nepisode):
            self._log(return_metrics, cur_stats, log_prefix, stats_episode, custom_scalars)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        if self.t_env - self.log_heatmaps_t >= self.args.runner_log_heatmaps_interval:
            heatmaps = self.behaviour_registry.get_heatmaps()

            padded_obs = pad_second_dim(self.obs_per_episode, [[0.0] * self.n_obs] * len(self.obs_per_episode[0][0]))
            padded_actions = pad_second_dim(self.actions_per_episode, [[0] * self.n_actions] * len(self.actions_per_episode[0][0]))

            obs_np = np.array(padded_obs)
            actions_np = np.array(padded_actions)

            # Sum along the 2nd and 3rd dimensions (indices 1 and 2)
            reduced_obs = np.mean(obs_np, axis=(0, 2))
            reduced_actions = np.mean(actions_np, axis=(0, 2))

            observation_heatmap = MyHeatmap("observations_over_time", self.n_obs,
                                            y_label="observation index")
            observation_heatmap.data = reduced_obs.tolist()

            action_heatmap = MyHeatmap("actions_taken_over_time", height=self.n_actions,
                                       y_label="action index")
            action_heatmap.data = reduced_actions.tolist()

            heatmaps.append(observation_heatmap)
            heatmaps.append(action_heatmap)

            self.logger.log_heatmaps(heatmaps)
            self.log_heatmaps_t = self.t_env

            self.behaviour_registry.reset_heatmaps()

            self.obs_per_episode = []
            self.actions_per_episode = []

            self.t_env_test = 0

        return self.batch

    def _log(self, return_metrics: [ReturnMetric], stats, prefix, stats_episode, custom_scalars):
        for return_metric in return_metrics:
            self.logger.log_stat(prefix + return_metric.name + "return_mean", np.mean(return_metric.returns),
                                 self.t_env)
            self.logger.log_stat(prefix + return_metric.name + "return_std", np.std(return_metric.returns), self.t_env)
            return_metric.returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)

        mean_sum = 0
        for k, v in self.behaviour_registry.get_behaviour_stats().items():
            mean_sum += v
            self.logger.log_stat(k, v / stats["n_episodes"], self.t_env)

        behaviour_count = len(self.behaviour_registry.get_behaviour_stats())
        self.logger.log_stat("custom_coop_summed_mean", mean_sum / behaviour_count, self.t_env)

        self.behaviour_registry.reset_stats()

        stats.clear()
