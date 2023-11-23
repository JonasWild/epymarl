from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th

from custom.BehaviourRegistry import BehaviourRegistry
from custom.CustomStarCraftEnv import Metrics, Behaviours
from custom.CustomStarCraftEnv.Behaviours import StayTogether, TeamUp
from custom.MetricRegistry import MetricRegistry
from custom.custom_reward_learner import CustomRewardLearner
from custom.lbforaging.behaviours import BEHAVIOUR_MAP
from custom.lbforaging.metrics import METRIC_MAP
from custom.meta_replay_buffer import MetaReplayBuffer
from src.components.episode_buffer import ReplayBuffer
from src.runners.episode_runner import ReturnMetric


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger, ):
        self.meta_buffer = None
        self.new_meta_batch = None
        self.mac = None
        self.new_batch = None
        self.batch = None
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        metrics_map = None
        behaviour_map = None
        self.custom_env_args = None

        if "sc2" in self.args.env_args['key']:
            self.custom_env_args = self.args.env_args.get("custom_env_args")
            metrics_map = Metrics.METRIC_MAP
            behaviour_map = Behaviours.BEHAVIOUR_MAP
        elif "Foraging" in self.args.env_args['key']:
            self.custom_env_args = self.args.env_args.pop("custom_env_args")
            metrics_map = METRIC_MAP
            behaviour_map = BEHAVIOUR_MAP

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.reward_with_meta_learner = self.custom_env_args['reward_with_meta_learner']
        self.meta_learner_multiplier = self.custom_env_args['meta_learner_multiplier']

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg)),
                                                    BehaviourRegistry(
                                                        self.custom_env_args,
                                                        4,
                                                        behaviour_map
                                                    ),
                                                    MetricRegistry(
                                                        self.custom_env_args,
                                                        4,
                                                        metrics_map
                                                    ), self.reward_with_meta_learner,
                                                    self.meta_learner_multiplier))
                   for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

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

        self.log_train_stats_t = -100000

        self.train_meta_target_net_t = 0
        self.train_meta_target_net_interval = 10000
        self.meta_learner_loss_per_episode = []
        self.episode_meta_learner_predictions = []
        self.use_meta_target_net = self.custom_env_args['use_meta_target_net']
        self.meta_learner = CustomRewardLearner(self.env_info["state_shape"], 32, 1,
                                                use_target_network=self.use_meta_target_net)

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

        self.new_meta_batch = partial(ReplayBuffer, scheme, groups, self.batch_size, self.episode_limit + 1,
                                      preprocess=None, device=self.args.device)

        self.meta_buffer = ReplayBuffer(
            scheme,
            groups,
            32,
            self.episode_limit + 1,
            preprocess=preprocess,
            device="cpu",
        )

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        self.parent_conns[0].send(("save_replay", None))

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()
        self.meta_batch = self.new_meta_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)
        self.meta_batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def get_original_env(self):
        if "sc2" in self.args.env_args['key']:
            return self.ps
        elif "Foraging" in self.args.env_args['key']:
            return self.ps.original_env.env

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        episode_total_return = [0 for _ in range(self.batch_size)]
        episode_only_built_in_return = [0 for _ in range(self.batch_size)]
        episode_only_custom_return = [0 for _ in range(self.batch_size)]
        meta_learner_predictions = []

        stats_episode = []
        obs_episode = []
        actions_episode = []
        custom_scalars = {}

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated,
                                              test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.meta_batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env
                    if idx == 0 and test_mode and self.args.render:
                        parent_conn.send(("render", None))

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }

            meta_post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()

                    reward = data["reward"]

                    obs_for_meta_learner = np.array(data["state"]).astype(np.float32)
                    predicted_custom_reward = self.meta_learner.predict_reward(obs_for_meta_learner)
                    predicted_custom_reward = predicted_custom_reward.item() * (1 / self.meta_learner_multiplier)
                    meta_learner_predictions.append(predicted_custom_reward)

                    custom_reward = predicted_custom_reward if self.reward_with_meta_learner else data["custom_reward"]

                    episode_only_built_in_return[idx] += reward

                    if self.args.allow_custom_rewards:
                        episode_only_custom_return[idx] += custom_reward
                        reward += custom_reward

                    episode_total_return[idx] += reward

                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((reward,))
                    meta_post_transition_data["reward"].append((data["custom_reward"],))

                    episode_returns[idx] += reward
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))
                    meta_post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.meta_batch.update(meta_post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)
            self.meta_batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        self.episode_meta_learner_predictions.append(
            sum(meta_learner_predictions) / max(len(meta_learner_predictions), 1))
        if not test_mode:
            self.t_env += self.env_steps_this_run
            cur_stats = self.train_stats
            cur_total_returns = self.train_total_returns
            cur_only_custom_returns = self.train_only_custom_returns
            cur_only_built_in_returns = self.train_only_built_in_returns
            log_prefix = ""
        else:
            cur_stats = self.test_stats
            cur_total_returns = self.test_total_returns
            cur_only_custom_returns = self.test_only_custom_returns
            cur_only_built_in_returns = self.test_only_built_in_returns
            # self.metric_registry.evaluate_episode(self.get_original_env())
            log_prefix = "test_"

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_total_returns.extend(episode_total_return)
        cur_only_custom_returns.extend(episode_only_custom_return)
        cur_only_built_in_returns.extend(episode_only_built_in_return)

        return_metrics = [
            ReturnMetric("summed_", cur_total_returns),
            ReturnMetric("only_custom_", cur_only_custom_returns),
            ReturnMetric("only_built_in_", cur_only_built_in_returns),
        ]

        self.meta_buffer.insert_episode_batch(self.meta_batch)

        if self.meta_buffer.can_sample(self.batch_size):
            episode_sample = self.meta_buffer.sample(self.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            meta_learner_loss = self.meta_learner.train_model(episode_sample)
            self.meta_learner_loss_per_episode.append(meta_learner_loss)

        if self.use_meta_target_net and self.t_env - self.train_meta_target_net_t >= self.train_meta_target_net_interval:
            self.meta_learner.update_target_network()
            self.train_meta_target_net_t = self.t_env

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_total_returns) == n_test_runs):
            self._log(return_metrics, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(return_metrics, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, return_metrics: [ReturnMetric], stats, prefix):
        for return_metric in return_metrics:
            self.logger.log_stat(prefix + return_metric.name + "return_mean", np.mean(return_metric.returns),
                                 self.t_env)
            self.logger.log_stat(prefix + return_metric.name + "return_std", np.std(return_metric.returns), self.t_env)
            return_metric.returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_metric_registry", None))

        metric_registries = []
        for parent_conn in self.parent_conns:
            metric_registry = parent_conn.recv()
            metric_registries.append(metric_registry)

        metrics = {}
        for metric_registry in metric_registries:
            for k, v in metric_registry.get_metric_results().items():
                if k in metrics:
                    metrics[k] += v
                else:
                    metrics[k] = v

        for k, v in metrics.items():
            self.logger.log_stat("metric_" + k, float(v) / float(len(metric_registries)), self.t_env)

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_behaviour_registry", None))

        behaviour_registries = []
        for parent_conn in self.parent_conns:
            behaviour_registry = parent_conn.recv()
            behaviour_registries.append(behaviour_registry)

        behaviours_stats = {}
        mean_sum = 0
        for behaviour_registry in behaviour_registries:
            single_mean_sum = 0
            behaviour_stats = {} if behaviour_registry.get_behaviour_stats() is None else behaviour_registry.get_behaviour_stats().items()
            for k, v in behaviour_stats:
                if k in behaviours_stats:
                    behaviours_stats[k] += v
                else:
                    behaviours_stats[k] = v
                single_mean_sum += v

            behaviour_count = max(len(behaviour_stats), 1)
            mean_sum += float(single_mean_sum) / float(behaviour_count)
            behaviour_registry.reset_stats()

        self.logger.log_stat("custom_coop_summed_mean", mean_sum / len(behaviour_registries), self.t_env)
        for k, v in behaviours_stats.items():
            self.logger.log_stat("behaviour_" + k, float(v) / float(len(behaviour_registries)), self.t_env)

        self.logger.log_stat("meta_learner_prediction_mean",
                             sum(self.episode_meta_learner_predictions) / max(
                                 len(self.episode_meta_learner_predictions), 1),
                             self.t_env)
        self.episode_meta_learner_predictions.clear()

        self.logger.log_stat("meta_learner_loss_mean",
                             sum(self.meta_learner_loss_per_episode) / max(len(self.meta_learner_loss_per_episode), 1),
                             self.t_env)
        self.meta_learner_loss_per_episode.clear()


def env_worker(remote, env_fn, behaviour_registry, metric_registry, reward_with_meta_learner, meta_learner_multiplier):
    # Make environment
    env = env_fn.x()

    behaviour_registry.initialize()
    metric_registry.initialize()

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)

            # reward = reward * 2

            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()

            # Evaluate behaviors and metrics here
            custom_reward_from_behaviour = behaviour_registry.evaluate_behaviors(env.original_env.env, actions, obs)

            custom_reward_from_behaviour = custom_reward_from_behaviour * meta_learner_multiplier if reward_with_meta_learner else custom_reward_from_behaviour

            metric_registry.add_metrics_data(env.original_env.env, actions, obs)

            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "custom_reward": custom_reward_from_behaviour,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            metric_registry.evaluate_episode(env.original_env.env)

            env.reset()

            behaviour_registry.reset_episode()

            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats(), )
        elif cmd == "render":
            env.render()
        elif cmd == "save_replay":
            env.save_replay()
        elif cmd == "get_metric_registry":
            remote.send(metric_registry)
        elif cmd == "get_behaviour_registry":
            remote.send(behaviour_registry)
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
