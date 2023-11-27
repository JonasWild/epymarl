import numpy as np
import os
import random
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th

from utils.logging import get_logger
import yaml

from run import run

# sc2test2vs1Marine
# env_config_filename = "gymmatest"
# env_config_filename = "sc2corridor_spawn_behind"
# env_config_filename = "sc2corridor_spawn_behind_6zealot_vs_12zerg"
# marl_alg = "qmix"
marl_alg = "mappo"



# default_config_filename = "default.yaml"
# = "custom.yaml"
# default_config_filename = "multiTest.yaml"

# from smaclite github: python3 src/main.py --config=mappo --env-config=gymma with seed=1 env_args.time_limit=120 env_args.key="smaclite:smaclite/MMM2-v0
# config = "--config=mappo"
# env_config = "--env-config=smaclite/{env}-v0"


# results_path = "/home/ubuntu/data"

def _get_marl_alg_parameter(marl_alg_name):
    return "--config=" + marl_alg_name


def _get_env_config_filename_parameter(env_config_filename):
    return "--env-config=" + env_config_filename


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)),
                  "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def _build_config_dict(p_params, p_default_config_filename, p_custom_env_config_filename):
    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", p_default_config_filename + ".yaml"), "r") as f:
        try:
            new_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    p_env_config = _get_config(p_params, "--env-config", "envs")

    if p_custom_env_config_filename is not None:
        # Get the defaults from default.yaml
        with open(os.path.join(os.path.dirname(__file__), "config", f"customEnv/{p_custom_env_config_filename}.yaml"),
                  "r") as f:
            try:
                p_custom_env_args = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "default.yaml error: {}".format(exc)
        p_env_config["env_args"]["custom_env_args"] = p_custom_env_args

    p_alg_config = _get_config(p_params, "--config", "algs")
    new_config_dict = recursive_dict_update(new_config_dict, p_env_config)
    new_config_dict = recursive_dict_update(new_config_dict, p_alg_config)

    return new_config_dict


def run_experiment(experiment_name, p_default_config_filename, p_env_config_filename, p_custom_env_config_filename):
    ex = Experiment(experiment_name)
    ex.logger = logger
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    params = [_get_marl_alg_parameter(marl_alg), _get_env_config_filename_parameter(p_env_config_filename)]
    config_dict = _build_config_dict(params, p_default_config_filename, p_custom_env_config_filename)

    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        map_name = config_dict["env_args"]["key"]

    # now add all the config to sacred
    ex.add_config(config_dict)

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, f"sacred/{config_dict['name']}/{map_name.replace(':', '_')}")

    ex.observers = []  # clear previous observersgym
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    @ex.main
    def my_main(_run, _config, _log):
        # Setting the random seed throughout the modules
        config = config_copy(_config)
        np.random.seed(config["seed"])
        th.manual_seed(config["seed"])
        config['env_args']['seed'] = config["seed"]

        # run the framework
        run(_run, config, _log)

    ex.run_commandline()


if __name__ == '__main__':
    SETTINGS['CAPTURE_MODE'] = "no"  # set to "no" if you want to see stdout/stderr in console
    logger = get_logger()
    # Disable DEBUG-level logging for PIL and Matplotlib

    results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

    th.set_num_threads(1)

    default_config_filenames = [
        # ("custom", "sc2smaller_corridor_spawn_behind_2zealot_vs_8zerg", "customEnvCorridor"),
        # ("custom", "sc2smaller_corridor_spawn_behind_2zealot_vs_8zerg", "customEnvCorridor3"),
        # ("custom", "sc2smaller_corridor_spawn_behind_2zealot_vs_8zerg", "customEnvCorridor1"),
        # ("custom", "sc2smaller_corridor_spawn_behind_2zealot_vs_8zerg", "customEnvCorridor2"),
        # ("custom", "sc2corridor", "customEnvDefault"),
        # ("custom", "gymmatest", "customEnvDefault"),epyma
        # ("custom", "sc2MetaTimeObj", "customEnvSplitGroups"),
        ("lbforagingbase", "lbforagingenv", "lbforagingcustom"),
        # ("lbforagingbase", "lbforagingenv", "lbforagingcustom1"),
        # ("lbforagingbase", "lbforagingenv", "lbforagingcustom2"),
        # ("lbforagingbase", "lbforagingenv", "lbforagingcustom3"),
        # ("lbforagingbase", "lbforagingenv", "lbforagingcustom4"),
        # ("lbforagingbase", "lbforagingenv", "lbforagingcustom5"),
        # ("lbforagingbase", "lbforagingenv", "lbforagingcustom6"),
    ]

    for index, (default_config_filename, env_config_filename, custom_env_config_filename) in enumerate(
            default_config_filenames):
        run_experiment(f"pymarl_{index}", default_config_filename, env_config_filename, custom_env_config_filename)
