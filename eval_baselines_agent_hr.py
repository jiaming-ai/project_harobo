# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os

from evaluator import OVMMEvaluator
from utils.config_utils import (
    create_agent_config,
    create_env_config,
    get_habitat_config,
    get_omega_config,
)

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot.agent.ovmm_agent.random_agent import RandomAgent
import sys
from pathlib import Path
sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)
from harobo.agent.ovmm.ovmm_agent import HaroboAgent

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_type",
        type=str,
        choices=["local", "local_vectorized", "remote"],
        default="local",
    )
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/harobo/configs/agent/harobo_agent_place_hr.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/env/hssd_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="harobo",
        choices=["baseline", "random", "harobo"],
        help="Agent to evaluate",
    )
    parser.add_argument(
        "overrides",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    # get habitat config
    habitat_config, _ = get_habitat_config(
        args.habitat_config_path, overrides=args.overrides
    )

    # get baseline config
    baseline_config = get_omega_config(args.baseline_config_path)

    # get env config
    env_config = get_omega_config(args.env_config_path)

    # merge habitat and env config to create env config
    env_config = create_env_config(
        habitat_config, env_config, evaluation_type=args.evaluation_type
    )

    # merge env config and baseline config to create agent config
    agent_config = create_agent_config(env_config, baseline_config)

    device_id = env_config.habitat.simulator.habitat_sim_v0.gpu_device_id

    # create agent
    if args.agent_type == "random":
        agent = RandomAgent(agent_config, device_id=device_id)
    elif args.agent_type == "baseline":
        agent = OpenVocabManipAgent(agent_config, device_id=device_id)
    elif args.agent_type == "harobo":
        agent = HaroboAgent(agent_config, device_id=device_id)
    else:
        raise NotImplementedError

    # create evaluator
    evaluator = OVMMEvaluator(env_config)

    # evaluate agent
    metrics = evaluator.evaluate(
        agent=agent,
        evaluation_type=args.evaluation_type,
        num_episodes=args.num_episodes,
    )
    print("Metrics:\n", metrics)
