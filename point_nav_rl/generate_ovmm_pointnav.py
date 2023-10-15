import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from enum import IntEnum, auto

from home_robot.core.interfaces import DiscreteNavigationAction, Observations, ContinuousNavigationAction
import numpy as np
from omegaconf import DictConfig, OmegaConf
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)
from harobo.agent.ovmm.ovmm_agent import HaroboAgent as OVMMAgent
from utils.ovmm_env_visualizer import Visualizer
HOME_ROBOT_BASE_DIR = str(Path(__file__).resolve().parent.parent / "home-robot") + "/"

sys.path.insert(
    0,
    HOME_ROBOT_BASE_DIR + "src/home_robot"
)
sys.path.insert(
    0,
    HOME_ROBOT_BASE_DIR + "src/home_robot_sim"
)
import cv2
from habitat import make_dataset
from habitat.core.environments import get_env_class
from habitat.core.vector_env import VectorEnv
from habitat.utils.gym_definitions import _get_env_name
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer

# from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)
from typing import Optional, Tuple

from habitat_baselines.config.default import get_config as get_habitat_config
from omegaconf import DictConfig
from utils.viewer import OpenCVViewer
from utils.visualization import (
    display_grayscale,
    display_rgb,
    plot_image,
    save_image, 
    draw_top_down_map, 
    Recording, 
    visualize_gt,
    visualize_pred,
    save_img_tensor)
import torch
import random

import glob
import gzip
import json
import multiprocessing
import os
from os import path as osp

import tqdm

import habitat
from omegaconf import DictConfig, OmegaConf
from habitat.config.default import get_agent_config
from habitat.datasets.pointnav.pointnav_generator import (
    generate_pointnav_episode,
)
from habitat_baselines.config.default import get_config as get_habitat_config

NUM_EPISODES_PER_SCENE = int(5e4)
# Sample all scenes with a minimum quality
QUAL_THRESH = 2


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


CONFIG = "projects/harobo/configs/pointnav/pointnav_hssd.yaml"
SCENE_DIR = 'data/hssd-hab/scenes-uncluttered'
DATASET_DIR = 'data/datasets/pointnav/hssd/train/content'



def _generate_fn(sim,existing_episodes,eps_id_start):
    scene = sim.ep_info.scene_id
    print(f"Generating dataset for scene {scene}")
    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim, NUM_EPISODES_PER_SCENE, is_gen_shortest_path=False
        )
    )
    for i, ep in enumerate(dset.episodes):
        ep.scene_id = scene[len('data/hssd-hab/'):]
        ep.scene_dataset_config = existing_episodes.scene_dataset_config
        ep.additional_obj_config_paths = existing_episodes.additional_obj_config_paths
        ep.episode_id = eps_id_start + i

    scene_key = scene.split("/")[-1].split(".")[0]
    out_file = (
        f"{DATASET_DIR}/"
        f"{scene_key}.json.gz"
    )
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())



def generate_dataset(config,split='train'):
    
    OmegaConf.set_readonly(config, False)
    config.habitat.dataset.split = split
    OmegaConf.set_readonly(config, True)

    habitat_config = config.habitat
    dataset = make_dataset(habitat_config.dataset.type, config=habitat_config.dataset)

    # we select a subset of episodes to generate the dataset
    eps_select = {}
    for eps in dataset.episodes:
        scene_id = eps.scene_id
        if scene_id not in eps_select:
            eps_select[scene_id] = eps
    dataset.episodes = list(eps_select.values())

    # create env
    env_class_name = _get_env_name(config)
    env_class = get_env_class(env_class_name)
    habitat_env = env_class(config=habitat_config, dataset=dataset)
    habitat_env.seed(habitat_config.seed)
    env = HabitatOpenVocabManipEnv(habitat_env, config, dataset=dataset)
    
    # generate
    safe_mkdir(DATASET_DIR)
    for i in range(len(dataset.episodes)):
        env.reset()
        _generate_fn(env.habitat_env.env._env._env.sim,dataset.episodes[i],i*NUM_EPISODES_PER_SCENE)
        

    path = f"data/datasets/pointnav/hssd/train/train.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(dict(episodes=[]), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/harobo/configs/agent/hssd_eval_old.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=2,
        help="GPU id to use for evaluation",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Whether to render the environment or not",
        default=False,
    )
    parser.add_argument(
        "--no_interactive",
        action="store_true",
        help="Whether to render the environment or not",
        default=True,
    )
    parser.add_argument(
        "--eval_eps",
        help="evaluate a subset of episodes",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--split_dataset",
        help="evaluate a subset of episodes",
        type=int,
        nargs="+",
        default=None, #[0,100]
    )
    parser.add_argument(
        "--eval_eps_total_num",
        help="evaluate a subset of episodes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--collect_data",
        help="wheter to collect data for training",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--exp_name",
        help="experiment name",
        type=str,
        default='debug',
    )
    
    parser.add_argument(
        "--save_video",
        help="Save video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--eval_policy",
        help="policy to evaluate: fbe | rl | ur",
        type=str,
        default='ur',
    )
    parser.add_argument(
        "--seed",
        help="random seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--gt_semantic",
        help="whether to use ground truth semantic map",
        action="store_true",
        default=True,    
    )
    parser.add_argument(
        "--no_use_prob_map",
        help="whether to use probability map",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--skip_existing",
        help="whether to skip existing results",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--allow_sliding",
        help="whether to allow sliding",
        action="store_true",
        default=True,
    )
    
    print("Arguments:")
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

  
    print("Configs:")
    

    config = get_habitat_config(args.habitat_config_path, overrides=[])
    baseline_config = OmegaConf.load(args.baseline_config_path)
    extra_config = OmegaConf.from_cli(args.opts)
    baseline_config = OmegaConf.merge(baseline_config, extra_config)
    print(OmegaConf.to_yaml(baseline_config))
    config = DictConfig({**config, **baseline_config})

    generate_dataset(config,'train')
    generate_dataset(config,'val')

 




