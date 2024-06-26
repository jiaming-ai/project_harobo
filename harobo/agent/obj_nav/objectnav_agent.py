
from typing import Any, Dict, List, Tuple
from collections import deque
import numpy as np
from skimage.measure import find_contours
import random
import time
import torch
from torch.nn import DataParallel
import home_robot.utils.pose as pu
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations, ContinuousNavigationAction
from harobo.mapping.polo.polo_map_state import POLoMapState, dialate_tensor
from harobo.mapping.polo.constants import MapConstants as MC
from harobo.mapping.polo.polo_map_module import POLoMapModule
import home_robot.mapping.map_utils as mu
import numpy as np
import torch
from utils.visualization import (
    display_grayscale,
    display_rgb,
    plot_image,
    save_image, 
    draw_top_down_map, 
    Recording, 
    visualize_gt,
    render_plt_image,
    visualize_pred,
    show_points, 
    show_points_with_prob,
    show_voxel_with_prob, 
    show_voxel_with_logit,
    save_img_tensor)    
from harobo.navigation_planner.fmm_planner import FMMPlanner
from harobo.navigation_planner.discrete_planner import add_boundary, remove_boundary, DiscretePlanner
from harobo.igp_net.predictor import IGPredictor
import enum
from .pointnav import PNAgent

# For visualizing exploration issues
debug_frontier_map = False
debug_info_gain = False


class STATES(enum.Enum):
    SEARCHING = 0 
    CHECKING = 1
    GOING_TO_GOAL = 2 

class ObjectNavAgent(Agent):

    # Flag for debugging data flow and task configuraiton
    verbose = False

    def __init__(self, config, device_id: int = 0, **kwargs):
        self.max_steps = config.AGENT.max_steps
        self.num_environments = config.NUM_ENVIRONMENTS
        self.config = config

        # self._module = ObjectNavAgentModule(config)
        self._module = self.init_map(config)

        if config.NO_GPU:
            self.device = torch.device("cpu")
            self.module = self._module
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
            self._module = self._module.to(self.device)
            # Use DataParallel only as a wrapper to move model inputs to GPU
            self.module = DataParallel(self._module, device_ids=[self.device_id])

        self.visualize = config.VISUALIZE or config.PRINT_IMAGES or kwargs.get('visualize',False)
        self.render_igp = True
        self.semantic_map = POLoMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            probability_prior=config.AGENT.SEMANTIC_MAP.probability_prior,
        )

        self.map_size_parameters = mu.MapSizeParameters(
            resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
        )
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
            
        self.pn_agent = PNAgent(self.device)
        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.goal_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=False #config.AGENT.PLANNER.discrete_actions,
        )
        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories, device=self.device
        )

        self.timesteps = None
        self.timesteps_before_goal_update = None
        self.episode_panorama_start_steps = None
        self.last_poses = None
        self.verbose = config.AGENT.PLANNER.verbose

        self.probability_prior=config.AGENT.SEMANTIC_MAP.probability_prior

        self.num_high_goals = 1
        self.max_ur_goal_dist_change = config.AGENT.IG_PLANNER.max_ur_goal_dist_change_cm / config.AGENT.SEMANTIC_MAP.map_resolution
        self.ur_dist_reach_threshold = config.AGENT.IG_PLANNER.ur_dist_reach_threshold_cm / config.AGENT.SEMANTIC_MAP.map_resolution
        self.agent_cell_radius = agent_cell_radius
        self.total_look_around_steps = config.AGENT.IG_PLANNER.total_look_around_steps
        self.turn_angle_rad = config.AGENT.IG_PLANNER.turn_angle_rad
        self.goal_update_steps = config.AGENT.IG_PLANNER.goal_update_steps
        self.max_num_changed_cells = config.AGENT.IG_PLANNER.max_num_changed_cells
        self.max_num_changed_promising_cells = config.AGENT.IG_PLANNER.max_num_changed_promising_cells
        self.info_gain_alpha = config.AGENT.IG_PLANNER.info_gain_alpha
        self.ur_obstacle_dialate_radius = config.AGENT.IG_PLANNER.ur_obstacle_dialate_radius
        self.ur_dist_obstacle_dialate_radius = config.AGENT.IG_PLANNER.ur_obstacle_dialate_radius
        self.init_util_lambda = config.AGENT.IG_PLANNER.util_lambda
        self.random_ur_goal = config.AGENT.IG_PLANNER.random_ur_goal

        """
        UR Exploration:
            0: go to ur goal (look around point)
            1: look around
        
        Point Goal Navigation:
            2: object identified, go to object
        
        """
        self._state = [STATES.CHECKING for _ in range(self.num_environments)]
        self._ur_goal_dist = np.zeros(self.num_environments)
        self._global_hgoal_pose = torch.zeros((self.num_environments,self.num_high_goals, 2), device=self.device) # (num_envs, num_goals, 2)
        # self._ur_local_goal_coords = torch.zeros((self.num_environments,self.num_high_goals, 2), device=self.device) # (num_envs, num_goals, 2)
        self._force_goal_update_once = np.full(self.num_environments, False)
        self._look_around_steps = np.zeros(self.num_environments)
        self._num_explored_grids = np.zeros(self.num_environments)
        self._num_promising_grids = np.zeros(self.num_environments)
        self._util_lambda = config.AGENT.IG_PLANNER.util_lambda # we can decay this over time
        

        igp_model_dir = config.AGENT.IG_PLANNER.igp_model_dir
        self.utility_exp = config.AGENT.IG_PLANNER.utility_exp
        self.ig_predictor = IGPredictor(igp_model_dir,self.device)

        self._pre_num_obstacles = np.zeros(self.num_environments,dtype=int) # to track if map has changed
        self._map_changed = np.zeros(self.num_environments,dtype=bool) # to track if map has changed
        self._found_goal = np.zeros(self.num_environments,dtype=bool) # to track if map has changed

        self._hgoal_stuck_count = np.zeros(self.num_environments,dtype=int) # num of times we have been stuck for the same high goal
        self._last_action = [None for _ in range(self.num_environments)]
        self._last_pose = [None for _ in range(self.num_environments)]
        self._max_obj_detection_scores = np.zeros(self.num_environments) # to track the highest object detection score

        self._to_end_rec = np.zeros(self.num_environments,dtype=bool) # to track if searching for the end receptacle
        self._end_rec_ins_idx = np.zeros(self.num_environments,dtype=int) # to track the instance index of the current goal end receptacle
        self.end_rec_mask_percent_threshold = 1/12
        
        self.use_instance_based_goal_rec = config.AGENT.IG_PLANNER.use_instance_based_goal_rec
        self._terminate_list = [False for _ in range(self.num_environments)]
        self._end_rec_view_point = [None for _ in range(self.num_environments)]
        
    def force_update_high_goal(self,e):
        self._force_goal_update_once[e] = True


    def init_map(self,config):
    
        polo_map_module = POLoMapModule(
            frame_height=config.ENVIRONMENT.frame_height,
            frame_width=config.ENVIRONMENT.frame_width,
            camera_height=config.ENVIRONMENT.camera_height,
            hfov=config.ENVIRONMENT.hfov,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
            min_depth=config.ENVIRONMENT.min_depth,
            max_depth=config.ENVIRONMENT.max_depth,
            explored_radius=config.AGENT.SEMANTIC_MAP.explored_radius,
            been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            cat_pred_threshold=config.AGENT.SEMANTIC_MAP.cat_pred_threshold,
            exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
            map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
            must_explore_close=config.AGENT.SEMANTIC_MAP.must_explore_close,
            min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
            dilate_obstacles=config.AGENT.SEMANTIC_MAP.dilate_obstacles,
            dilate_size=config.AGENT.SEMANTIC_MAP.dilate_size,
            dilate_iter=config.AGENT.SEMANTIC_MAP.dilate_iter,
            probabilistic=config.AGENT.SEMANTIC_MAP.use_probability_map,
            probability_prior=config.AGENT.SEMANTIC_MAP.probability_prior,
            close_range=config.AGENT.SEMANTIC_MAP.close_range,
            confident_threshold=config.AGENT.SEMANTIC_MAP.confident_threshold,
        )
        return polo_map_module

    # ------------------------------------------------------------------
    # macro actions
    # ALL macro actions must be responsible to update the state
    # ------------------------------------------------------------------
    def _look_around(self,e):
        action = ContinuousNavigationAction(np.array([0.,0.,self.turn_angle_rad]))
        # action = DiscreteNavigationAction.TURN_RIGHT
        # self.timesteps_before_goal_update[0] += 1 # make sure we don't update the goal during look around
        self._look_around_steps[e] += 1
        # we must change state directly here, otherwise the urp planner will not be called
        if self._look_around_steps[e] >= self.total_look_around_steps:
            self._state[e] = STATES.SEARCHING # go to ur goal
            self._look_around_steps[e] = 0 # reset look around steps
            self.timesteps_before_goal_update[e] = 0 # relan for the next ur goal
            # avoid checking the same area again
            self._mark_hgoal_unreachable(e,10)

        return action
    
    def _plan_hgoal(self,e):
        """ plan a high goal for the agent to go to and check using polo
        """
        pred_ig, ig_vis = self._compute_info_gains_igp(e)
        utility_map,local_goal_coords,dist = self._select_goal_igp(e, pred_ig)
        self._ur_goal_dist[e] = dist
        self._global_hgoal_pose[e] = self.semantic_map.local_map_coords_to_hab_world_frame(e, local_goal_coords)
        goal_map_e = self._get_goal_map(local_goal_coords)

        if self.visualize and self.render_igp:
            ig_vis['utility'] = render_plt_image(utility_map)

        # self._state[e] = STATES.SEARCHING # go to the ur goal
        self.semantic_map.set_goal_for_env(e, goal_map_e)
        self.semantic_map.set_global_goal_map(e, goal_map_e)

        self._hgoal_stuck_count[e] = 0

        return ig_vis

    def _get_planner_inputs(self,e):
        planner_inputs = {
                "global_obstacle_map": self.semantic_map.get_global_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": self._found_goal[e].item(),
                "global_goal_pose": self._global_hgoal_pose[e].cpu().numpy(),
                "map_changed": self._map_changed[e],
            }
        return planner_inputs
    def _call_low_level_planner(self,e):
        """ call the low level planner to go to the high goal
        """
        planner_inputs = self._get_planner_inputs(e)
        planner_outputs = self.planner.plan(
                **planner_inputs,
                timestep=self.timesteps[e],
                debug=self.verbose,
            )
        return planner_inputs, planner_outputs
    
    def _update_hgoal_map(self,e):
        self.semantic_map.update_goal_for_env(e)
        # if self._state[e] == STATES.GOING_TO_GOAL:
        #     if self._to_end_rec[e]:
        #         if self.use_instance_based_goal_rec:
        #             goal_map = self._get_end_rec_goal_map_ins(e,3)
        #         else:
        #             goal_map = self._get_end_rec_goal_simple(e,3)
        #     else:
        #         # goal_map = self._detect_object_goal_simple(e,1,2)
        #         # fine grined control over which object to go to
        #         goal_coords = self.semantic_map.hab_world_to_map_local_frame(e, self._global_hgoal_pose[e])
        #         goal_map = self._get_goal_map(goal_coords)
        # else:
        #     goal_coords = self.semantic_map.hab_world_to_map_local_frame(e, self._global_hgoal_pose[e])
        #     goal_map = self._get_goal_map(goal_coords)

        # self.semantic_map.set_goal_for_env(e, goal_map)

    def _get_object_goal_ins(self,e,goal_idx,rec_idx=None):
        """
        Instance based object goal selection
        """
        goal_map = self.semantic_map.local_map[e, MC.NON_SEM_CHANNELS+goal_idx].clone()
        goal_map = dialate_tensor(goal_map.unsqueeze(0))[0]
        goal_map = goal_map.cpu().numpy()

        if goal_map.sum() == 0:
            return None

        # we need to cluster the goal map into different objects
        # and find the corresponding bounding boxes for the dilated goal map
        
        # we need to identify each of the possible goals
        contours = find_contours(goal_map,0.5)
        # find the bounding box of each contour
        
        if len(contours) == 0:
            return None
        
        bboxs = []
        for contour in contours:
            Xmin = np.rint(np.min(contour[:,0])).astype(int)
            Xmax = np.rint(np.max(contour[:,0])).astype(int)
            Ymin = np.rint(np.min(contour[:,1])).astype(int)
            Ymax = np.rint(np.max(contour[:,1])).astype(int)
            
            bboxs.append([Xmin, Xmax, Ymin, Ymax])
        

        # then we calculate the object scores for each identified object
        if rec_idx is not None:
            rec_map = self.semantic_map.local_map[e, MC.NON_SEM_CHANNELS+rec_idx]>0.3
            goal_map *= rec_map.cpu().numpy()


        # find the object with highest scores
        obj_scores = []
        for bbox in bboxs:
            obj_map = goal_map[bbox[0]:bbox[1],bbox[2]:bbox[3]]
            # obj_size = obj_map.numel()
            obj_prob = obj_map.sum()
            obj_scores.append(obj_prob)

        visualize=False
        if visualize:
            import matplotlib.pyplot as plt
            plt.imshow(goal_map)
            for bbox in bboxs:
                plt.plot([bbox[2],bbox[2],bbox[3],bbox[3],bbox[2]],[bbox[0],bbox[1],bbox[1],bbox[0],bbox[0]])
            plt.show()

        obj_scores = np.array(obj_scores)
        obj_idx = np.argmax(obj_scores)

        if obj_scores[obj_idx] > 0:
            obj_bbox = bboxs[obj_idx]
            goal_map_chosen = np.zeros_like(goal_map)
            goal_map_chosen[obj_bbox[0]:obj_bbox[1],obj_bbox[2]:obj_bbox[3]] = \
                (goal_map[obj_bbox[0]:obj_bbox[1],obj_bbox[2]:obj_bbox[3]]>0)
            goal_map = goal_map_chosen
        else:
            goal_map = None

        return goal_map

    def _get_object_goal_simple(self,e,goal_idx,rec_idx=None):
        
        obj_map_ori = self.semantic_map.local_map[e, MC.NON_SEM_CHANNELS+goal_idx]
        goal_map = dialate_tensor(obj_map_ori.unsqueeze(0))[0]
        goal_map = goal_map.cpu().numpy()

        rec_map = self.semantic_map.local_map[e, MC.NON_SEM_CHANNELS+rec_idx]>0.3
        goal_map *= rec_map.cpu().numpy()

        if goal_map.sum() > 0:
            goal_map = goal_map > 0
        else:
            goal_map = None

        return goal_map

    def _get_end_rec_goal_map_ins(self,e,end_recep_goal_idx):
        """
        select the best end receptacle for placement globally
        1. find the end receptacle that is reachable
        2. for each of the reachable end receptacle, find the instance with the highest score
        3. check if there is a flat slab on the end receptacle

        if there is a suitable placement, we need to first go close to the end receptacle, and re-generate the point cloud
        since the acturator is noisy
        """
        if len(self.semantic_map.global_instances[e][end_recep_goal_idx]) == 0:
            return None
        
        reachable_end_rec = torch.logical_and(self.semantic_map.global_map[e,MC.NON_SEM_CHANNELS+end_recep_goal_idx]>0.5, \
            ~self.semantic_map.hgoal_unreachable[e])
        lmb = self.semantic_map.lmb[e]
        local_end_rec_map = reachable_end_rec[lmb[0]:lmb[1],lmb[2]:lmb[3]]

        if local_end_rec_map.sum() <= 30:
            return None
        
        polo_reachable = self._compute_reachable_area_to_end_rec(e)
        local_polo_reachable = polo_reachable[lmb[0]:lmb[1],lmb[2]:lmb[3]]
        
        # dialate to find
        MAX_PLACEMENT_DIST = 10
        OBS_DIST = 4 # slightly farther so that the object detection can identify the object 
        
        obs_map = self.semantic_map.local_map[e,MC.OBSTACLE_MAP].unsqueeze(0)
        for _ in range(OBS_DIST):
            obs_map = dialate_tensor(obs_map)
        obs_map = obs_map[0] > 0

        polo_no_obs = torch.logical_and(local_polo_reachable>1,~obs_map)
        # dist_map = self._get_dist_map(e)
        # dist_unreachable = dist_map == dist_map.max()

        # end_rec_ins_all = self.semantic_map.global_instances[e][end_recep_goal_idx]
        end_rec_ins_all = self.semantic_map.get_ins(e,end_recep_goal_idx)
        for ins_idx, ins in enumerate(end_rec_ins_all):
            
            # first we find the surrounding area of the end receptacle
            bb = ins['bb']
            bb = [bb[0]-lmb[0],bb[1]-lmb[0],bb[2]-lmb[2],bb[3]-lmb[2]]
            end_rec_map_ins = torch.zeros_like(obs_map).float()
            end_rec_map_ins[bb[0]:bb[1],bb[2]:bb[3]] = local_end_rec_map[bb[0]:bb[1],bb[2]:bb[3]]
            
            # if the receptacle is too small
            if end_rec_map_ins.sum() <= 20:
                print(f'end rec {ins_idx} is too small, continue')
                continue
            
            end_rec_map_ins_dia = end_rec_map_ins.clone().unsqueeze(0)
            for _ in range(MAX_PLACEMENT_DIST):
                end_rec_map_ins_dia = dialate_tensor(end_rec_map_ins_dia)
            end_rec_map_ins_dia = end_rec_map_ins_dia[0]
            
            # check if 1) navigable 2) not blocked by obstacles
            reachable_area_ins = torch.logical_and(end_rec_map_ins_dia>0.5, polo_no_obs)
            
            # mark to avoid rechecking
            if reachable_area_ins.sum()<5:
                print(f'end rec {ins_idx} is unreachable, mark as unreachable and continue')
                self.semantic_map.mark_end_rec_ins_unreachable(e,ins_idx)
                continue

            reachable_poses = torch.nonzero(reachable_area_ins) # local map coords
            reachable_poses_global_pose = self.semantic_map.local_map_coords_to_hab_world_frame(e, reachable_poses)

            # TODO: filter out pose that are not navigable
            
            # select pose with highest polo score
            polo_scores = local_polo_reachable[reachable_poses[:,0],reachable_poses[:,1]]
            polo_idx = torch.argmax(polo_scores)
            
            self.place_policy.reset()
            self.place_policy.set_rec_points([ins['pc']])
            agent_pose = reachable_poses_global_pose[polo_idx]
            placement = self.place_policy.get_receptacle_placement_point(agent_pose)

            if placement is not None:
                goal_map = np.zeros_like(self.semantic_map.goal_map[e])
                goal_map[reachable_poses[polo_idx,0],reachable_poses[polo_idx,1]] = 1
                self._end_rec_ins_idx[e] = ins_idx

                # view_point = torch.from_numpy(ins['view_point'][:2]).unsqueeze(0).to(self.device)
                # view_point_local_map = self.semantic_map.hab_world_to_map_local_frame(e, view_point).cpu().numpy()
                # goal_map[view_point_local_map[0,0],view_point_local_map[0,1]] = 1
                self._end_rec_view_point[e] = ins['view_point']
                return goal_map
            
            # # sample some points on the reachable area
            # sampled_idx = random.sample(range(reachable_poses.shape[0]),min(10,reachable_poses.shape[0]))
            # for point_idx in sampled_idx:
            
            #     self.place_policy.reset()
            #     self.place_policy.set_rec_points([ins['pc']])
            #     agent_pose = reachable_poses_global_pose[point_idx]
            #     placement = self.place_policy.get_receptacle_placement_point(agent_pose)

            #     if placement is not None:
            #         return reachable_area_ins.cpu().numpy()

            else: 
                print(f'end rec {ins_idx} has no good placement surface, mark as unreachable and continue')
                self.semantic_map.mark_end_rec_ins_unreachable(e,ins_idx)
            
        return None
    # def _get_end_rec_goal_map_ins(self,e,end_recep_goal_idx):
    #     """
    #     Instance based goal selection
    #     find a end rec that has not been marked as unreachable
    #     """
    #     end_rec_ins = self.semantic_map.global_instances[e][end_recep_goal_idx]
    #     reachable_end_rec = torch.logical_and(self.semantic_map.global_map[e,MC.NON_SEM_CHANNELS+end_recep_goal_idx]>0.5, \
    #         ~self.semantic_map.hgoal_unreachable[e])
    #     for idx, ins in enumerate(end_rec_ins):
    #         bb = ins['bb']
    #         end_rec_ins = reachable_end_rec[bb[0]:bb[1],bb[2]:bb[3]]
    #         if end_rec_ins.sum() > 30:
    #             lmb = self.semantic_map.lmb[e]
    #             global_goal = torch.zeros_like(self.semantic_map.global_map[e,MC.NON_SEM_CHANNELS+end_recep_goal_idx])
    #             global_goal[bb[0]:bb[1],bb[2]:bb[3]] = end_rec_ins
    #             goal_map_local = global_goal[lmb[0]:lmb[1],lmb[2]:lmb[3]].cpu().numpy()
    #             self._end_rec_ins_idx[e] = idx
    #             print(f'found end rec goal, id: {end_recep_goal_idx}, score: {ins["score"]}')
    #             return goal_map_local
    #     return None
        
    
    def _get_end_rec_goal_simple(self,e,end_recep_goal_idx):
        obj_map_ori = self.semantic_map.local_map[e, MC.NON_SEM_CHANNELS+end_recep_goal_idx]
        goal_map = obj_map_ori.cpu().numpy()>0.5

        return goal_map
    
    def _detect_goal(self,e,obs=None,force_detect=False):
        """
        Detect the goal object and set found goal
        """

        # avoid detecting the same goal again
        if self._found_goal[e]:
            return None
        
        goal_obj_idx,start_rec_idx,end_rec_idx = 1,2,3
        
        goal_map = None
        if self._to_end_rec[e]:
            if self.use_instance_based_goal_rec:
                # filter for speed up
                if force_detect or (obs.semantic == obs.task_observations["end_recep_goal"]).any():
                    goal_map = self._get_end_rec_goal_map_ins(e,end_rec_idx)
            else:
                goal_map = self._get_end_rec_goal_simple(e,end_rec_idx)
        
        else:
            if force_detect or (obs.semantic == obs.task_observations["object_goal"]).any() or \
                (obs.semantic == obs.task_observations["start_recep_goal"]).any() :
                goal_map = self._get_object_goal_ins(e,goal_obj_idx,start_rec_idx)

        if goal_map is not None and goal_map.sum() > 0:
            self._found_goal[e] = 1 

            self._set_object_goal(e,goal_map)
        
        return goal_map
      
    def _set_object_goal(self,e,object_goal_map):

        self.semantic_map.set_goal_for_env(e, object_goal_map)
        self.semantic_map.set_global_goal_map(e, object_goal_map)

        # calculate the global goal pose
        xs, ys = object_goal_map.nonzero()
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
        pos = torch.tensor([xc, yc], device=self.device).unsqueeze(0)
        self._global_hgoal_pose[e] = self.semantic_map.local_map_coords_to_hab_world_frame(e, pos)

        self._state[e] = STATES.GOING_TO_GOAL # go to object goal
        self._hgoal_stuck_count[e] = 0

        if self._to_end_rec[0]:
            self.planner.reset_for_rec()

    def _mark_hgoal_unreachable(self,e,mark_radius=5):
        hgoal_global_coords = self.semantic_map.hab_world_to_map_global_frame(e, self._global_hgoal_pose[e])
        x1 = max(0, hgoal_global_coords[e,0] - mark_radius)
        x2 = min(self.semantic_map.global_map.shape[2], hgoal_global_coords[e,0] + mark_radius)
        y1 = max(0, hgoal_global_coords[e,1] - mark_radius)
        y2 = min(self.semantic_map.global_map.shape[3], hgoal_global_coords[e,1] + mark_radius)

        self.semantic_map.hgoal_unreachable[e,x1:x2,y1:y2] = True

        # lmb = self.semantic_map.lmb
        # self.semantic_map.local_map[e,MC.OBSTACLE_MAP] = \
        #     self.semantic_map.global_map[e,MC.OBSTACLE_MAP,lmb[e,0]:lmb[e,1], lmb[e,2]:lmb[e,3]]
                                                            
    def _reset_end_rec_goal(self,e):

        # mark the current end rec as unreachable
        self.semantic_map.mark_end_rec_ins_unreachable(e,self._end_rec_ins_idx[e])
        self._found_goal[e] = 0
        self.place_policy.reset()
        
        # find the next end rec
        goal_map = self._detect_goal(e,force_detect=True)
        
        self._hgoal_stuck_count[e] = 0
        
        return goal_map
    # ------------------------------------------------------------------
    
    def change_to_rec(self,e, obs):
        """
        Start to navigate to the end receptacle
        """
        self._found_goal[e] = 0
        self._to_end_rec[e] = True

        # clear necessary maps
        self.semantic_map.clear_prob_maps(e)
        self.semantic_map.clear_hgoal_unreachable(e)
        
        self._state[e] = STATES.SEARCHING

        # detect using current map
        object_goal_map = self._detect_goal(e, obs, force_detect=True)

        # reset hgoal states
        self._hgoal_stuck_count[e] = 0
        self.semantic_map.hgoal_unreachable[e] = 0
        
        # reset planner states
        self.planner.reset_for_rec()
        
    @torch.no_grad()
    def act(
        self,obs
    ) -> Tuple[List[dict], List[dict]]:
        """ 
        This function assumes that the agent has been initialized.
        """
        # t0 = time.time()
        # NOTE: input for _preprocess_obs is not a vector 
        (
            obs_processed,
            pose_delta,
            object_goal_category,
            start_recep_goal_category,
            end_recep_goal_category,
            goal_name,
            camera_pose,
            detection_results,
        ) = self._preprocess_obs(obs)

        # t1 = time.time()
        # print(f"[Agent] Obs preprocessing time: {t1 - t0:.2f}")

        # -----------------------------------------------
        # update map

        # Update map with observations and generate map features
        agent_pose = np.concatenate([obs.gps,obs.compass])
        updated_local_map, updated_local_pose, instances = self.module(
            obs_processed,
            pose_delta,
            camera_pose,
            self.semantic_map.local_map,
            self.semantic_map.local_pose,
            detection_results,
            self.semantic_map.lmb,
            agent_pose,
        )

        for e in range(self.num_environments):
            lmb = self.semantic_map.lmb
            self.semantic_map.global_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]] = updated_local_map[e]
            self.semantic_map.global_pose[e] = updated_local_pose[e] + self.semantic_map.origins[e]
            mu.recenter_local_map_and_pose_for_env(
                e,
                self.semantic_map.local_map,
                self.semantic_map.global_map,
                self.semantic_map.local_pose,
                self.semantic_map.global_pose,
                lmb,
                self.semantic_map.origins,
                self.map_size_parameters,
            )

        # NOTE: this is not vectorized
        if instances is not None:
           
            self.semantic_map.update_instances(0, instances,
                                               obs.task_observations["object_goal"], 
                                               obs.task_observations["start_recep_goal"], 
                                               obs.task_observations["end_recep_goal"],
                                               )
            
            # # if new end rec is detected, we need to update the end rec goal index
            # if len(instances[3]) > 0:
            #     self._end_rec_ins_idx[0] = 0

            visualize=False
            if visualize:
                import matplotlib.pyplot as plt
                # plot the object goal, start receptacle goal, end receptacle goal
                for i in range(1,4):
                    ax = plt.subplot(1,3,i)
                    ax.imshow(self.semantic_map.global_map[0,MC.NON_SEM_CHANNELS+i].cpu().numpy())
                    for ins in self.semantic_map.global_instances[0][i]:
                        bbox = ins['bb']
                        ax.plot([bbox[2],bbox[2],bbox[3],bbox[3],bbox[2]],[bbox[0],bbox[1],bbox[1],bbox[0],bbox[0]])
                plt.show()
                
               
        # t2 = time.time()
        # print(f"[Agent] Semantic mapping time: {t2 - t1:.2f}")

        # detect if an goal object is identified, 
        # to speed up, we only call the detection if a new object with higher scores is detected
        # trigger state change if object goal is found
        for e in range(self.num_environments):
            self._detect_goal(e,obs)

        # detect if map has changed
        obs_num = self.semantic_map.global_map[:, MC.OBSTACLE_MAP].sum((1,2)).int().cpu().numpy()
        self._map_changed = obs_num != self._pre_num_obstacles
        self._pre_num_obstacles = obs_num

        # detect collision
        curr_pose = self.semantic_map.global_pose[:2].cpu().numpy()

        for e in range(self.num_environments):
            last_action = self._last_action[e]
            if last_action == DiscreteNavigationAction.MOVE_FORWARD or (
                type(last_action) == ContinuousNavigationAction
                and np.linalg.norm(last_action.xyt[:2]) > 0
            ):
                travel_dist = np.linalg.norm(self._last_pose[e] - curr_pose[e])
                if travel_dist < 0.15:
                    self._hgoal_stuck_count += 1
            self._last_pose[e] = curr_pose[e]
            
        # step
        ig_vis_list = [None for e in range(self.num_environments)]
        lc_input_list = [{} for _ in range(self.num_environments)]
        lc_output_list = [{} for _ in range(self.num_environments)]
        for e in range(self.num_environments):
            """
            UR Exploration:
                0: go to ur goal (look around point)
                1: look around
            
            Point Goal Navigation:
                2: object identified, go to object
            
            """
            if self._state[e] == STATES.SEARCHING:
                # first check if we need to update the high goal
                need_update = self._check_n_update_need_replan_ur_goal(e)
                
                hgoal_stuck_too_much = self._hgoal_stuck_count[e] > 10
                if need_update or hgoal_stuck_too_much:
                    if hgoal_stuck_too_much:
                        print("hgoal stuck too much, replan hgoal")
                        self._mark_hgoal_unreachable(e)

                    ig_vis_list[e] = self._plan_hgoal(e)
                else:
                    # if no need to update high goal,
                    # we just need to transform the high goal to account for the robot's movement
                    self._update_hgoal_map(e) 
                    
                lc_input_list[e], lc_output_list[e] = self._call_low_level_planner(e)

                replan_hgoal = lc_output_list[e]['replan_hgoal']

                replan_num = 0 
                replan_num_max = 10
                while replan_hgoal and replan_num < replan_num_max:
                    # replan the high goal
                    self._mark_hgoal_unreachable(e)
                    ig_vis_list[e] = self._plan_hgoal(e)
                    lc_input_list[e], lc_output_list[e] = self._call_low_level_planner(e)
                    replan_hgoal = lc_output_list[e]['replan_hgoal']
                    replan_num += 1

                action = lc_output_list[e]['action']
                # transit to state CHECKING if we are close to the high goal
                if lc_output_list[e]['reach_hgoal']:
                    self._state[e] = STATES.CHECKING

                if replan_num >= replan_num_max:
                    print("replan too many times, terminate episode")
                    self._terminate_list[e] = True
                    action = DiscreteNavigationAction.STOP
                    
                # # TEST PN ANGENT
                # agent_pose = torch.tensor(np.concatenate([obs.gps,obs.compass]))
                # goal= torch.tensor([2.,0.])
                # action = self.pn_agent.plan(obs.depth,goal,agent_pose)
                # print(action)

            elif self._state[e] == STATES.CHECKING:
                # this will transit to state SEARCH if we finish looking around
                action = self._look_around(e)
                
            elif self._state[e] == STATES.GOING_TO_GOAL:
                self._update_hgoal_map(e)
                lc_input_list[e], lc_output_list[e] = self._call_low_level_planner(e)
                action = lc_output_list[e]['action']

                # # view point for end rec
                # if action == DiscreteNavigationAction.STOP and self._end_rec_view_point[e] is not None:
                #     delta_angle = self._end_rec_view_point[e][2] - obs.compass
                #     action = ContinuousNavigationAction(np.array([0,0,delta_angle.item()]))

                # # construct the end rec point cloud if it's present
                # # We gradually decrease the distance to the rec goal, so that we can determine the best view point
                # if self._to_end_rec[e] and \
                #     lc_output_list[e]['action'] == DiscreteNavigationAction.STOP:
                #     rec_score = (obs.semantic == obs.task_observations["end_recep_goal"]).sum() / obs.semantic.size
                #     # if we are still far from the end goal
                #     if self.planner.goal_tolerance > 4:
                #         # if we observe the end rec goal, we add points
                #         if rec_score > self.end_rec_mask_percent_threshold:
                #             self.place_policy.add_rec_points(obs)
                        
                #         # continue moving towards the end rec goal by decreasing the goal tolerance
                #         self.planner.goal_tolerance -= 4
                #         # take some random action and move closer the the end rec goal
                #         random_action = [DiscreteNavigationAction.TURN_RIGHT, DiscreteNavigationAction.TURN_LEFT]
                #         action = random_action[np.random.randint(2)]
                #         print(f"Continue to move to end rec with goal threshold {self.planner.goal_tolerance}")
                #     # if we are close to the end goal
                #     else:
                #         agent_pose = np.concatenate([obs.gps,obs.compass])
                #         placement = self.place_policy.get_receptacle_placement_point(agent_pose)
                #         if placement is not None:
                #             print("end rec goal reached with score. Stop")
                #             action = DiscreteNavigationAction.STOP
                #         else:
                #             print("end rec goal reached with score. No feasible placement point. Go to other end rec")
                #             #  no feasible placement point, go to other end rec
                #             self._hgoal_stuck_count[e] = 9 # trigger a replan

                # if cannot go to the current goal because: 
                # 1) low level planner cannot find a path, 2) stuck too much
                if self._hgoal_stuck_count[e] % 10 == 9 or lc_output_list[e]['replan_hgoal']:
                    # for end rec, we can go to the next instance
                    if self._to_end_rec[e]:
                        print(f'end rec stuck too much, change to next end rec')
                        goal_map = self._reset_end_rec_goal(e) 
                        if goal_map is None:
                            print(f'no more end rec in the map, continue to explore')
                            self._state[e] = STATES.SEARCHING

                    else:
                        # for goal obj, we relax the threshold and continue to move towards the goal
                        # since there's only one goal obj
                        # self._terminate_list[e] = True
                        can_relex = self.planner.relax_goal_tolerance(self._hgoal_stuck_count[e])
                        if not can_relex:
                            self._terminate_list[e] = True

                    if action == DiscreteNavigationAction.STOP:
                        random_action = [DiscreteNavigationAction.TURN_RIGHT, DiscreteNavigationAction.TURN_LEFT]
                        action = random_action[np.random.randint(2)]
                
                # if cannot getout even after relaxing the goal tolerance, or change different end rec
                # this typically indicate the agent is trapped
                # we terminate the episode
                if self._hgoal_stuck_count[e] > 40:
                    self._terminate_list[e] = True
                    action = DiscreteNavigationAction.STOP
                    print("hgoal stuck too much, terminate episode")


            else:
                raise ValueError("Invalid state")
                
        # if low level controller want to terminate
        if lc_output_list[0].get('end_episode',False):
            action = DiscreteNavigationAction.STOP
            self._terminate_list[0] = True

        self.timesteps = [self.timesteps[e] + 1 for e in range(self.num_environments)]
        
        info = {}
        if self.visualize:
            vis_inputs = [
                {
                    "timestep": self.timesteps[e],
                    "checking_area": None,
                    "exp_coverage": self.semantic_map.get_exp_coverage_area(e),
                    "close_coverage": self.semantic_map.get_close_coverage_area(e),
                    "entropy": self.semantic_map.get_probability_map_entropy(e),
                    "explored_map": self.semantic_map.get_explored_map(e),
                    "semantic_map": self.semantic_map.get_semantic_map(e),
                    "been_close_map": self.semantic_map.get_been_close_map(e),
                    "ig_vis": ig_vis_list[e],
                    "obstacle_map": self.semantic_map.get_obstacle_map(e),
                }
                for e in range(self.num_environments)
            ]

            vis_inputs[0]["semantic_frame"] = obs.task_observations["semantic_frame"]
            # vis_inputs[0]["closest_goal_map"] = closest_goal_map
            vis_inputs[0]["third_person_image"] = obs.third_person_image
            vis_inputs[0]["short_term_goal"] = lc_output_list[0].get('short_term_goal',None)
            vis_inputs[0]["dilated_obstacle_map"] = lc_output_list[0].get('dilated_obstacle_map',None)
            vis_inputs[0]["probabilistic_map"] = self.semantic_map.get_probability_map(0)
            vis_inputs[0]["goal_name"]: obs.task_observations["goal_name"]
            vis_inputs[0]["semantic_map_config"] = self.config.AGENT.SEMANTIC_MAP

            if lc_input_list[0] == {}:
                lc_input_list[0] = self._get_planner_inputs(0)
            info = {**lc_input_list[0], **vis_inputs[0]}
        
        self._last_action[0] = action
            
        info['early_termination'] = self._terminate_list[0]

        if self.timesteps[0] % 100 == 99:
            # decay the util_lambda
            self._util_lambda *= 0.95
        
        
        return action, info


    def reset_vectorized(self,episodes=None):
        """Initialize agent state."""
        self.timesteps = [0] * self.num_environments
        self.timesteps_before_goal_update = [0] * self.num_environments
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.semantic_map.init_map_and_pose()
        self.planner.reset()

        self._state = [STATES.CHECKING for _ in range(self.num_environments)]
        self._ur_goal_dist = np.zeros(self.num_environments)
        self._force_goal_update_once = np.full(self.num_environments, False)
        self._global_hgoal_pose = torch.zeros((self.num_environments,self.num_high_goals, 2), device=self.device)
        # self._ur_local_goal_coords = torch.zeros((self.num_environments,self.num_high_goals, 2), 
        #                                          dtype=torch.long,
        #                                          device=self.device)
        self._look_around_steps = np.zeros(self.num_environments)
        self._num_explored_grids = np.zeros(self.num_environments)
        self._num_promising_grids = np.zeros(self.num_environments)

        self._pre_num_obstacles = np.zeros(self.num_environments,dtype=int) # to track if map has changed
        self._map_changed = np.zeros(self.num_environments,dtype=bool) # to track if map has changed
        self._found_goal = np.zeros(self.num_environments,dtype=bool) # to track if map has changed
        self._hgoal_stuck_count = np.zeros(self.num_environments,dtype=int) # num of times we have been stuck for the same high goal
        self._last_action = [None for _ in range(self.num_environments)]
        self._last_pose = [None for _ in range(self.num_environments)]
        self._util_lambda = self.init_util_lambda # we can decay this over time
        self._max_obj_detection_scores = np.zeros(self.num_environments) # to track the highest object detection score
        self._to_end_rec = np.zeros(self.num_environments,dtype=bool) # to track if searching for the end receptacle
        self._end_rec_ins_idx = np.zeros(self.num_environments,dtype=int) # to track the instance index of the current goal end receptacle
        self._terminate_list = [False for _ in range(self.num_environments)]
        self._end_rec_view_point = [None for _ in range(self.num_environments)]
        
    def reset_vectorized_for_env(self, e: int, episode=None):
        """Initialize agent state for a specific environment."""
        self.timesteps[e] = 0
        self.timesteps_before_goal_update[e] = 0
        self.last_poses[e] = np.zeros(3)
        self.semantic_map.init_map_and_pose_for_env(e)
        self.planner.reset()
        self._state[e] = STATES.CHECKING
        self._ur_goal_dist[e] = 0
        self._force_goal_update_once[e] = False
        self._global_hgoal_pose[e] = torch.zeros((self.num_high_goals, 2), device=self.device)
        # self._ur_local_goal_coords[e] = torch.zeros((self.num_high_goals, 2), device=self.device).long()
        self._look_around_steps[e] = 0
        self._num_explored_grids[e] = 0
        self._num_promising_grids[e] = 0

        self._pre_num_obstacles[e] = 0
        self._map_changed[e] = False
        self._found_goal[e] = False
        self._hgoal_stuck_count[e] = 0

        self._last_action[e] = None
        self._last_pose[e] = None
        self._util_lambda = self.init_util_lambda # we can decay this over time
        self._max_obj_detection_scores[e] = 0
        self._to_end_rec[e] = False
        self._end_rec_ins_idx[e] = 0
        self._terminate_list[e] = False
        self._end_rec_view_point[e] = None
    # ---------------------------------------------------------------------
    # Inference methods to interact with the robot or a single un-vectorized
    # simulation environment
    # ---------------------------------------------------------------------

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()
        self.planner.reset()

    def get_nav_to_recep(self):
        return None

  
    #####################################################################
    # Helper methods for UR exploration
    #####################################################################
    def _select_goal_igp(self,e:int, ig_map):
        """
        Calculate the (local) utility map
        args: 
            ig_map: global info gain map 
        """
        dist_map = self._get_dist_map(e)
        lmb = self.semantic_map.lmb[e]
        local_ig_map = ig_map[lmb[0]:lmb[1],lmb[2]:lmb[3]]
        
        # we want to prioritize the grids that are close to the current location
        # and have high info gain
        # U = I*e^(-lambda * d)
        # smaller lambda means we want to prioritize the grids that are close to the current location
        if self.utility_exp:
            utility_map = local_ig_map * torch.exp(-self._util_lambda * dist_map)
        else:
            dist_map[dist_map==0] = 10
            utility_map = local_ig_map / dist_map ** 0.7 * 10
            
        # select goal with max utility
        if not self.random_ur_goal:
            local_goal_idx = torch.argmax(utility_map)

        # select goal randomly from topk utility
        else:
            u, idx = torch.topk(utility_map.view(-1), 100)
            local_goal_idx = idx[torch.randint(0,20,(1,))]
        
        
        x = local_goal_idx // utility_map.shape[1]
        y = local_goal_idx % utility_map.shape[1]
        local_goal = torch.tensor([[x,y]]).to(self.device)
        
        return utility_map,local_goal,dist_map[x,y].item()
    
    def _get_dist_map(self,e:int) -> torch.tensor:
        """
        Calculate the (local) distance field (in cells) from the current location
        args:
            e: environment index
        returns:
            dist_map: distance field in cells

        """
        # NOTE: if we dialate too much, the obstacle may swallow the agent, and the dist map will be wrong
        dilated_obstacles = self.semantic_map.get_dialated_obstacle_map_local(e,self.ur_dist_obstacle_dialate_radius)
        agent_rad = self.agent_cell_radius

        # we need to make sure the agent is not inside the obstacle
        while agent_rad < 25:
            traversible = 1 - dilated_obstacles
            start = self.semantic_map.get_local_coords(e)

            agent_rad += 1
            traversible[
                start[0] - agent_rad : start[0] + agent_rad + 1,
                start[1] - agent_rad : start[1] + agent_rad + 1,
            ] = 1
            traversible = add_boundary(traversible)
            vis_planner = FMMPlanner(traversible)
            curr_loc_map = np.zeros_like(traversible)

            # Update our location for finding the closest goal
            start = self.semantic_map.get_local_coords(e)
            curr_loc_map[start[0], start[1]] = 1
            # curr_loc_map[short_term_goal[0], short_term_goal]1]] = 1
            vis_planner.set_multi_goal(curr_loc_map)
            fmm_dist = vis_planner.fmm_dist
            fmm_dist = remove_boundary(fmm_dist)
            fmm_dist[dilated_obstacles==1] = 10000

            # check if the agent is inside the obstacle
            if np.unique(fmm_dist).shape[0] > 10:
                return torch.from_numpy(fmm_dist).float().to(self.device)
    
        # if we cannot find a valid dist map, we just return a map with all 1s
        print("cannot find a valid dist map, return a uniform map")
        fmm_dist = np.ones_like(dilated_obstacles)
        return torch.from_numpy(fmm_dist).float().to(self.device)
    
    def _get_goal_map(self,locs:torch.tensor) -> np.ndarray:
        """
        Convert the goal locations to a goal map that can be used by the planner
        args:
            locs: locations of the goals, tensor of shape [N,2] (in local map coords)
        returns:
            goal_map: goal map, numpy array of shape [H,W]
        """

        goal_map = torch.zeros([self.semantic_map.local_map_size,self.semantic_map.local_map_size])
        goal_map[locs[:,0],locs[:,1]] = 1        

        return goal_map.cpu().numpy()

    

    def _check_n_update_need_replan_ur_goal(self, e:int) -> bool:
        """
        decide whether to replan. We also update ur_goal_dist here.
        Replan if:
            1. if the selected goal is occupaid by obstacles
            2. if the map changes too much (so there could be more interesting locs to explore?)
            3. if the distance to selected goal changed too much, because of newly 
            observed obstacles. (so better to explore nearby first?)
            4. if force_update or timesteps_before_goal_update is reached, or we have traveled too far
            5. if the agent is stuck (low level controller check this)
        
        """
        
        replan = False

        # case 4
        if self.timesteps_before_goal_update[e] == 0 or self._force_goal_update_once[e]:
            replan = True
        
        # case 2
        num_exp_grid = self.semantic_map.get_num_explored_cells(e)
        num_changed_cells = num_exp_grid - self._num_explored_grids[e]
        num_promising_cells = self.semantic_map.get_num_promising_cells(e)
        num_changed_promising_cells = abs(num_promising_cells - self._num_promising_grids[e])
        if num_changed_cells > self.max_num_changed_cells:
            replan = True
        if num_changed_promising_cells > self.max_num_changed_promising_cells:
            replan = True
        
      

        # update internal states
        if replan:
            self.timesteps_before_goal_update[e] = self.goal_update_steps
            self._force_goal_update_once[e] = False
            self._num_explored_grids[e] = num_exp_grid
            self._num_promising_grids[e] = num_promising_cells
            
        else:
            # timestep to replan for the ur goal
            self.timesteps_before_goal_update = [
                self.timesteps_before_goal_update[e] - 1
                for e in range(self.num_environments)
            ]

        return replan

    def _compute_reachable_area_to_end_rec(self,e):
        """
        Use polo network to approximately compute the reachable area to the end rec goal
        """

        voxel = self.semantic_map.get_global_end_rec_voxel(e)
        voxel[voxel==0] = -torch.inf
        voxel[voxel==1] = -13
        voxel[voxel==2] = 13
        pred = self.ig_predictor.predict(voxel)

        return pred[1]

        
    def _compute_info_gains_igp(self,e):

        # points, feats = self.semantic_map.get_global_pointcloud(e) # [P, 3], [P, 1]
        # feats = feats.unsqueeze(0) # [1, P, 1]
        # point_idx = self.semantic_map.get_global_pointcloud_flat_idx(e).squeeze(-1).cpu().to(torch.int32)
        # p = feats[0].squeeze(-1).cpu().to(torch.float16) # save space
        # pred_pc = self.ig_predictor.predict_pc(point_idx, p) # [2, P]

        # save for debug
        # torch.save([point_idx, p],f'data/info_gain/test/{self.timesteps[0]}.pt')

        voxel = self.semantic_map.get_global_voxel(e)
        pred = self.ig_predictor.predict(voxel)
        # obstacle = self.semantic_map.global_map[e,0] > 0 # [M x M]
        obstacle = self.semantic_map.get_dialated_obstacle_map_global(e,self.ur_obstacle_dialate_radius) # [M x M]
        unreachable = self.semantic_map.hgoal_unreachable[e] # [M x M]
        
        # we add obstacles from collision map
        collision_map = torch.from_numpy(self.planner.collision_map).to(self.device)
        exclude = torch.logical_or(obstacle, torch.logical_or(collision_map,unreachable))
        pred[exclude.repeat(2,1,1)] = 0
        
        pred_ig = pred[0] + self.info_gain_alpha * pred[1] / self.ig_predictor.i_s_weight

        if pred_ig.max() < 0.1:
            print("WARNING: all info gains are too small, terminate episode")
            self._terminate_list[e] = True

        vis = {}
        if self.visualize and self.render_igp:
            vis['cs'] = render_plt_image(pred[0])
            vis['is'] = render_plt_image(pred[1]/self.ig_predictor.i_s_weight) 
            vis['ig'] = render_plt_image(pred_ig)

                
        return pred_ig, vis
    
    def _preprocess_obs(self, obs: Observations):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map.
        Note: obs is a single observation, not a batch of observations.
        """
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm
            
        semantic = np.full_like(obs.semantic, 4)
        obj_goal_idx, start_recep_idx, end_recep_idx = 1, 2, 3
        semantic[obs.semantic == obs.task_observations["object_goal"]] = obj_goal_idx
        if "start_recep_goal" in obs.task_observations:
            semantic[
                obs.semantic == obs.task_observations["start_recep_goal"]
            ] = start_recep_idx
        if "end_recep_goal" in obs.task_observations:
            semantic[
                obs.semantic == obs.task_observations["end_recep_goal"]
            ] = end_recep_idx
        semantic = self.one_hot_encoding[torch.from_numpy(semantic).to(self.device)]
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1).unsqueeze(0)
        obs_preprocessed = obs_preprocessed.permute(0, 3, 1, 2)

        detection_results = None
        # if self.use_probability_map:
        # we always update the prob map for evaluation
        relevance = torch.zeros(obs.task_observations['semantic_max_val']+1).to(self.device)
        if not self._to_end_rec[0]:
            relevance[obs.task_observations['object_goal']] = 1
            relevance[obs.task_observations['start_recep_goal']] = 0.7
        else:
            relevance[obs.task_observations['end_recep_goal']] = 1
        detection_results = {'scores':torch.tensor(obs.task_observations['instance_scores']).unsqueeze(0),
                                'classes':torch.tensor(obs.task_observations['instance_classes']).unsqueeze(0),
                                'masks':torch.tensor(obs.task_observations['masks']).unsqueeze(0),
                                'relevance':relevance}
        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses[0])
        ).unsqueeze(0)
        self.last_poses[0] = curr_pose
        object_goal_category = None
        end_recep_goal_category = None
        if (
            "object_goal" in obs.task_observations
            and obs.task_observations["object_goal"] is not None
        ):
            if self.verbose:
                print("object goal =", obs.task_observations["object_goal"])
            object_goal_category = torch.tensor(obj_goal_idx).unsqueeze(0)
        start_recep_goal_category = None
        if (
            "start_recep_goal" in obs.task_observations
            and obs.task_observations["start_recep_goal"] is not None
        ):
            if self.verbose:
                print("start_recep goal =", obs.task_observations["start_recep_goal"])
            start_recep_goal_category = torch.tensor(start_recep_idx).unsqueeze(0)
        if (
            "end_recep_goal" in obs.task_observations
            and obs.task_observations["end_recep_goal"] is not None
        ):
            if self.verbose:
                print("end_recep goal =", obs.task_observations["end_recep_goal"])
            end_recep_goal_category = torch.tensor(end_recep_idx).unsqueeze(0)
        goal_name = [obs.task_observations["goal_name"]]

        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)
        return (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            start_recep_goal_category,
            end_recep_goal_category,
            goal_name,
            camera_pose,
            detection_results,
        )
