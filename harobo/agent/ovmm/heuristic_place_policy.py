# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import trimesh.transformations as tra
from home_robot.utils import rotation as ru
import home_robot.utils.depth as du
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    DiscreteNavigationAction,
    ContinuousNavigationAction,
    Observations,
)
from pytorch3d.ops import sample_farthest_points

from home_robot.motion.stretch import STRETCH_STANDOFF_DISTANCE, STRETCH_ARM_EXTENSION
from home_robot.utils.image import smooth_mask
from home_robot.utils.rotation import get_angle_to_pos
from utils.visualization import (
    display_grayscale,
    display_rgb,
    plot_image,
    save_image, 
    draw_top_down_map, 
    Recording, 
    visualize_gt,
    show_points_with_prob,
    render_plt_image,
    visualize_pred,
    show_points, 
    show_voxel_with_prob, 
    show_voxel_with_logit,
    save_img_tensor)    
RETRACTED_ARM_APPROX_LENGTH = 0.15
HARDCODED_ARM_EXTENSION_OFFSET = 0.15 + 0.1
HARDCODED_YAW_OFFSET = 0.25
GRIPER_HEIGHT = 0.28
STAND_OFF_DISTANCE = 0.5
def transform_to_global(XYZ,current_pose,device):
    
    R = ru.get_r_matrix([0.0, 0.0, 1.0], angle=current_pose[2]-np.pi/2)
    XYZ = torch.matmul(
        XYZ.reshape(-1, 3), torch.from_numpy(R).float().transpose(1, 0).to(device)
    ).reshape(XYZ.shape)
    XYZ[..., 0] += current_pose[0]
    XYZ[..., 1] += current_pose[1]
    return XYZ
    
def get_heading_delta(target,obs, degrees=False):
    
    # find the angle between the center of the voxel and the agent
    phi = np.arctan2(target[1] - obs.gps[1],
                target[0] - obs.gps[0])
    # transform to agent's frame
    delta_heading = phi - obs.compass
    if delta_heading > np.pi:
        delta_heading -= 2*np.pi
        
    if degrees:
        delta_heading = np.rad2deg(delta_heading)
    return delta_heading
class HeuristicPlacePolicy(nn.Module):
    """
    Policy to place object on end receptacle using depth and point-cloud-based heuristics. Objects will be placed nearby, on top of the surface, based on point cloud data. Requires segmentation to work properly.
    """

    # TODO: read these values from the robot kinematic model
    look_at_ee = np.array([-np.pi / 2, -np.pi / 4])
    max_arm_height = 1.2

    def __init__(
        self,
        config,
        device,
        placement_drop_distance: float = 0.4,
        debug_visualize_xyz: bool = False,
        verbose: bool = False,
    ):
        """
        Parameters:
            config
            device
            placement_drop_distance: distance from placement point that we add as a margin
            debug_visualize_xyz: whether to display point clouds for debugging
            verbose: whether to print debug statements
        """
        super().__init__()
        self.timestep = 0
        self.config = config
        self.device = device
        self.debug_visualize_xyz = debug_visualize_xyz
        self.erosion_kernel = np.ones((5, 5), np.uint8)
        self.placement_drop_distance = placement_drop_distance
        self.verbose = verbose
        self.rec_points = []
        self.du_scale = 1  # TODO: working with full resolution for now
        self.selected_voxel = None
        self.replanned = False

    def reset(self):
        self.timestep = 0
        self.rec_points = []
        self.selected_voxel = None
        self.replanned = False
        self.placement_drop_distance = 0.4

    def set_rec_points(self,rec_points):
        self.rec_points = rec_points
        
    def add_rec_points(self,obs,trans_global=True,filter_mask=True):
        """
        Add points from the end receptacle to the point cloud.
        
        """
        # first get points in base coordinates
        goal_rec_depth = torch.tensor(
            obs.depth, device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        camera_matrix = du.get_camera_matrix(
            self.config.ENVIRONMENT.frame_width,
            self.config.ENVIRONMENT.frame_height,
            self.config.ENVIRONMENT.hfov,
        )
        # Get object point cloud in camera coordinates
        pcd_camera_coords = du.get_point_cloud_from_z_t(
            goal_rec_depth, camera_matrix, self.device, scale=self.du_scale
        )

        # get point cloud in base coordinates
        camera_pose = np.expand_dims(obs.camera_pose, 0)
        angles = [tra.euler_from_matrix(p[:3, :3], "rzyx") for p in camera_pose]
        tilt = angles[0][1]  # [0][1]

        # Agent height comes from the environment config
        agent_height = torch.tensor(camera_pose[0, 2, 3], device=self.device)

        # Object point cloud in base coordinates
        pcd_base_coords = du.transform_camera_view_t(
            pcd_camera_coords, agent_height, np.rad2deg(tilt), self.device
        ) # N x H x W x 3
        
        pcd_base_coords = pcd_base_coords[0].view(-1,3) # N*H*W x 3

        # apply mask to remove points outside of the end receptacle
        goal_rec_mask = (
            obs.semantic
            == obs.task_observations["end_recep_goal"] * du.valid_depth_mask(obs.depth)
        ).astype(np.uint8)
        # Get dilated, then eroded mask (for cleanliness)
        goal_rec_mask = smooth_mask(
            goal_rec_mask, self.erosion_kernel, num_iterations=5
        )[1]
        # Convert to booleans
        goal_rec_mask = torch.from_numpy(goal_rec_mask.astype(bool)).to(self.device)
        if filter_mask:
            pcd_base_coords = pcd_base_coords[goal_rec_mask.view(-1),:] # P x 3

        
        if trans_global:
            # transform to global frame
            agent_pose = np.concatenate([obs.gps,obs.compass])
            xyz_global = du.transform_pose_t(
                pcd_base_coords, agent_pose , self.device
            )
        else:
            xyz_global = pcd_base_coords

        # add to list of points
        self.rec_points.append(xyz_global)

        return xyz_global

    def filter_arm_reachable_points(self, pcd_base_coords, agent_height):
        # filtering out unreachable points based on Y and Z coordinates of voxels (Z is up)
        height_reachable_mask = (pcd_base_coords[..., 2] < agent_height).to(int)
        length_reachable_mask = (pcd_base_coords[..., 1] < agent_height).to(int)
        reachable_mask = torch.logical_and(height_reachable_mask, length_reachable_mask)

        return pcd_base_coords[reachable_mask]
        
    def get_agent_pose_matrix(self,agent_pose):
        """
        get 4x4 transformation matrix describing agent pose. using habitat world frame (y-right, z-up, x-forward)

        Args:
            agent_pose: tuple (x,y,theta), in agent base frame, unit is meter and radian, theta is counter-clockwise
        """
        x, y, theta = agent_pose
        R = ru.get_r_matrix([0.0, 0.0, 1.0], angle=theta)
        T = torch.eye(4, 4)
        T[:3, 3] = torch.tensor([x, y, 0.0])
        T[:3, :3] = torch.tensor(R)
        return T

    def get_agent_base_frame_pc(self,current_pose):
        # def get_camera_frame(self, agent_pose):
        """
        get the camera frame expressed in the world frame
        TODO: support batch
        Args:
            agent_pose: tuple (x,y,theta), in agent base frame, unit is meter and radian
        """
        # T = self.get_agent_pose_matrix(agent_pose)

        XYZ = torch.cat(self.rec_points, dim=0) # N x 3
        XYZ[..., 0] -= current_pose[0]
        XYZ[..., 1] -= current_pose[1]
        R = ru.get_r_matrix([0.0, 0.0, 1.0], angle=-(current_pose[2] - np.pi / 2.0))
        XYZ = torch.matmul(
            XYZ, torch.from_numpy(R).float().transpose(1, 0).to(self.device)
        ).reshape(XYZ.shape)

        return XYZ
    
    def get_global_frame_pc(self):
        """
        get the point cloud expressed in the world frame
        """
        XYZ = torch.cat(self.rec_points, dim=0) # N x 3
        return XYZ
    
    def get_receptacle_placement_point(
        self,
        agent_pose: np.array,
        slab_min_point=300,
        max_reach=1.5,
    ):
        """
        Compute placement point in 3d space.

        Parameters:
            obs: Observation object; describes what we've seen.
            vis_inputs: optional dict; data used for visualizing outputs
        """
        NUM_POINTS_TO_SAMPLE = 50  # number of points to sample from receptacle point cloud to find best placement point
        SLAB_PADDING = 0.2  # x/y padding around randomly selected points
        SLAB_HEIGHT_THRESHOLD = 0.015  # 1cm above and below, i.e. 2cm overall
        if self.rec_points == []:
            return None
        else:
            ## randomly sampling NUM_POINTS_TO_SAMPLE of receptacle point cloud – to choose for placement

            # NOTE: we are using the pc in global frame
            # pcd_base_coords = self.get_agent_base_frame_pc(agent_pose)
            pc_global = self.get_global_frame_pc()

            # NOTE: we don't sample since the point cloud is already sparse
            # pc_global, _ = sample_farthest_points(pc_global.unsqueeze(0), K=3000)
            # pc_global = pc_global.squeeze(0)

            reachable = torch.sqrt((pc_global[:,0] - agent_pose[0])**2 + \
                        (pc_global[:,1] - agent_pose[1])**2)< max_reach
            reachable_point_cloud = pc_global[reachable]
            
            if reachable_point_cloud.shape[0] < slab_min_point:
                print("No good placement point found")
                return None
            # reachable_point_cloud = self.filter_arm_reachable_points(pc_global, agent_height)

            # find the indices of the non-zero elements in the first two dimensions of the matrix

            # select a random subset of the non-zero indices
            num_point = reachable_point_cloud.shape[0]
            random_indices = random.sample(list(range(num_point)), min(NUM_POINTS_TO_SAMPLE, num_point))

            x_values = reachable_point_cloud[..., 0]
            y_values = reachable_point_cloud[..., 1]
            z_values = reachable_point_cloud[..., 2]

            max_surface_points = 0
            # max_height = 0

            max_surface_mask, best_voxel_ind, best_voxel = None, None, None

            # iterating through all randomly selected voxels and choosing one with most XY neighboring surface area within some height threshold
            for ind in random_indices:
                sampled_voxel = reachable_point_cloud[ind]

                slab_points_mask_z = torch.abs(z_values - sampled_voxel[2]) < SLAB_HEIGHT_THRESHOLD
                slab_points_mask_x = torch.abs(x_values - sampled_voxel[0]) < SLAB_PADDING
                slab_points_mask_y = torch.abs(y_values - sampled_voxel[1]) < SLAB_PADDING
                slab_points_mask = torch.logical_and(slab_points_mask_x, slab_points_mask_y)
                slab_points_mask = torch.logical_and(slab_points_mask, slab_points_mask_z)
                
              
                if slab_points_mask.sum() >= max_surface_points:
                    max_surface_points = slab_points_mask.sum()
                    max_surface_mask = slab_points_mask
                    best_voxel_ind = ind
                    best_voxel = sampled_voxel

            
            if max_surface_points < slab_min_point:
                print(f"No good placement point found, number of points {max_surface_points}")
                return None

            # find the pixel at the center of the slab
            best_voxel[0] = reachable_point_cloud[max_surface_mask][:,0].mean()
            best_voxel[1] = reachable_point_cloud[max_surface_mask][:,1].mean()

            # visualize
            visualize=False
            if visualize:
                p = torch.zeros_like(pc_global)[:,:1]
                p[reachable] = 0.3
                best_surrounding = (torch.abs(pc_global[:,0]-best_voxel[0]) < 0.03 ) \
                    & (torch.abs(pc_global[:,1]-best_voxel[1]) < 0.03 ) \
                    & (torch.abs(pc_global[:,2]-best_voxel[2]) < 0.03 )
                p[best_surrounding] = 0.9
                show_points_with_prob(pc_global, p)
            
         
            # Add placement margin to the best voxel that we chose
            # best_voxel[2] += self.placement_drop_distance


            self.selected_voxel = best_voxel.cpu().numpy()
            return self.selected_voxel

    def forward(self, obs: Observations, vis_inputs: Optional[Dict] = None):
        """
        This should only be called after the get_placement_point find a placement point.
        1. Get estimate of point on receptacle to place object on.
        2. Orient towards it.
        3. Move forward to get close to it.
        4. Rotate 90º to have arm face the object. Then rotate camera to face arm.
        5. (again) Get estimate of point on receptacle to place object on.
        6. With camera, arm, and object (hopefully) aligned, set arm lift and
        extension based on point estimate from 4.

        Returns:
            action: what the robot will do - a hybrid action, discrete or continuous
            vis_inputs: dictionary containing extra info for visualizations
        """

        if not self.replanned and self.timestep > 0 and self.timestep < 3:
            # try to identify the end receptacle
            self.rec_points = []
            self.add_rec_points(obs)
            agent_pose = np.concatenate([obs.gps,obs.compass])
            placement = self.get_receptacle_placement_point(agent_pose,5000)
            if placement is not None:
                # use new placement point
                # self.selected_voxel = placement
                self.replanned = True
                self.timestep = 0

        if self.timestep == 0:

            self.forward_steps = 1 
            self.initial_orient_num_turns = 1 # use contiuous action to orient, use 2 steps to make sure it's aligned
            self.total_turn_and_forward_steps = (
                self.forward_steps + self.initial_orient_num_turns
            )
            self.fall_wait_steps = 0
            self.t_go_to_top = self.total_turn_and_forward_steps + 1
            self.t_go_to_place = self.total_turn_and_forward_steps + 2
            self.t_release_object = self.total_turn_and_forward_steps + 3
            self.t_lift_arm = self.total_turn_and_forward_steps + 4
            self.t_retract_arm = self.total_turn_and_forward_steps + 5
            self.t_extend_arm = -1
            self.t_done_waiting = (
                self.total_turn_and_forward_steps + 5 + self.fall_wait_steps
            )
            if self.verbose:
                print("-" * 20)
                print(f"Turn to orient for {self.initial_orient_num_turns} steps.")
                print(f"Move forward for {self.forward_steps} steps.")

        if self.verbose:
            print("-" * 20)
            print("Timestep", self.timestep)
        if self.timestep < self.initial_orient_num_turns:
            
            delta_heading = get_heading_delta(self.selected_voxel,obs)
            action = ContinuousNavigationAction(np.array([0,0,delta_heading.item()]))
            print("[Placement] Turning to orient towards object")
        elif self.timestep < self.total_turn_and_forward_steps:
            print("[Placement] Moving forward")
            dist_to_placement = np.linalg.norm(obs.gps - self.selected_voxel[:2]).item()
            fwd_dist = max( dist_to_placement - STAND_OFF_DISTANCE, 0.01)
            action = ContinuousNavigationAction(np.array([fwd_dist,0,0]))
        elif self.timestep == self.total_turn_and_forward_steps:
            ARM_WIDTH = 0.2
            ARM_HEIGHT = 0.2
            dist = (self.selected_voxel[0] - obs.gps[0])**2 + \
                (self.selected_voxel[1] - obs.gps[1])**2
            dist = np.sqrt(dist)
            # calculate lift before switching to manipulation mode 
            local_pc = self.get_agent_base_frame_pc(np.concatenate([obs.gps,obs.compass]))
            for delta_height in range(1, 4):
                placement_height = self.selected_voxel[2] + delta_height*0.1 + GRIPER_HEIGHT
                # check if there are points in the arm extension volume
                x_in = torch.logical_and(local_pc[:,0] < ARM_WIDTH/2, local_pc[:,0] > -ARM_WIDTH/2)
                y_in = torch.logical_and(local_pc[:,1] < dist, local_pc[:,1] > 0)
                z_in = torch.logical_and(local_pc[:,2] < placement_height + ARM_HEIGHT/2, \
                                        local_pc[:,2] > placement_height - GRIPER_HEIGHT - 0.05)
                any_in = torch.logical_and(x_in, y_in)
                any_in = torch.logical_and(any_in, z_in)
                if not any_in.sum() > 0:
                    self.placement_drop_distance = delta_height*0.1
                    break
                
                
            action = DiscreteNavigationAction.MANIPULATION_MODE
        elif self.timestep == self.t_go_to_top:
            # We should move the arm back and retract it to make sure it does not hit anything as it moves towards the target position
            action = self._retract(obs)
            # action = DiscreteNavigationAction.EMPTY_ACTION
        elif self.timestep == self.t_go_to_place:
            if self.verbose:
                print("[Placement] Move arm into position")
                

            
            placement_height = self.selected_voxel[2] + self.placement_drop_distance + GRIPER_HEIGHT
            dist = (self.selected_voxel[0] - obs.gps[0])**2 + \
                (self.selected_voxel[1] - obs.gps[1])**2
            placement_extension = np.sqrt(dist)

            current_arm_lift = obs.joint[4]
            delta_arm_lift = placement_height - current_arm_lift

            current_arm_ext = obs.joint[:4].sum()
            
            delta_arm_ext = (
                placement_extension
                # + 0.1
                - RETRACTED_ARM_APPROX_LENGTH
                - current_arm_ext
                + HARDCODED_ARM_EXTENSION_OFFSET
            )

            delta_heading = get_heading_delta(self.selected_voxel,obs, degrees=True)

            delta_gripper_yaw = delta_heading / 90 - HARDCODED_YAW_OFFSET

            if self.verbose:
                print("[Placement] Delta arm extension:", delta_arm_ext)
                print("[Placement] Delta arm lift:", delta_arm_lift)
            joints = np.array(
                [delta_arm_ext]
                + [0] * 3
                + [delta_arm_lift]
                + [delta_gripper_yaw.item()]
                + [0] * 4
            )
            joints = self._look_at_ee(joints)
            action = ContinuousFullBodyAction(joints)
        elif self.timestep == self.t_release_object:
            # desnap to drop the object
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif self.timestep == self.t_lift_arm:
            action = self._lift(obs)
        elif self.timestep == self.t_retract_arm:
            action = self._retract(obs)
        elif self.timestep == self.t_extend_arm:
            action = DiscreteNavigationAction.EXTEND_ARM
        elif self.timestep <= self.t_done_waiting:
            if self.verbose:
                print("[Placement] Empty action")  # allow the object to come to rest
            action = DiscreteNavigationAction.EMPTY_ACTION
        else:
            if self.verbose:
                print("[Placement] Stopping")
            action = DiscreteNavigationAction.STOP

        debug_texts = {
            self.total_turn_and_forward_steps: "[Placement] Aligning camera to arm",
            self.t_go_to_top: "[Placement] Raising the arm before placement.",
            self.t_go_to_place: "[Placement] Move arm into position",
            self.t_release_object: "[Placement] Desnapping object",
            self.t_lift_arm: "[Placement] Lifting the arm after placement.",
            self.t_retract_arm: "[Placement] Retracting the arm after placement.",
            self.t_extend_arm: "[Placement] Extending the arm out for placing.",
            self.t_done_waiting: "[Placement] Empty action",
        }
        if self.verbose and self.timestep in debug_texts:
            print(debug_texts[self.timestep])

        self.timestep += 1
        return action, vis_inputs

    def _lift(self, obs: Observations) -> ContinuousFullBodyAction:
        """Compute a high-up lift position to avoid collisions when releasing"""
        # Hab sim dimensionality for arm == 10
        joints = np.zeros(10)
        # We take the lift position = 1
        current_arm_lift = obs.joint[4]
        # Target lift is 0.99
        lift_delta = self.max_arm_height - current_arm_lift
        joints[4] = lift_delta
        joints = self._look_at_ee(joints)
        action = ContinuousFullBodyAction(joints)
        return action

    def _look_at_ee(self, joints: np.ndarray) -> np.ndarray:
        """Make sure it's actually looking at the end effector."""
        joints[8] = self.look_at_ee[0]
        joints[9] = self.look_at_ee[1]
        return joints

    def _retract(self, obs: Observations) -> ContinuousFullBodyAction:
        """Compute a high-up retracted position to avoid collisions"""
        # Hab sim dimensionality for arm == 10
        joints = np.zeros(10)
        # We take the lift position = 1
        current_arm_lift = obs.joint[4]
        # Target lift is 0.99
        lift_delta = self.max_arm_height - current_arm_lift
        # Arm should be fully retracted
        arm_delta = -1 * np.sum(obs.joint[:4])
        joints[0] = arm_delta
        joints[4] = lift_delta
        joints = self._look_at_ee(joints)
        action = ContinuousFullBodyAction(joints)
        return action
