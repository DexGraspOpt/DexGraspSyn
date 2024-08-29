"""
Last modified date: 2023.02.19
Author: Ruicheng Wang
Description: Class IsaacValidator
"""

from isaacgym import gymapi
from isaacgym import gymutil
import math
from time import sleep

gym = gymapi.acquire_gym()


class IsaacValidator():

    def __init__(self,
                 mode='direct',
                 hand_friction=3.,
                 obj_friction=3.,
                 threshold_dis=0.1,
                 env_batch=1,
                 sim_step=100,
                 gpu=0,
                 debug_interval=0.05,
                 joint_names=None):

        self.hand_friction = hand_friction
        self.obj_friction = obj_friction
        self.debug_interval = debug_interval
        self.threshold_dis = threshold_dis
        self.env_batch = env_batch
        self.gpu = gpu
        self.sim_step = sim_step
        self.envs = []
        self.hand_handles = []
        self.obj_handles = []
        self.hand_rigid_body_sets = []
        self.obj_rigid_body_sets = []
        self.num_hand_dofs = 16
        if joint_names is None:
            self.joint_names = ['1','0','2','3','5','4','6','7','9','8','10','11','12','13','14','15']
        else:
            self.joint_names = joint_names

        self.hand_asset = None
        self.obj_asset = None

        self.sim_params = gymapi.SimParams()

        # set common parameters
        self.sim_params.dt = 1 / 60
        self.sim_params.substeps = 2
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0, -9.8)
        self.sim_params.num_client_threads = 8

        # set PhysX-specific parameters
        self.sim_params.physx.use_gpu = True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.contact_offset = 0.004  # 0.01 (smaller value is more strictly, and preserve less valid grasp)
        self.sim_params.physx.rest_offset = 0.0
        self.sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        self.sim_params.physx.num_threads = 6
        self.physics_engine = gymapi.SIM_PHYSX

        self.sim_params.use_gpu_pipeline = False
        self.sim = gym.create_sim(self.gpu, self.gpu, self.physics_engine, self.sim_params)
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 800
        self.camera_props.height = 600
        self.camera_props.use_collision_geometry = True

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 1.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        gym.add_ground(self.sim, plane_params)

        # set viewer
        self.viewer = None
        if mode == "gui":
            self.has_viewer = True
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
            gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(0, 1, 1),
                                      gymapi.Vec3(0, 0, 0))
        else:
            self.has_viewer = False

        self.hand_asset_options = gymapi.AssetOptions()
        self.hand_asset_options.disable_gravity = True
        self.hand_asset_options.fix_base_link = True
        self.hand_asset_options.collapse_fixed_joints = True
        self.hand_asset_options.vhacd_enabled = False
        # self.hand_asset_options.override_com = True
        # self.hand_asset_options.override_inertia = True
        # self.hand_asset_options.density = 1000
        if self.physics_engine == gymapi.SIM_PHYSX:
            self.hand_asset_options.use_physx_armature = True
        self.hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        self.obj_asset_options = gymapi.AssetOptions()
        self.obj_asset_options.fix_base_link = False
        self.obj_asset_options.override_com = True
        self.obj_asset_options.override_inertia = True
        self.obj_asset_options.density = 500

        self.test_rotations = [
            gymapi.Transform(gymapi.Vec3(0, 0, 0), gymapi.Quat(0, 0, 0, 1)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                            1 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                            0.5 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                            -0.5 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0),
                                            0.5 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0),
                                            -0.5 * math.pi)),
        ]

    def set_asset(self, hand_root, hand_file, obj_root, obj_file):
        self.hand_asset = gym.load_asset(self.sim, hand_root, hand_file,
                                         self.hand_asset_options)
        self.obj_asset = gym.load_asset(self.sim, obj_root, obj_file,
                                        self.obj_asset_options)

    # def _load_hand_asset(self, hand_asset_root, hand_asset_file):
    #     hand_asset_options = gymapi.AssetOptions()
    #     hand_asset_options.disable_gravity = True
    #     hand_asset_options.fix_base_link = True
    #     hand_asset_options.collapse_fixed_joints = True
    #     # # Convex decomposition
    #     hand_asset_options.vhacd_enabled = False
    #     # hand_asset_options.vhacd_params.resolution = 50000
    #     # hand_asset_options.vhacd_params.max_convex_hulls = 4
    #     # self.robot_asset_options.override_com = True
    #     # self.robot_asset_options.override_inertia = True
    #     # self.robot_asset_options.density = 1000
    #
    #     if self.physics_engine == gymapi.SIM_PHYSX:
    #         hand_asset_options.use_physx_armature = True
    #     hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    #
    #     self.hand_asset = gym.load_asset(self.sim, hand_asset_root, hand_asset_file, hand_asset_options)
    #
    # def _load_obj_asset(self, obj_asset_root, obj_asset_file):
    #     obj_asset_options = gymapi.AssetOptions()
    #     obj_asset_options.fix_base_link = False
    #     obj_asset_options.override_com = True
    #     obj_asset_options.override_inertia = True
    #     obj_asset_options.density = 500
    #
    #     self.obj_asset = gym.load_asset(self.sim, obj_asset_root, obj_asset_file, obj_asset_options)

    def create_envs(self, hand_rotation, hand_translation, hand_qpos, obj_scale, target_qpos=None):
        self.hand_actor = []
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.num_hand_bodies = gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_hand_shapes = gym.get_asset_rigid_shape_count(self.hand_asset)
        num_hand_dofs = gym.get_asset_dof_count(self.hand_asset)
        assert (
                self.num_hand_dofs == num_hand_dofs
        ), f"Number of DOFs in asset {self.hand_asset} is {num_hand_dofs}, but {self.num_hand_dofs} was expected"

        # hand_rigid_body_names = [
        #     gym.get_asset_rigid_body_name(self.hand_asset, i) for i in range(self.num_hand_bodies)
        # ]
        # print(f"Robot num rigid bodies: {self.num_hand_bodies}")
        # print(f"Robot rigid bodies: {hand_rigid_body_names}")

        self.num_obj_bodies = gym.get_asset_rigid_body_count(self.obj_asset)
        self.num_obj_shapes = gym.get_asset_rigid_shape_count(self.obj_asset)

        max_agg_bodies = self.num_hand_bodies + self.num_obj_bodies
        max_agg_shapes = self.num_hand_shapes + self.num_obj_shapes
        batch_size = hand_qpos.shape[0]
        for i in range(batch_size):
            for test_rot in self.test_rotations:
                env = gym.create_env(self.sim, env_lower, env_upper, 6)
                gym.begin_aggregate(env, max_agg_bodies, max_agg_shapes, True)
                self.envs.append(env)
                pose = gymapi.Transform()
                pose.r = gymapi.Quat(*hand_rotation[i, 1:], hand_rotation[i, 0])
                pose.p = gymapi.Vec3(*hand_translation[i])
                pose = test_rot * pose
                hand_actor_handle = gym.create_actor(env, self.hand_asset, pose, "leaphand", 0, -1)

                self.hand_handles.append(hand_actor_handle)
                hand_props = gym.get_actor_dof_properties(env, hand_actor_handle)
                hand_props["driveMode"].fill(gymapi.DOF_MODE_POS)
                hand_props["stiffness"].fill(10.0)
                hand_props["damping"].fill(0.1)
                gym.set_actor_dof_properties(env, hand_actor_handle, hand_props)
                cur_dof_states = gym.get_actor_dof_states(env, hand_actor_handle, gymapi.STATE_ALL)
                tar_dof_states = cur_dof_states.copy()
                for idx, joint in enumerate(self.joint_names):
                    joint_idx = gym.find_actor_dof_index(env, hand_actor_handle, joint, gymapi.DOMAIN_ACTOR)

                    cur_dof_states["pos"][joint_idx] = hand_qpos[i, idx]  # .detach().cpu().numpy()
                    if target_qpos is not None:
                        tar_dof_states["pos"][joint_idx] = target_qpos[i, idx]
                    else:
                        tar_dof_states["pos"][joint_idx] = hand_qpos[i, idx]

                gym.set_actor_dof_states(env, hand_actor_handle, cur_dof_states, gymapi.STATE_ALL)

                gym.set_actor_dof_position_targets(env, hand_actor_handle, tar_dof_states["pos"])

                hand_shape_props = gym.get_actor_rigid_shape_properties(env, hand_actor_handle)
                hand_rigid_body_set = set()
                for j in range(gym.get_actor_rigid_body_count(env, hand_actor_handle)):
                    hand_rigid_body_set.add(gym.get_actor_rigid_body_index(env, hand_actor_handle, j, gymapi.DOMAIN_ENV))
                self.hand_rigid_body_sets.append(hand_rigid_body_set)
                for k in range(len(hand_shape_props)):
                    hand_shape_props[k].friction = self.hand_friction
                gym.set_actor_rigid_shape_properties(env, hand_actor_handle, hand_shape_props)

                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(0, 0, 0)
                pose.r = gymapi.Quat(0, 0, 0, 1)
                pose = test_rot * pose
                obj_actor_handle = gym.create_actor(env, self.obj_asset, pose, "obj", 0, 1)
                self.obj_handles.append(obj_actor_handle)
                gym.set_actor_scale(env, obj_actor_handle, obj_scale)
                obj_shape_props = gym.get_actor_rigid_shape_properties(env, obj_actor_handle)
                obj_rigid_body_set = set()
                for l in range(gym.get_actor_rigid_body_count(env, obj_actor_handle)):
                    obj_rigid_body_set.add(gym.get_actor_rigid_body_index(env, obj_actor_handle, l, gymapi.DOMAIN_ENV))
                self.obj_rigid_body_sets.append(obj_rigid_body_set)
                for m in range(len(obj_shape_props)):
                    obj_shape_props[m].friction = self.obj_friction
                gym.set_actor_rigid_shape_properties(env, obj_actor_handle, obj_shape_props)
                gym.end_aggregate(env)

    def run_sim(self):
        gym.prepare_sim(self.sim)
        for _ in range(self.sim_step):
            # step the physics
            gym.simulate(self.sim)
            gym.fetch_results(self.sim, True)
            # refresh tensors
            gym.refresh_rigid_body_state_tensor(self.sim)
            gym.refresh_dof_state_tensor(self.sim)
            gym.refresh_jacobian_tensors(self.sim)
            gym.refresh_mass_matrix_tensors(self.sim)

            if self.has_viewer:
                sleep(self.debug_interval)
                if gym.query_viewer_has_closed(self.viewer):
                    break
                gym.step_graphics(self.sim)
                gym.draw_viewer(self.viewer, self.sim, False)

        success = []
        for i, env in enumerate(self.envs):
            contacts = gym.get_env_rigid_contacts(env)
            flag = False
            for contact in contacts:
                if (contact[2] in self.hand_rigid_body_sets[i]) and (
                        contact[3] in self.obj_rigid_body_sets[i]):
                    flag = True
                    break
                if (contact[3] in self.hand_rigid_body_sets[i]) and (
                        contact[2] in self.obj_rigid_body_sets[i]):
                    flag = True
                    break
            success.append(flag)
        return success

    def reset_simulator(self):
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.viewer)
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX,
                                  self.sim_params)
        for env in self.envs:
            gym.destroy_env(env)
        self.envs = []
        self.hand_handles = []
        self.obj_handles = []
        self.hand_rigid_body_sets = []
        self.obj_rigid_body_sets = []
        self.hand_asset = None
        self.obj_asset = None

    def destroy(self):
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.viewer)