
import os
import open3d as o3d
import trimesh
import numpy as np
import point_cloud_utils as pcu
from mesh_to_sdf import get_surface_point_cloud
import coacd

from utils.grasp_group import GraspGroup, Grasp
from urdfpy import URDF
import torch
import roma


# def set_init_joint_mu(hand_model):
#     if hand_model.name == 'leap_hand':
#         joint_angles_mu = (hand_model.joints_upper - hand_model.joints_lower) / 6.0 + hand_model.joints_lower
#         joint_angles_mu[1] = -0.1
#         joint_angles_mu[5] = 0.0
#         joint_angles_mu[9] = 0.1
#         joint_angles_mu[12] = 0.8
#     elif hand_model.name == 'allegro_hand':
#         joint_angles_mu = (hand_model.joints_upper - hand_model.joints_lower) / 6.0 + hand_model.joints_lower
#         joint_angles_mu[0] = 0.1
#         joint_angles_mu[4] = 0.0
#         joint_angles_mu[8] = -0.1
#         joint_angles_mu[12] = 0.8
#     elif hand_model.name == 'shadow_hand':
#         joint_angles_mu = torch.tensor([-0.15, 0, 0.6, 0, 0, 0, 0.6, 0, -0.15, 0, 0.6, 0, 0, -0.25, 0, 0.6, 0,
#                                         0, 1.2, 0, 0.0, 0], dtype=torch.float, device=hand_model.device)
#     elif hand_model.name == 'svh_hand':
#         joint_angles_mu = torch.tensor([0.4, 0.0,  # thumb
#                                             0.15, # ring
#                                             0.5,  # spread
#                                             0.15,  # little
#                                             0.0, 0.15,  # index
#                                             0.0, 0.15,  # middle
#                                             ], dtype=torch.float, device=hand_model.device)
#     else:
#         raise NotImplementedError
#     return joint_angles_mu


def get_leap_hand_mesh(joint_angles):
    robot = URDF.load('/home/v-wewei/repository/dex-retargeting/assets/robots/hands/leap_hand/leap_hand_right.urdf')

    joint_names = []
    for joint in robot.actuated_joints:
        joint_names.append(joint.name)
    print(joint_names)

    cfg = {}
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        cfg[joint_name] = joint_angle

    fk = robot.forward_kinematics(cfg, use_collision=True)

    meshes = []

    link_pose = []
    for mesh in fk:
        pose, name = fk[mesh]
        if 'head' in name:
            link_pose.append(pose)
        mesh.apply_transform(pose)
        meshes.append(mesh)
    hand_mesh = np.sum(meshes)

    return hand_mesh, link_pose


def get_grasp_group(grasp_filepath):
    grasp_array = np.load(grasp_filepath).reshape(-1, 17)
    SCORE_THRESHOLD = 0.3
    scores = grasp_array[:, 0]
    mask = scores > SCORE_THRESHOLD
    grasp_array = grasp_array[mask]

    # grasp_array[:, 1] += 0.0
    grasp_array[:, 3] += 0.005
    grasp_group = GraspGroup(grasp_array)
    grasp_group.sort_by_score()
    grasp_group = grasp_group.nms(translation_thresh=0.015, rotation_thresh=45.0 / 180.0 * np.pi)
    return grasp_group


def parallel_to_leap(grasp_group, device):
    local_transform_matrix = np.array([[1, 0, 0, -.1],
                                       [0, 0, -1, .02],
                                       [0, 1, 0, -0.04],
                                       [0, 0, 0, 1]])
    rot_z = trimesh.transformations.rotation_matrix(np.deg2rad(30), [0, 0, 1])
    # local_transform_matrix = np.array([[1, 0, 0, -.15],
    #                                    [0, 0, -1, .04],
    #                                    [0, 1, 0, -0.04],
    #                                    [0, 0, 0, 1]])
    # rot_z = trimesh.transformations.rotation_matrix(np.deg2rad(30), [0, 0, 1])

    local_transform_matrix = trimesh.transformations.concatenate_matrices(rot_z, local_transform_matrix)

    rotation_matrix = trimesh.transformations.euler_matrix(0, -np.pi/2, 0, axes='sxyz')
    rotation_matrix[:3, 3] = np.array([0, 0.038, 0.00])

    rot = grasp_group.rotation_matrices
    trans = grasp_group.translations

    transform = np.eye(4).reshape(1, 4, 4).repeat(len(grasp_group), 0)
    transform[:, :3, :3] = rot
    transform[:, :3, 3] = trans

    transform = transform @ local_transform_matrix.reshape(1, 4, 4) @ rotation_matrix.reshape(-1, 4, 4)

    transform = torch.from_numpy(transform.astype(np.float32)).to(device)

    bs = transform.shape[0]
    joint_angles = torch.zeros((bs, 16)).to(device)

    joint_angles[:, 0] = -0.1
    joint_angles[:, 2] = 0.8
    joint_angles[:, 3] = 0.4
    joint_angles[:, 12] = 1.0
    joint_angles[:, 13] = 0.5
    joint_angles[:, 14] = 0.3
    joint_angles[:, 15] = 0.2

    center_point = grasp_group.translations + grasp_group.depths.reshape(-1, 1) * grasp_group.rotation_matrices[:, :, 0]
    fingertip_shift = grasp_group.widths.reshape(-1, 1) / 2 * grasp_group.rotation_matrices[:, :, 1]

    left_point, right_point = center_point - fingertip_shift, center_point + fingertip_shift
    left_point = torch.from_numpy(left_point.astype(np.float32)).to(device)
    right_point = torch.from_numpy(right_point.astype(np.float32)).to(device)

    hand_params = {'joint_angles': joint_angles, 'wrist_tsl': transform[:, :3, 3],
                   'wrist_rot6d': transform[:, :3, :3].transpose(1, 2)[:, :2].reshape(-1, 6),
                   'wrist_quat': roma.rotmat_to_unitquat(transform[:, :3, :3]),
                   'parallel_contact_point': torch.cat([left_point, right_point], dim=1),
                   }
    return hand_params, transform


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    object_name = 'tmpfvthwtwg'
    filename = '/home/v-wewei/code/parallel_grasp_annotation/asset/mujoco_asset/{}/{}.obj'.format(object_name, object_name)
    # print(filename)
    assert os.path.exists(filename)
    o3d_mesh = o3d.io.read_triangle_mesh(filename=filename)
    o3d_mesh.compute_vertex_normals()

    filename = '/home/v-wewei/code/parallel_grasp_annotation/grasp_annotation/{}_grasp_group.npy'.format(object_name)
    grasp_group = get_grasp_group(filename)
    hand_params, transform = parallel_to_leap(grasp_group, device)

    # transform = torch.eye(4).reshape(1, 4, 4).repeat(len(grasp_group), 1, 1).to(device)

    # leap_mesh, _ = get_leap_hand_mesh(np.zeros(16))
    # leap_mesh.apply_transform(transform[0].cpu().numpy())
    # leap_mesh = leap_mesh.as_open3d
    # leap_mesh.compute_vertex_normals()

    from hand_layers.leap_hand_layer.leap_layer import LeapHandLayer
    leap_layer = LeapHandLayer(to_mano_frame=False, show_mesh=True, device=device)

    pose = transform
    # pose = torch.eye(4).reshape(1, 4, 4).repeat(len(grasp_group), 1, 1).to(device)
    theta = hand_params['joint_angles']

    select_idx = 100

    grasp = grasp_group[select_idx]
    gripper = grasp.to_open3d_geometry()

    left_point, right_point = hand_params['parallel_contact_point'][select_idx, :3], hand_params['parallel_contact_point'][select_idx, 3:]
    fingertip_sphere_left = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    vertices = np.asarray(fingertip_sphere_left.vertices)
    vertices += left_point.cpu().numpy()
    fingertip_sphere_left.vertices = o3d.utility.Vector3dVector(vertices)
    fingertip_sphere_left.compute_vertex_normals()
    fingertip_sphere_left.paint_uniform_color([0, 1, 0])
    fingertip_sphere_right = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    vertices = np.asarray(fingertip_sphere_right.vertices)
    vertices += right_point.cpu().numpy()
    fingertip_sphere_right.vertices = o3d.utility.Vector3dVector(vertices)
    fingertip_sphere_right.compute_vertex_normals()
    fingertip_sphere_right.paint_uniform_color([0, 0, 1])

    hand_mesh = leap_layer.get_forward_hand_mesh(pose[select_idx:select_idx+1], theta[select_idx:select_idx+1])[0]
    hand_mesh = hand_mesh.as_open3d
    hand_mesh.compute_vertex_normals()
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([o3d_mesh, gripper, hand_mesh, coordinate_frame, fingertip_sphere_left, fingertip_sphere_right])
