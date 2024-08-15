"""
modified from: DexGraspNet  https://github.com/PKU-EPIC/DexGraspNet/blob/main/grasp_generation/utils/initializations.py
Origin Author: Jialiang Zhang, Ruicheng Wang
Description: initializations
"""
import os

import torch
import roma
import transforms3d
import pytorch3d
import pytorch3d.structures
import pytorch3d.ops
import math
import trimesh

import trimesh as tm
import numpy as np
from meshlib import mrmeshpy
from meshlib import mrmeshnumpy as mrn

# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def initialize_grasp_space(hand_model, object_mesh_list, args,):
    """
    Initialize grasp translation, rotation, joint angles, and contact point indices

    Parameters
    ----------
    hand_model: hand_model.HandModel
    object_model: object_model.ObjectModel
    args: Namespace
    """

    device = hand_model.device
    n_objects = len(object_mesh_list)
    batch_size_each = args.batch_size_each
    total_batch_size = n_objects * batch_size_each

    # initialize translation and rotation
    translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)

    for i in range(n_objects):
        # get inflated convex hull
        object_mesh_list[i].remove_degenerate_faces()
        use_cvx = False
        if use_cvx:
            mesh_ori = object_mesh_list[i].convex_hull
            vertices = mesh_ori.vertices.copy()
            faces = mesh_ori.faces

            vertices += 0.2 * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

            mesh_tmp = tm.Trimesh(vertices=vertices, faces=faces)
            mesh = mesh_tmp.convex_hull
            # (mesh_ori + mesh).show()
        else:
            # use offsetMesh
            mesh_ori = object_mesh_list[i].copy()

            closedMesh = mrn.meshFromFacesVerts(mesh_ori.faces, mesh_ori.vertices)

            params = mrmeshpy.OffsetParameters()
            params.voxelSize = 0.01
            posOffset = mrmeshpy.offsetMesh(closedMesh, 0.025, params)

            out_verts = mrn.getNumpyVerts(posOffset)
            out_faces = mrn.getNumpyFaces(posOffset.topology)
            mesh = tm.Trimesh(vertices=out_verts, faces=out_faces)
            # (mesh_ori + mesh).show()
        vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.float, device=device)
        mesh_pytorch3d = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

        # sample points

        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_pytorch3d, num_samples=100 * batch_size_each)
        p = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=batch_size_each)[0][0]

        closest_points, _, _ = mesh_ori.nearest.on_surface(p.detach().cpu().numpy())
        closest_points = torch.tensor(closest_points, dtype=torch.float, device=device)
        n = (closest_points - p) / (closest_points - p).norm(dim=1).unsqueeze(1)

        vis = False
        if vis:
            pc = trimesh.PointCloud(p.detach().cpu().numpy(), colors=(0, 255, 255))
            # create some rays
            ray_origins = p.detach().cpu().numpy()
            ray_directions = (closest_points - p).detach().cpu().numpy()
            # stack rays into line segments for visualization as Path3D
            ray_visualize = trimesh.load_path(
                np.hstack((ray_origins, ray_origins + ray_directions)).reshape(-1, 2, 3)
            )
            scene = trimesh.Scene([mesh_ori, mesh, pc, ray_visualize])
            scene.show()

        # sample parameters

        distance = args.distance_lower + (args.distance_upper - args.distance_lower) * torch.rand([batch_size_each],
                                                                                                  dtype=torch.float,
                                                                                                  device=device)

        deviate_theta = args.theta_lower + (args.theta_upper - args.theta_lower) * torch.rand([batch_size_each],
                                                                                              dtype=torch.float,
                                                                                              device=device)
        process_theta = 2 * math.pi * torch.rand([batch_size_each], dtype=torch.float, device=device)
        rotate_theta = 2 * math.pi * torch.rand([batch_size_each], dtype=torch.float, device=device)

        # solve transformation
        # rotation_hand: rotate the hand to align its grasping direction with the +z axis
        # rotation_local: jitter the hand's orientation in a cone
        # rotation_global and translation: transform the hand to a position corresponding to point p sampled from the inflated convex hull

        rotation_local = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)
        rotation_global = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)
        for j in range(batch_size_each):
            rotation_local[j] = torch.tensor(
                transforms3d.euler.euler2mat(process_theta[j], deviate_theta[j], rotate_theta[j], axes='rzxz'),
                dtype=torch.float, device=device)
            rotation_global[j] = torch.tensor(
                transforms3d.euler.euler2mat(math.atan2(n[j, 1], n[j, 0]) - math.pi / 2, -math.acos(n[j, 2]), 0,
                                             axes='rzxz'), dtype=torch.float, device=device)
        translation[i * batch_size_each: (i + 1) * batch_size_each] = p - distance.unsqueeze(1) * (
                    rotation_global @ rotation_local @ torch.tensor([0, 0, 1], dtype=torch.float,
                                                                    device=device).reshape(1, -1, 1)).squeeze(2)
        rotation_hand = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 2, 0, axes='rzxz'), dtype=torch.float,
                                     device=device)
        rotation[i * batch_size_each: (i + 1) * batch_size_each] = rotation_global @ rotation_local @ rotation_hand

    # initialize joint angles
    # joint_angles_mu: hand-crafted canonicalized hand articulation
    # use truncated normal distribution to jitter the joint angles
    # joint_angles_mu = set_init_joint_mu(hand_model)
    joint_angles_mu = hand_model.get_init_angle()

    joint_angles_sigma = args.jitter_strength * (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros([total_batch_size, hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i],
                                    hand_model.joints_lower[i] + 1e-6, hand_model.joints_upper[i] - 1e-6)

    hand_params = {'joint_angles': joint_angles, 'wrist_tsl': translation,
                   'wrist_rot6d': rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
                   'wrist_quat': roma.rotmat_to_unitquat(rotation),  # [x, y, z, w]
                   'parallel_contact_point': None  # use for mapping from parallel jaw gripper to multi-finger hand
                   }
    return hand_params


if __name__ == "__main__":
    import glob
    from hand_layers.leap_hand_layer.leap_layer import LeapHandLayer
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size_each", type=int, default=1)
    parser.add_argument("--distance_upper", type=float, default=0.2)
    parser.add_argument("--distance_lower", type=float, default=0.1)
    parser.add_argument("--jitter_strength", type=float, default=0.1)
    parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
    parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
    parser.add_argument("--mesh_dir", type=str, default='/home/v-wewei/data/test_meshes')
    args = parser.parse_args()

    mesh_files = glob.glob(args.mesh_dir + '/*.obj')
    # print(mesh_files)
    object_mesh_list = []
    object_scale_list = []
    for filename in mesh_files:
        print(filename)
        mesh = trimesh.load(filename)
        object_mesh_list.append(mesh)
        object_scale_list.append(1.0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    show_mesh = False
    make_contact_points = False

    leap = LeapHandLayer(show_mesh=show_mesh, to_mano_frame=True, device=device)

    hand_params = initialize_grasp_space(leap, object_mesh_list, args)
    # print(hand_params['wrist_rot'].shape)
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    pose[0, :3, :3] = roma.unitquat_to_rotmat(hand_params['wrist_quat'])[0]
    pose[0, :3, 3] = hand_params['wrist_tsl'][0]

    theta = hand_params['joint_angles'][:1]

    verts, normals = leap.get_forward_vertices(pose, theta)
    pc = trimesh.PointCloud(verts.squeeze().cpu().numpy(), colors=(0, 255, 255))

    obj_mesh = object_mesh_list[0]
    scene = trimesh.Scene([obj_mesh, pc])
    scene.show()


