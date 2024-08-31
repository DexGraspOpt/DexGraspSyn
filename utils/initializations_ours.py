"""
modified from: DexGraspNet  https://github.com/PKU-EPIC/DexGraspNet/blob/main/grasp_generation/utils/initializations.py
Author: Jialiang Zhang, Ruicheng Wang
Description: initializations
"""

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


def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)


def initialize_grasp_space(hand_model, object_mesh_list, args):
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
    print(n_objects)
    batch_size_each = args.batch_size_each
    total_batch_size = n_objects * batch_size_each

    # initialize translation and rotation

    translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)
    # print('n_objects', n_objects)
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
        else:
            # use offsetMesh
            mesh_ori = object_mesh_list[i].copy()

            closedMesh = mrn.meshFromFacesVerts(mesh_ori.faces, mesh_ori.vertices)

            params = mrmeshpy.OffsetParameters()
            params.voxelSize = 0.01
            posOffset = mrmeshpy.offsetMesh(closedMesh, 0.01, params)

            out_verts = mrn.getNumpyVerts(posOffset)
            out_faces = mrn.getNumpyFaces(posOffset.topology)
            mesh = tm.Trimesh(vertices=out_verts, faces=out_faces)

        vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.float, device=device)
        mesh_pytorch3d = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

        # sample points
        point_num = min(10000, 10 * batch_size_each)
        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_pytorch3d, num_samples=point_num)
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
        process_theta = args.theta_lower + (args.theta_upper - args.theta_lower) * torch.rand([batch_size_each],
                                                                                              dtype=torch.float,
                                                                                              device=device)
        # solve transformation
        # rotation_hand: rotate the hand to align its grasping direction with the +z axis
        # rotation_local: jitter the hand's orientation in a cone
        # rotation_global and translation: transform the hand to a position corresponding to point p sampled from the inflated convex hull

        rotation_local = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)
        rotation_global = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)
        radius = 0.05
        random_sign = np.random.choice([-1, 1])
        for j in range(batch_size_each):
            rotation_local[j] = torch.tensor(
                transforms3d.euler.euler2mat(process_theta[j], deviate_theta[j], 0, axes='sxyz'),
                dtype=torch.float, device=device)

            point_of_interest = p[j].cpu().numpy()
            vertices = dense_point_cloud.cpu().numpy()[0]

            distances = np.linalg.norm(vertices - point_of_interest, axis=1)
            neighborhood_indices = np.where(distances < radius)[0]
            neighborhood_points = vertices[neighborhood_indices]

            cov_matrix = np.cov(neighborhood_points.T)

            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            sorted_indices = np.argsort(eigenvalues)[::-1]

            sorted_eigenvectors = eigenvectors[:, sorted_indices]

            rotation_global[j] = torch.from_numpy(sorted_eigenvectors).to(device)

        # transform based on z direction
        mask = (rotation_global[:, :3, 2] * n).sum(dim=-1) < 0
        rotation_global[mask, :3, 0] *= -1
        rotation_global[mask, :3, 2] *= -1

        if random_sign == 1:
            rotation_hand = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi/2,  np.deg2rad(60), axes='szxz'),
                                         dtype=torch.float, device=device)
        else:
            rotation_hand = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi/2, -np.deg2rad(120), axes='szxz'),
                                         dtype=torch.float, device=device)

        rotation[i * batch_size_each: (i + 1) * batch_size_each] = rotation_global @ rotation_local @ rotation_hand
        translation[i * batch_size_each: (i + 1) * batch_size_each] = p - distance.unsqueeze(1) * (
                rotation_global @ rotation_local @ torch.tensor([0, 0, 1], dtype=torch.float,
                                                                device=device).reshape(1, -1, 1)).squeeze(2) - (
                rotation_global @ rotation_hand @ torch.tensor([0.02, 0.00, 0], dtype=torch.float,
                                                                device=device).reshape(1, -1, 1)).squeeze(2)

    # initialize joint angles
    # joint_angles_mu: hand-crafted canonicalized hand articulation
    # use truncated normal distribution to jitter the joint angles
    joint_angles_mu = hand_model.get_init_angle()

    joint_angles_sigma = args.jitter_strength * (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros([total_batch_size, hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i],
                                    hand_model.joints_lower[i] + 1e-6, hand_model.joints_upper[i] - 1e-6)
    # joint_angles[:, [1, 5, 9]] = 0

    hand_params = {'joint_angles': joint_angles, 'wrist_tsl': translation,
                   'wrist_rot6d': rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
                   'wrist_quat': roma.rotmat_to_unitquat(rotation),
                   'parallel_contact_point': None
                   }
    return hand_params

    # hand_pose = torch.cat([
    #     translation,
    #     rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
    #     joint_angles
    # ], dim=1)
    # hand_pose.requires_grad_()
    #
    # # initialize contact point indices
    #
    # contact_point_indices = torch.randint(hand_model.n_contact_candidates, size=[total_batch_size, args.n_contact],
    #                                       device=device)
    #
    # hand_model.set_parameters(hand_pose, contact_point_indices)


if __name__ == "__main__":
    import glob
    from hand_layers.leap_hand_layer.leap_layer import LeapHandLayer
    import argparse
    from utils.seed_utils import set_seed
    from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size_each", type=int, default=100)
    parser.add_argument("--distance_upper", type=float, default=0.05)
    parser.add_argument("--distance_lower", type=float, default=0.05)
    parser.add_argument("--jitter_strength", type=float, default=0.1)
    parser.add_argument('--theta_lower', default=-np.pi/6, type=float)
    parser.add_argument('--theta_upper', default=np.pi/6, type=float)
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

    allegro = LeapHandLayer(show_mesh=show_mesh,  to_mano_frame=True, device=device)


    hand_params = initialize_grasp_space(allegro, object_mesh_list, args)
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    for i in range(args.batch_size_each):
        pose[0, :3, :3] = robust_compute_rotation_matrix_from_ortho6d(hand_params['wrist_rot6d'])[i]
        pose[0, :3, 3] = hand_params['wrist_tsl'][i]
        import trimesh.creation

        coordinate_frame = trimesh.creation.axis(origin_size=0.01, transform=pose[0].cpu().numpy())

        # theta = np.zeros((1, 16), dtype=np.float32)
        # theta = torch.from_numpy(theta).to(device)
        theta = hand_params['joint_angles'][i]
        # theta[0, 14] = 1.0
        # theta[0, 15] = 1.0
        verts, normals = allegro.get_forward_vertices(pose, theta)
        pc = trimesh.PointCloud(verts[0].squeeze().cpu().numpy(), colors=(0, 255, 255))

        obj_mesh = object_mesh_list[0]
        scene = trimesh.Scene([obj_mesh, pc, coordinate_frame])
        scene.show()


