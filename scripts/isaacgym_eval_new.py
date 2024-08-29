"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: validate grasps on Isaac simulator
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.isaac_validator_new import IsaacValidator
import argparse
import torch
import numpy as np
import transforms3d
import roma
import trimesh

from hand_layers.leap_hand_layer.leap_layer import LeapHandLayer
from utils.object_utils import get_object_params
from utils.loss_utils import point2point_signed


if __name__ == '__main__':

    HOME_DIR = os.environ['HOME']
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--val_batch', default=250, type=int)
    parser.add_argument('--mesh_path', default="{}/data/grasp_diffusion_data/".format(HOME_DIR), type=str)
    parser.add_argument('--grasp_path', default="{}/data/acronym_grasp_leap_hand".format(HOME_DIR), type=str)
    parser.add_argument('--result_path', default="{}/data/acronym_grasp_leap_hand_refine".format(HOME_DIR), type=str)
    parser.add_argument('--object_code',
                        default="5b0c679eb8a2156c4314179664d18101",
                        type=str)
    # if index is received, then the debug mode is on
    parser.add_argument('--index',  type=int)
    parser.add_argument('--thres_cont', default=0.0025, type=float)
    parser.add_argument('--dis_move', default=0.001, type=float)
    parser.add_argument('--grad_move', default=250, type=float)
    parser.add_argument('--penetration_threshold', default=0.004, type=float)

    args = parser.parse_args()

    os.makedirs(args.result_path, exist_ok=True)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    data_dict = np.load(os.path.join(
        args.grasp_path, args.object_code + '.npy'), allow_pickle=True).item()

    batch_size = data_dict['wrist_quat'].shape[0]
    hand_state = torch.from_numpy(data_dict['joint_angles']).to(device).requires_grad_()
    scale_tensor = torch.tensor(data_dict['obj_scale']).reshape(1, -1).to(device)
    hand_layer = LeapHandLayer(to_mano_frame=False, device=device)

    pose = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1).to(device).float()
    quat = torch.from_numpy(data_dict['wrist_quat'][:, [1, 2, 3, 0]]).to(device)

    pose[:, :3, :3] = roma.unitquat_to_rotmat(quat)
    pose[:, :3, 3] = torch.from_numpy(data_dict['wrist_tsl']).to(device)
    verts, verts_normal = hand_layer.get_forward_vertices(pose, hand_state)

    pred = {'vertices': verts, 'normals': verts_normal}

    # object model
    obj_params = get_object_params(data_dict['filepath'], scale=data_dict['obj_scale'], vox_size=0.001, vox_watertight=False)

    obj_params['points'] = torch.from_numpy(obj_params['points']).to(device)
    obj_params['normals'] = torch.from_numpy(obj_params['normals']).to(device)

    # calculate contact points and contact normals
    contact_points_hand = torch.zeros((batch_size, len(hand_layer.order_keys), 3)).to(device)
    contact_normals = torch.zeros((batch_size, len(hand_layer.order_keys), 3)).to(device)

    o2h_signed, h2o_signed, hand_near_idx, obj_near_idx, _, _ = point2point_signed(
        pred['vertices'], obj_params['points'].repeat(batch_size, 1, 1), pred['normals'],
        obj_params['normals'].repeat(batch_size, 1, 1), use_cosine_collision=True)

    o2h_dist_neg = torch.logical_and(o2h_signed.abs() <= 0.01, o2h_signed < 0.0)

    o2h = o2h_signed * o2h_dist_neg
    min_o2h = torch.min(o2h_signed * o2h_dist_neg, dim=1)[0]
    # print(min_o2h)
    estimated = (torch.abs(min_o2h) < args.penetration_threshold).cpu().numpy()

    # print(estimated)
    vis = False
    selected_idx = torch.min(torch.min(o2h, dim=1)[0], dim=0)[1].item()

    if vis:
        pc = trimesh.PointCloud(verts[selected_idx].detach().cpu().numpy(), colors=[0, 255, 255])
        pc_obj = trimesh.PointCloud(obj_params['points'].detach().cpu().numpy(), colors=[255, 0, 255])
        ray_visualize = trimesh.load_path(np.hstack((verts[selected_idx].detach().cpu().numpy(),
                                                     verts[selected_idx].detach().cpu().numpy() + verts_normal[selected_idx].detach().cpu().numpy() * 0.01)).reshape(-1, 2, 3))
        o2h_dist_neg = torch.logical_and(o2h_signed.abs() < 0.05, o2h_signed < 0.0)[selected_idx]

        collision_obj_point = obj_params['points'][o2h_dist_neg].detach().cpu().numpy()
        pc_obj_collision = trimesh.PointCloud(collision_obj_point, colors=[255, 0, 0])

        h2o_dist_neg = torch.logical_and(h2o_signed.abs() < 0.05, h2o_signed < 0.0)[selected_idx]
        collision_hand_point = verts[selected_idx][[h2o_dist_neg]].detach().cpu().numpy()
        pc_hand_collision = trimesh.PointCloud(collision_hand_point, colors=[0, 0, 255])

        scene = trimesh.Scene([obj_params['mesh'], pc, pc_obj, pc_hand_collision, ray_visualize, pc_obj_collision])
        scene.show()

    for i, link_name in enumerate(hand_layer.order_keys):

        if link_name == 'palm_lower':
            continue

        segment_indices = hand_layer.hand_segment_indices[link_name]
        if len(segment_indices) == 0:
            continue

        nearest_distances, nearest_indices = h2o_signed[:, segment_indices].min(dim=1)

        nearest_points_hand = torch.gather(pred['vertices'][:, segment_indices], 1, nearest_indices.reshape(batch_size, 1, 1).expand(-1, 1, 3))

        obj_indices = torch.gather(obj_near_idx[:, segment_indices], 1, nearest_indices.reshape(-1, 1))
        nearest_normals = -torch.gather(obj_params['normals'].repeat(batch_size, 1, 1), 1, obj_indices.reshape(-1, 1, 1).expand(-1, 1, 3))

        admited = nearest_distances < args.thres_cont
        admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
        contact_points_hand[:, i:i + 1, :] = torch.where(
            admited, nearest_points_hand, contact_points_hand[:, i:i+1, :])

        contact_normals[:, i:i + 1, :] = torch.where(
            admited, nearest_normals, contact_normals[:, i:i+1, :])


    target_points = contact_points_hand + contact_normals * args.dis_move
    loss = (target_points.detach().clone() - contact_points_hand).square().sum()

    loss.backward()
    with torch.no_grad():
        # print(hand_state.grad[0])
        hand_state -= hand_state.grad * args.grad_move
        hand_state.grad.zero_()

        if vis:
            pose = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1).to(device).float()
            theta = hand_state.reshape(-1, 16)
            pose[:, :3, :3] = roma.unitquat_to_rotmat(
                torch.from_numpy(data_dict['wrist_quat'][:, [1, 2, 3, 0]]).to(device))
            pose[:, :3, 3] = torch.from_numpy(data_dict['wrist_tsl']).to(device)
            verts, verts_normal = hand_layer.get_forward_vertices(pose, theta)

            pred = {'vertices': verts, 'normals': verts_normal}
            pc = trimesh.PointCloud(verts[selected_idx].detach().cpu().numpy(), colors=[0, 255, 255])
            pc_obj = trimesh.PointCloud(obj_params['points'].cpu().numpy(), colors=[255, 0, 255])
            scene = trimesh.Scene([obj_params['mesh'], pc, pc_obj])
            scene.show()

    sim = IsaacValidator(gpu=args.gpu, joint_names=hand_layer.joint_names)

    data_dict = np.load(os.path.join(args.grasp_path, args.object_code + '.npy'), allow_pickle=True).item()

    only_estimated_simulate = True
    simulated = np.zeros(batch_size, dtype=np.bool_)
    sim.set_asset("/home/v-wewei/repository/dex-urdf/robots/hands/leap_hand",
                  "leap_hand_right_no_base_transform.urdf",
                  os.path.join(args.mesh_path, 'xmls/obj_xml_static'), "{}.xml".format(args.object_code))
    if only_estimated_simulate:
        estimated_size = estimated.sum()

        rotations = data_dict['wrist_quat'][estimated]
        translations = data_dict['wrist_tsl'][estimated]
        hand_poses = data_dict['joint_angles'][estimated]
        scale = 1.0

        target_hand_poses = hand_state.detach().cpu().numpy()[estimated]

        sim.create_envs(rotations, translations, hand_poses, scale, target_hand_poses)
        result = sim.run_sim()

        for i, value in enumerate(np.where(estimated)[0]):
            simulated[value] = np.array(sum(result[i * 6:(i + 1) * 6]) == 6)
    else:
        estimated_size = data_dict['wrist_quat'].shape[0]

        rotations = data_dict['wrist_quat']
        translations = data_dict['wrist_tsl']
        hand_poses = data_dict['joint_angles']
        scale = 1.0

        target_hand_poses = hand_state.detach().cpu().numpy()

        sim.create_envs(rotations, translations, hand_poses, scale, target_hand_poses)
        result = sim.run_sim()

        for i in range(estimated_size):
            simulated[i] = np.array(sum(result[i * 6:(i + 1) * 6]) == 6)

    valid = estimated * simulated

    print(f'estimated: {estimated.sum().item()}/{batch_size}, '
          f'simulated: {simulated.sum().item()}/{estimated_size}, '
          f'valid: {valid.sum().item()}/{batch_size}')

    data_dict['success'] = valid
    sim.destroy()

    np.save(os.path.join(args.result_path, args.object_code +'.npy'), data_dict)
