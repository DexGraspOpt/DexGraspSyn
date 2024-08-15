import os
import torch
import numpy as np
import roma
import trimesh
from easydict import EasyDict as edict
import glob

from graspsyn.hand_optimizer import HandOptimizer
from utils.object_utils import get_object_params
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from utils.seed_utils import set_seed


if __name__ == "__main__":
    set_seed(0)
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hand_name = 'shadow_hand'

    opt_args = edict({'batch_size_each': 10, 'distance_lower': 0.05, 'distance_upper': 0.1,
                      'jitter_strength': 0.1, "theta_lower": -np.pi / 6, 'theta_upper': np.pi / 6})

    mesh_dir = './test_data/meshes/'
    filepath_list = glob.glob('{}/*.obj'.format(mesh_dir))

    for obj_filepath in filepath_list:

        object_params = get_object_params(obj_filepath)
        obj_name = obj_filepath.split('/')[-1].split('.')[0]

        hand_opt = HandOptimizer(device=device, hand_name=hand_name, hand_params={}, object_params=object_params,
                                 apply_fc=False, args=opt_args)
        hand_opt.optimize(obstacle=None, n_iters=200)

        grasp = hand_opt.best_grasp_configuration(save_real=False)
        # grasp = hand_opt.last_grasp_configuration(save_real=False)
        # grasp_real = hand_opt.best_grasp_configuration(save_real=True)
        # np.save(obj_name + '.npy', grasp_real)
        vis_grasp = True
        if vis_grasp:
            # init grasp
            pose = torch.eye(4).reshape(1, 4, 4).repeat(opt_args.batch_size_each, 1, 1).to(device).float()
            theta = hand_opt.init_joint_angles.reshape(-1, hand_opt.hand_layer.n_dofs)
            if hand_opt.use_quat:
                pose[:, :3, :3] = roma.unitquat_to_rotmat(hand_opt.init_wrist_rot)
            else:  # use rot6d representation
                pose[:, :3, :3] = robust_compute_rotation_matrix_from_ortho6d(hand_opt.init_wrist_rot)
            pose[:, :3, 3] = hand_opt.init_wrist_tsl
            verts_init, verts_normal_init = hand_opt.hand_layer.get_forward_vertices(pose, theta)

            # show grasp and hand anchors
            pose = torch.eye(4).reshape(1, 4, 4).repeat(opt_args.batch_size_each, 1, 1).to(device).float()
            theta = torch.from_numpy(grasp['joint_angles']).to(device).reshape(-1, hand_opt.hand_layer.n_dofs)
            pose[:, :3, :3] = roma.unitquat_to_rotmat(torch.from_numpy(grasp['wrist_quat'][:, [1, 2, 3, 0]]).to(device))
            pose[:, :3, 3] = torch.from_numpy(grasp['wrist_tsl']).to(device)
            verts, verts_normal = hand_opt.hand_layer.get_forward_vertices(pose, theta)
            anchors = hand_opt.hand_anchor_layer.forward(verts)

            for idx in range(opt_args.batch_size_each):
                pc = trimesh.PointCloud(verts[idx].squeeze().cpu().numpy(), colors=(0, 255, 255))
                pc_anchor = trimesh.PointCloud(anchors[idx].squeeze().cpu().numpy(), colors=(255, 0, 0))
                pc_init = trimesh.PointCloud(verts_init[idx].squeeze().cpu().numpy(), colors=(255, 0, 255))
                scene = trimesh.Scene([pc, pc_anchor, pc_init, object_params['mesh']])
                scene.show()