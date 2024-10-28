import os
import torch
import numpy as np
import roma
import trimesh
from easydict import EasyDict as edict
import glob

from graspsyn.ibs_optimizer import HandOptimizer
from utils.object_utils import get_object_params
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from utils.seed_utils import set_seed

from hand_layers.mano_hand_layer.mano_layer import ManoHandLayer
from loguru import logger
from urdfpy import URDF
from scipy.spatial.transform import Rotation as R
from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig


def get_leap_hand_mesh(joint_angles):
    robot = URDF.load('/home/v-wewei/repository/dex-retargeting/assets/robots/hands/leap_hand/leap_hand_v2_right.urdf')
    joint_names = []
    for joint in robot.actuated_joints:
        joint_names.append(joint.name)
    print(joint_names)
    cfg = {}
    for joint_name, joint_angle in zip(joint_names, joint_angles):
        cfg[joint_name] = joint_angle

    fk = robot.forward_kinematics(cfg, use_collision=True)

    meshes = []
    link_pose = {}
    for mesh in fk:
        pose, name = fk[mesh]
        link_pose[name] = pose
        mesh.apply_transform(pose)
        meshes.append(mesh)
    hand_mesh = np.sum(meshes)

    return hand_mesh, link_pose


def compute_offset_to_arctic(mano_out, wrist_pose):
    center = mano_out.joints[:, 0:1]
    offset = (wrist_pose[:, :3, :3] @ -center.transpose(1, 2)).transpose(1, 2) + center + mano_out.center_joint
    return offset


def create_retargeting_config():
    robot_name: RobotName = RobotName.leapv2
    retargeting_type: RetargetingType = RetargetingType.pose
    hand_type: HandType = HandType.right

    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)

    robot_dir = "/home/v-wewei/repository/dex-retargeting/assets/robots/hands"

    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    # config = RetargetingConfig.load_from_file(config_path)
    return retargeting


def load_mano_params(device):
    arctic_seq_path = "./test_data/mano_datasets/arctic_seq"
    seq_id = "s01"
    seq_name = "capsulemachine_use_01"
    seq_frame_number = 342

    seq_path = os.path.join(arctic_seq_path, seq_id, seq_name + '.npy')
    seq_params = np.load(seq_path, allow_pickle=True).item()

    # Get the hand pose and vertices
    rot_vec = seq_params["mano_params"][seq_frame_number]["rot_r"]
    tran = seq_params["mano_params"][seq_frame_number]["tran_r"]
    theta = seq_params["mano_params"][seq_frame_number]["pose_r"]
    betas = seq_params["mano_params"][seq_frame_number]["shape_r"]
    pose = np.eye(4)
    pose[:3, :3] = R.from_rotvec(rot_vec).as_matrix()
    pose[:3, 3] = tran
    pose = torch.from_numpy(pose).reshape(1, 4, 4).float().to(device)
    theta = torch.from_numpy(theta).reshape(-1, 45).to(device)
    betas = torch.from_numpy(betas).reshape(-1, 10).to(device)
    return pose, theta, betas


if __name__ == "__main__":
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    set_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lTm = np.eye(4)
    lTm[:3, :3] = np.array([[0.00, -1.00, 0.00],
                            [0.00, 0.00, 1.00],
                            [-1.00, 0.00, 0.00]])

    hand_name = 'leap_hand'
    bs = 1
    opt_args = edict({'batch_size_each': bs, 'distance_lower': 0.15, 'distance_upper': 0.25,
                      'jitter_strength': 0.1, "theta_lower": -np.pi / 6, 'theta_upper': np.pi / 6})

    mesh_dir = './test_data/meshes/'
    filepath_list = glob.glob('{}/*.obj'.format(mesh_dir))

    mano_hand = ManoHandLayer(use_pca=False, device=device)

    mano_wrist_pose, theta_mano, betas = load_mano_params(device)
    points, normals, mano_out = mano_hand.get_forward_vertices(mano_wrist_pose, theta_mano.clone(),
                                                               return_mano_output=True, transform_wrt_center_joint=True)
    mano_mesh = trimesh.Trimesh(mano_out.verts[0].cpu().numpy(), mano_hand.chain.th_faces.cpu().numpy(),
                                vertex_colors=np.array([[255, 224, 189]]).repeat(778, 0))
    offset = compute_offset_to_arctic(mano_out, mano_wrist_pose)  # (Bs, 1, 3)
    offset_matrix = trimesh.transformations.translation_matrix(offset.cpu().numpy().squeeze())

    mano_mesh.apply_transform(mano_wrist_pose[0].cpu().numpy())
    mano_mesh.apply_transform(offset_matrix)
    mano_mesh.show()  # this mesh should be overlap with the mano mesh of arctic dataset

    mano_config = {'vertices': points, 'normals': normals, 'mesh': mano_mesh}

    translation_matrix = trimesh.transformations.translation_matrix(-mano_out.joints.cpu().numpy().squeeze()[0])
    transform_mano = trimesh.transformations.concatenate_matrices(lTm, translation_matrix, np.linalg.inv(mano_wrist_pose.cpu().numpy().squeeze()), )

    retargeting = create_retargeting_config()
    dex_retarget = True
    if dex_retarget:
        retargeting_type = retargeting.optimizer.retargeting_type
        indices = retargeting.optimizer.target_link_human_indices
        joint_pos = mano_out.joints.cpu().numpy().squeeze()
        joint_pos = trimesh.transform_points(joint_pos, lTm)
        joint_pos -= joint_pos[0]  # center to 0-th hand keypoint
        for _ in range(10):
            if retargeting_type == 'POSE':
                indices = indices
                bul_axes_glb = mano_hand.get_bul_axes_glb(mano_out)
                bul_axes_nor = torch.matmul(bul_axes_glb[0][0:1].transpose(2, 1), bul_axes_glb[0])[:, :3, :3]
                bul_axes_nor[15] = torch.matmul(bul_axes_nor[15],
                                                roma.rotvec_to_rotmat(torch.tensor([0, -np.pi/6, -np.pi / 12])).to(device))
                bul_axes_nor = roma.rotmat_to_rotvec(bul_axes_nor)

                fingertip_rot = bul_axes_nor[[15, 3, 6, 12]].cpu().numpy()

                fingertip_pos = joint_pos[indices]
                joint_pose = np.concatenate([fingertip_pos, fingertip_rot], axis=1)

                ref_value = joint_pose
            else:
                assert False, f"Unexpected retargeting type {retargeting_type}"
            robot_hand_q = retargeting.retarget(ref_value)
            logger.info(f"qpos {robot_hand_q}")

    order_1 = ["tx", "ty", "tz", "rx", "ry", "rz", "1", "0", "2", "3", "12", "13", "14", "15", "5", "4", "6", "7", "9",
               "8", "10", "11"]
    order_2 = ['1', '0', '2', '3', '5', '4', '6', '7', '9', '8', '10', '11', '12', '13', '14', '15']
    order_3 = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', '1', '12', '9', '5', '13', '8', '4', '0', '14', '10', '2', '6', '7',
               '3', '11', '15']
    reorder = [order_1.index(item) for item in order_3]
    joint_angles = robot_hand_q[reorder]

    # _, link_pose = get_leap_hand_mesh(joint_angles)

    index = retargeting.optimizer.get_link_indices(['palm_lower'])
    palm_pose = retargeting.optimizer.robot.get_link_pose(index[0])

    wrist_pose = trimesh.transformations.concatenate_matrices(offset_matrix, np.linalg.inv(transform_mano), palm_pose)

    # please note it is hand specific params
    wrist_T_mano = np.linalg.inv(np.array([[-1, 0, 0, 0],
                                          [0, 0, 1, 0.0175],
                                          [0, 1, 0, 0.0375],
                                           [0, 0, 0, 1]]))
    wrist_pose = np.matmul(wrist_pose, wrist_T_mano)
    wrist_pose = torch.from_numpy(wrist_pose).view(-1, 4, 4).to(device).float()
    reorder = [order_1.index(item) for item in order_2]
    joint_angles = robot_hand_q[reorder]

    vis = True
    if vis:
        from hand_layers.leap_hand_layer.leap_layer import LeapHandLayer
        leap_layer = LeapHandLayer(show_mesh=True, to_mano_frame=True, device=device)
        leap_hand_mesh = leap_layer.get_forward_hand_mesh(wrist_pose, torch.from_numpy(joint_angles).view(-1, 16).to(device).to(torch.float))[0]
        scene = trimesh.Scene([mano_mesh,  leap_hand_mesh])
        scene.show()
    retargeting_params = {'joint_angles': torch.from_numpy(joint_angles).view(-1, 16).to(device).to(torch.float),
                          'wrist_tsl': torch.tensor(wrist_pose[:, :3, 3]).view(-1, 3).to(device),
                          'wrist_rot6d': torch.from_numpy(wrist_pose[:, :3, :3]).view(-1, 3, 3).transpose(1, 2)[:, :2].reshape(-1, 6).to(device),
                          }

    for obj_filepath in filepath_list:

        object_params = get_object_params(obj_filepath, vis=False)
        obj_name = obj_filepath.split('/')[-1].split('.')[0]

        hand_opt = HandOptimizer(device=device, hand_name=hand_name, hand_params=retargeting_params,
                                 object_params=object_params,
                                 apply_fc=False, args=opt_args)

        hand_opt.optimize(obstacle=None, mano_config=mano_config, n_iters=500)

        grasp = hand_opt.best_grasp_configuration(save_real=False)
        # grasp = hand_opt.last_grasp_configuration(save_real=False)
        grasp_real = hand_opt.best_grasp_configuration(save_real=True)
        # np.save('./test_data/grasp_npy/{}.npy'.format(obj_name), grasp_real)
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

            # exit()

            # show grasp and hand anchors
            pose = torch.eye(4).reshape(1, 4, 4).repeat(opt_args.batch_size_each, 1, 1).to(device).float()
            theta = torch.from_numpy(grasp['joint_angles']).to(device).reshape(-1, hand_opt.hand_layer.n_dofs)
            pose[:, :3, :3] = roma.unitquat_to_rotmat(torch.from_numpy(grasp['wrist_quat'][:, [1, 2, 3, 0]]).to(device))
            pose[:, :3, 3] = torch.from_numpy(grasp['wrist_tsl']).to(device)
            verts, verts_normal = hand_opt.hand_layer.get_forward_vertices(pose, theta)
            anchors = hand_opt.hand_anchor_layer.forward(verts)

            for idx in range(opt_args.batch_size_each):
                # if not (idx == 55):
                #     continue
                pc = trimesh.PointCloud(verts[idx].squeeze().cpu().numpy(), colors=(0, 255, 255))
                pc_anchor = trimesh.PointCloud(anchors[idx].squeeze().cpu().numpy(), colors=(255, 0, 0))
                pc_init = trimesh.PointCloud(verts_init[idx].squeeze().cpu().numpy(), colors=(255, 0, 255))
                scene = trimesh.Scene([pc, pc_anchor, pc_init, object_params['mesh'], mano_mesh])
                scene.show()
