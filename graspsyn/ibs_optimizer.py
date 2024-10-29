import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import pytorch3d.ops
import numpy as np
import trimesh
import time
import roma
from tqdm import tqdm
import pytorch_volumetric as pv
import sys
sys.path.append('../')

from hand_layers.leap_hand_layer.leap_layer import LeapHandLayer, LeapAnchor
from hand_layers.allegro_hand_layer.allegro_layer import AllegroHandLayer, AllegroAnchor
from hand_layers.shadow_hand_layer.shadow_layer import ShadowHandLayer, ShadowAnchor
from hand_layers.svh_hand_layer.svh_layer import SvhHandLayer, SvhAnchor
from hand_layers.mano_hand_layer.mano_layer import ManoHandLayer, ManoAnchor

from utils.initializations import initialize_grasp_space
# from utils.initializations_ours import initialize_grasp_space
from utils.loss_utils import point2point_signed, point2point_nosigned
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
# roma quat [x, y, z, w]


class HandOptimizer(nn.Module):
    """Custom Pytorch model for gradient-base grasp optimization.
    """
    def __init__(self, device='cpu', hand_name='leap_hand', hand_params={}, object_params={}, to_mano_frame=True,
                 apply_fc=False, args=None):
        super().__init__()
        self.device = device
        self.hand_name = hand_name
        self.apply_fc = apply_fc

        assert bool(object_params)
        object_params['points'] = torch.from_numpy(object_params['points']).unsqueeze(0).to(self.device)
        object_params['normals'] = torch.from_numpy(object_params['normals']).unsqueeze(0).to(self.device)
        object_mesh_list = [object_params['mesh']]

        # initialize object params
        self.object_name = object_params['name']
        self.object_params = object_params

        self._init_hand_layer(to_mano_frame)
        self.use_mano_frame = self.hand_layer.to_mano_frame
        if not bool(hand_params):
            hand_params = initialize_grasp_space(self.hand_layer, object_mesh_list, args)

        self.bs = hand_params['joint_angles'].shape[0]
        self._init_hand(hand_params)


        self._init_mano_hand()
        
        self.best_wrist_rot = self.init_wrist_rot.clone()
        self.best_wrist_tsl = self.init_wrist_tsl.clone()
        self.best_joint_angles = self.init_joint_angles.clone()

        self.pose = torch.from_numpy(np.identity(4)).reshape(-1, 4, 4).repeat(self.bs, 1, 1).float().to(self.device)
        if self.use_quat:
            self.pose[:, :3, :3] = roma.unitquat_to_rotmat(self.init_wrist_rot)
        else:
            self.pose[:, :3, :3] = robust_compute_rotation_matrix_from_ortho6d(self.init_wrist_rot)
        self.pose[:, :3, 3] = self.init_wrist_tsl

        self.vox_size = 0.01

    def _init_mano_hand(self):
        self.mano_hand = ManoHandLayer(use_pca=False, device=self.device)

    def _init_hand_layer(self, to_mano_frame=True):
        if self.hand_name == 'leap_hand':
            self.hand_layer = LeapHandLayer(to_mano_frame=to_mano_frame, device=self.device)
            self.hand_anchor_layer = LeapAnchor()
        elif self.hand_name == 'allegro_hand':
            self.hand_layer = AllegroHandLayer(to_mano_frame=to_mano_frame, device=self.device)
            self.hand_anchor_layer = AllegroAnchor()
        elif self.hand_name == 'shadow_hand':
            self.hand_layer = ShadowHandLayer(to_mano_frame=to_mano_frame, device=self.device)
            self.hand_anchor_layer = ShadowAnchor()
        elif self.hand_name == 'svh_hand':
            self.hand_layer = SvhHandLayer(to_mano_frame=to_mano_frame, device=self.device)
            self.hand_anchor_layer = SvhAnchor()
        elif self.hand_name == 'mano_hand':
            self.hand_layer = ManoHandLayer(to_mano_frame=to_mano_frame, device=self.device)
            self.hand_anchor_layer = ManoAnchor()
        else:
            # custom hand layer should be added here
            assert NotImplementedError

    def _init_hand(self, hand_params):

        if self.hand_name == 'leap_hand' or self.finger_num == 'allegro_hand':
            self.finger_num = 4
        elif self.hand_name == 'shadow_hand' or self.hand_name == 'svh_hand' or self.hand_name == 'mano_hand':
            self.finger_num = 5
        else:
            # custom hand layer should be specified here
            raise NotImplementedError
        if self.hand_name == 'leap_hand':
            self.remap_finger_name = {'palm': 'palm_lower', 'thumb': 'thumb_fingertip', 'index': 'fingertip', 'middle': 'fingertip_2',
                          'ring': 'fingertip_3'}
            self.inv_remap_finger_name = {v: k for k, v in self.remap_finger_name.items()}

        self.joints_mean = self.hand_layer.joints_mean
        self.joints_range = self.hand_layer.joints_range
        self.finger_indices = self.hand_layer.hand_finger_indices

        self.joints_range = self.joints_range.to(self.device)
        self.joints_mean = self.joints_mean.to(self.device)
        self.hand_dofs = self.joints_range.shape[0]

        # initialize hand pose
        joint_normalized = (hand_params['joint_angles'] - self.joints_mean) / self.joints_range
        joint_angles = torch.atanh(joint_normalized.clamp(min=-1 + 1e-6, max=1 - 1e-6))

        self.use_quat = False  # rot6d seems able to achieve slight better result than quat  rotation representation

        if self.use_quat:
            self.init_wrist_rot = hand_params['wrist_quat']
        else:
            self.init_wrist_rot = hand_params['wrist_rot6d']

        self.init_wrist_tsl = hand_params['wrist_tsl']
        self.init_joint_angles = hand_params['joint_angles']

        # make weights torch parameters
        # if self.hand_name == 'leap_hand':
        #     self.theta = nn.Parameter(joint_angles.view(self.bs, self.hand_layer.n_dofs))
        # elif self.hand_name == 'allegro_hand':
        #     self.theta = nn.Parameter(joint_angles.view(self.bs, self.hand_layer.n_dofs))
        # elif self.hand_name == 'shadow_hand':
        #     self.theta = nn.Parameter(joint_angles.view(self.bs, self.hand_layer.n_dofs))
        # elif self.hand_name == 'svh_hand':
        #     self.theta = nn.Parameter(joint_angles.view(self.bs, self.hand_layer.n_dofs))
        # else:
        #     raise NotImplementedError
        self.theta = nn.Parameter(joint_angles.view(self.bs, self.hand_layer.n_dofs))

        self.wrist_rot = nn.Parameter(self.init_wrist_rot.clone())
        self.wrist_tsl = nn.Parameter(self.init_wrist_tsl.clone().view(self.bs, 3))

        # initialize the optimizer
        self.optimizer = torch.optim.AdamW([
            {'params': self.wrist_rot, 'lr': 0.005},
            {'params': self.wrist_tsl, 'lr': 0.005},
            {'params': self.theta, 'lr': 0.01},
        ], lr=0.01)  # used by allegro

        self.scheduler = StepLR(self.optimizer, step_size=150, gamma=0.9)

        self.theta_decode = None
        self.finger_id = None  # fore_finger

        # to device
        self.joints_lower_limit = self.joints_mean - self.joints_range
        self.joints_upper_limit = self.joints_mean + self.joints_range

        self.index_thumb = torch.zeros(self.bs, dtype=torch.float, device=self.device)
        self.middle_thumb = torch.zeros(self.bs, dtype=torch.float, device=self.device)

        self.n_contact = 6  # 4

        if self.hand_name == 'leap_hand' or self.hand_name == 'allegro_hand':
            valid_mask = torch.tensor([
                True, True, True, True, True,  # Thumb
                True, True,  # [Palm]
                True, True, True, True, True,  # Index
                True,  # [Palm]
                True, True, True, True, True,  # Middle
                True, True,  # [Palm]
                True, True, True, True, True,  # Ring
                False, False,  # [Palm]
                False, False, False, False, False,  # Little
                True, False, True, False,  # Index  Side
                True, False, True, False,  # Middle Side
                True, False, True, False,  # Ring   Side
                False, False  # little
            ])
        elif self.hand_name == 'shadow_hand' or self.hand_name == 'svh_hand' or self.hand_name == 'mano_hand':
            valid_mask = torch.tensor([
                True, True, True, True, True,  # Thumb
                True, True,  # [Palm]
                True, True, True, True, True,  # Index
                True,  # [Palm]
                True, True, True, True, True,  # Middle
                True, True,  # [Palm]
                True, True, True, True, True,  # Ring
                True, True,  # [Palm]
                True, True, True, True, True,  # Little
                True, False, True, False,  # Index  Side
                True, False, True, False,  # Middle Side
                True, False, True, False,  # Ring   Side
                True, False  # little
            ])
        else:
            raise NotImplementedError

        self.contact_idx = torch.tensor([
            0, 1, 2, 3, 4,  # Thumb
            5, 6,  # [Palm]
            7, 8, 9, 10, 11,  # Index
            12,  # [Palm]
            13, 14, 15, 16, 17,  # Middle
            18, 19,  # [Palm]
            20, 21, 22, 23, 24,  # Ring
            25, 26,  # [Palm]
            27, 28, 29, 30, 31,  # Little
            32, 33, 34, 35,  # Index  Side
            36, 37, 38, 39,  # Middle Side
            40, 41, 42, 43,  # Ring   Side
            44, 45,  # little
        ], dtype=torch.long).to(self.device)[valid_mask]

        self.contact_weight = torch.tensor([
            0.5, 1, 1, 0.5, 0.5,  # Thumb
            1.0, 1.0,  # [Palm]
            0.5, 0.5, 0.5, 0.5, 0.5,  # Index
            1.0,  # [Palm]
            0.5, 0.5, 0.5, 0.5, 0.5,  # Middle
            1.0, 1.0,  # [Palm]
            0.5, 0.5, 0.5, 0.5, 0.5,  # Ring
            1.0, 1.0,  # [Palm]
            0.5, 0.5, 0.5, 0.5, 0.5,  # Little
            0.5, 0, 0.5, 0,  # Index  Side
            0.5, 0, 0.5, 0,  # Middle Side
            0.5, 0, 0.5, 0,  # Ring   Side
            0.5, 0,  # little
        ]).to(self.device)[valid_mask]

        # self.contact_weight = torch.ones(len(self.contact_idx)).to(self.device)

        self.fc_transformation_matrix = torch.tensor([
            [0, 0, 0, 0, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, -1, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0]
        ], dtype=torch.float, device=self.device)

        self.dis_weight = 100.0
        self.fc_weight = 50.0
        self.contact_align_weight = 0.5  # 0.5

    def get_hand_verts_and_normal(self, pred, down_sample_rate=2):
        finger_verts = []
        finger_verts_normal = []
        split_indices = []
        count = 0

        split_indices.append(count)
        for key, value in self.finger_indices.items():
            finger_verts.append(pred['vertices'][:, value][:, ::down_sample_rate])
            finger_verts_normal.append(pred['normals'][:, value][:, ::down_sample_rate])
            count += finger_verts[-1].shape[1]
            split_indices.append(count)

        finger_verts = torch.cat(finger_verts, dim=1)
        finger_verts_normal = torch.cat(finger_verts_normal, dim=1)

        return finger_verts, finger_verts_normal, split_indices

    def get_hand_verts_and_normal_v1(self, pred):
        finger_verts = {}
        finger_verts_normal = {}
        if self.finger_num == 4:
            finger_names = ['palm', 'thumb', 'index', 'middle', 'ring']
        elif self.finger_num == 5:
            finger_names = ['palm', 'thumb', 'index', 'middle', 'ring', 'little']

        for (key, value), name in zip(self.finger_indices.items(), finger_names):
            finger_verts[name] = pred['vertices'][:, value]
            finger_verts_normal[name] = pred['normals'][:, value]

        return finger_verts, finger_verts_normal

    def remove_little_finger(self, pred):
        finger_verts = []
        finger_verts_normal = []

        for key, value in self.finger_indices.items():
            finger_verts.append(pred['vertices'][:, value])
            finger_verts_normal.append(pred['normals'][:, value])

        finger_verts = torch.cat(finger_verts, dim=1)
        finger_verts_normal = torch.cat(finger_verts_normal, dim=1)

        return finger_verts, finger_verts_normal

    def compute_grasp_matrix(self, contact_points, contact_normals):
        batch_size, n_contacts, _ = contact_points.shape
        G = torch.zeros((batch_size, 6, n_contacts), dtype=contact_points.dtype, device=contact_points.device)

        G[:, :3, :] = contact_normals.permute(0, 2, 1)
        G[:, 3:, :] = torch.cross(contact_points, contact_normals).permute(0, 2, 1)

        return G

    def decode_theta(self, with_limit=False):
        if with_limit:
            output = (self.joints_mean + self.joints_range * self.theta) % (2 * np.pi)
            output = torch.where(output > np.pi, output - 2 * np.pi, output)
            output = torch.clamp(output, min=self.joints_lower_limit, max=self.joints_upper_limit)
            return output
        else:
            assert not torch.isinf(torch.sum(self.theta)), f'{self.theta} contains an infinity value'
            return self.joints_mean + self.joints_range * torch.tanh(self.theta)

    def compute_self_collision(self, pred):
        finger_verts, finger_verts_normal, splits = self.get_hand_verts_and_normal(pred, down_sample_rate=2)
        self_collision_loss = None

        for i in range(1, len(splits)-2):
            j2i_signed, i2j_signed, _, _, _, _ = point2point_signed(finger_verts[:, splits[i]:splits[i+1]],
                                                              finger_verts[:, splits[i+1]:],
                                                              finger_verts_normal[:, splits[i]:splits[i+1]],
                                                              finger_verts_normal[:, splits[i+1]:])
            j2i_signed_dist_neg = torch.logical_and(j2i_signed.abs() < 0.01, j2i_signed < 0.0)
            i2j_signed_dist_neg = torch.logical_and(i2j_signed.abs() < 0.01, i2j_signed < 0.0)

            if self_collision_loss is None:
                self_collision_loss = torch.sum(i2j_signed * i2j_signed_dist_neg, dim=1)
            else:
                self_collision_loss += torch.sum(i2j_signed * i2j_signed_dist_neg, dim=1)
            self_collision_loss += torch.sum(j2i_signed * j2i_signed_dist_neg, dim=1)

        return self_collision_loss

    def get_finger_belonging_indices(self, hand_idx_pos, finger_indices):
        is_belong = (hand_idx_pos.unsqueeze(1) == finger_indices).any(dim=1)
        return is_belong.nonzero().squeeze(dim=1)

    def compute_ibs_hand(self, hand_vertices, hand_normals, object_points, object_normals, voxel_points,
                         hand_mesh=None, mano=False, vis=False):
        time_start = time.time()
        # print('---------------', hand_vertices.shape)
        sdf_hand, hand_idx = point2point_nosigned(hand_vertices, voxel_points.repeat(self.bs, 1, 1))
        sdf_obj, obj_idx = point2point_nosigned(object_points, voxel_points)

        # diff = (sdf_hand - sdf_obj).cpu().numpy()
        # mask_positive = diff >= 0
        # mask_negative = diff < 0
        # selected_mask = np.abs(diff) < 0.02
        #
        # selected_points = voxel_points.cpu().numpy()[selected_mask]
        #
        # selected_points = trimesh.PointCloud(voxel_points.cpu().numpy()[selected_mask], colors=(255, 255, 0))
        # # pc_positive = trimesh.PointCloud(voxel_points.cpu().numpy()[mask_positive], colors=(0, 255, 255))
        # # pc_negative = trimesh.PointCloud(voxel_points.cpu().numpy()[mask_negative], colors=(255, 0, 255))
        # obj_mesh = self.object_params['mesh']
        # # scene = trimesh.Scene([obj_mesh, pc_negative, pc_positive])
        # scene = trimesh.Scene([obj_mesh, selected_points])
        # scene.show()

        diff = (sdf_obj - sdf_hand)
        positive_mask = torch.logical_and(diff >= 0, diff <= np.sqrt(3)*self.vox_size+1e-4)
        positive_points = voxel_points[positive_mask].reshape(1, -1, 3)
        sdf_diff_pos = diff[positive_mask]
        hand_idx_pos = hand_idx[positive_mask]

        # print(positive_points.shape)
        negative_mask = torch.logical_and(diff >= -np.sqrt(3)*self.vox_size-1e-4, diff < 0)
        negative_points = voxel_points[negative_mask].reshape(1, -1, 3)
        sdf_diff_neg = diff[negative_mask]

        # y2x_signed, x2y_signed, yidx_near, xidx_near, y2x, x2y = point2point_signed(positive_points, negative_points)
        # mask_pos = x2y_signed < (self.vox_size+1e-4)
        # positive_points = positive_points[mask_pos]
        #
        # mask_neg = y2x_signed < (self.vox_size+1e-4)
        # negative_points = negative_points[mask_neg]

        dists, idx, knn = pytorch3d.ops.ball_query(positive_points, negative_points, radius=self.vox_size+1e-4, K=6, return_nn=True)
        valid = dists.sum(dim=-1) > 0
        positive_points = positive_points[valid]
        knn = knn[valid]
        idx = idx[valid]
        sdf_diff_pos = sdf_diff_pos[valid.squeeze()]

        weighted_sum = True  # use weighted sum get better and smoother IBS visualization
        if weighted_sum:
            ret = pytorch3d.ops.utils.masked_gather(sdf_diff_neg.reshape(1, -1, 1), idx.reshape(1, -1, 6)).squeeze(
                dim=-1)
            knn_center_sdf = ret.sum(dim=-1).squeeze()  # (N, )
            knn_center = knn.sum(dim=-2)  # (N, 3)

            idx_mask = idx > -1
            idx_mask_sum = idx_mask.sum(dim=-1)

            knn_center_sdf = knn_center_sdf / idx_mask_sum
            knn_center = knn_center / idx_mask_sum[..., None]

            t = -knn_center_sdf / (sdf_diff_pos - knn_center_sdf)

            ibs = knn_center + t.unsqueeze(dim=-1) * (positive_points-knn_center)
        else:
            knn_center = knn.sum(dim=-2)
            idx_mask = idx > -1
            idx_mask_sum = idx_mask.sum(dim=-1)
            # average points
            ibs = (knn_center + positive_points) / (idx_mask_sum[..., None] + 1)

        dists, idx, knn = pytorch3d.ops.knn_points(ibs.view(self.bs, -1, 3), ibs.view(self.bs, -1, 3), K=5, return_nn=True)

        center_points = knn.mean(2).unsqueeze(2)  # Shape (B, N, 1, 3)
        diff = knn - center_points  # Shape (B, N, k, 3)
        cov_matrices = torch.matmul(diff.transpose(2, 3), diff)  # Shape (B, N, 3, 3)
        cov_matrices_reshaped = cov_matrices.view(-1, 3, 3)
        _, _, vh = torch.svd(cov_matrices_reshaped)
        normals = vh.view(self.bs, knn.shape[1], 3, 3)[:, :, :, -1]  # Shape (B, N, 3)
        # compute the sign direction of the normal
        y2x_signed, _, yidx_near, _, y2x, _ = point2point_signed(hand_vertices, ibs.view(self.bs, -1, 3), hand_normals)
        mask = (-y2x * normals).sum(-1) < 0 & (y2x_signed > 0)
        # # this may help to process situation with slightly hand-object collision in GT
        # mask_b = (-y2x * normals).sum(-1) > 0 & (y2x_signed < 0)
        normals[mask] *= -1
        ibs_normals = normals.squeeze()  # hard code for batch size == 1

        # Note: we might need to remove ibs point inside the object but outside the hand

        if vis:
            origin = ibs.cpu().numpy()
            origin_pc = trimesh.PointCloud(origin, colors=(255, 0, 255))
            normal = normals.cpu().numpy().squeeze()
            ray_visualize = trimesh.load_path(np.hstack((origin,
                                                         origin + normal * 0.025)).reshape(-1, 2, 3))
            scene = trimesh.Scene([origin_pc, ray_visualize])
            scene.show()

        hand_idx_pos = hand_idx_pos[valid.squeeze()]

        # Identity which finger the ibs belong to
        if mano:
            indices = {name: self.get_finger_belonging_indices(hand_idx_pos, indices) for name, indices in
                       self.mano_hand.hand_finger_indices.items()}
        else:
            indices = {name: self.get_finger_belonging_indices(hand_idx_pos, indices) for name, indices in
                       self.hand_layer.hand_finger_indices.items()}

        # Extracting ibs for each finger with ibs normal
        ibs_results = {name: ibs[indices] for name, indices in indices.items()}
        ibs_normals_results = {name: ibs_normals[indices] for name, indices in indices.items()}
        sdf_obj, _ = point2point_nosigned(object_points, ibs.view(self.bs, -1, 3))
        time_end = time.time()
        # print('Time to compute knn points: ', time_end - time_start)
        # print('ibs_results:', ibs_results)

        # dists, idx, knn = pytorch3d.ops.knn_points(negative_points, positive_points, K=3, return_nn=True)
        # print(idx.shape, knn.shape)
        # dist_mask = dists < (self.vox_size + 1e-4)
        # print(dist_mask.shape, knn.shape)
        # # print(dist_mask.sum())
        # points = knn[dist_mask]
        # points = torch.unique(points, dim=0)
        # print(points.shape)

        # merge ibs of little finger to ring finger for leap hand
        if self.finger_num == 4 and mano:
            ibs_results['ring'] = torch.cat([ibs_results['ring'], ibs_results['little']], dim=0)
            ibs_results.pop('little')
        elif self.finger_num < 4:
            raise NotImplementedError('Finger num {} not implemented'.format(self.finger_num))

        if vis:
            # Define finger colors
            finger_colors = {
                'palm': (0, 255, 255),
                'thumb': (255, 255, 0),
                'index': (0, 0, 255),
                'middle': (0, 255, 0),
                'ring': (255, 0, 255),
                'little': (255, 0, 0),
            }
            pc = []
            for name, ibs_part in ibs_results.items():
                if len(ibs_part) == 0:
                    continue
                if mano:
                    pc_finger = trimesh.PointCloud(ibs_part.cpu().detach().numpy(), colors=finger_colors[name])
                else:
                    print(name, self.inv_remap_finger_name[name])
                    pc_finger = trimesh.PointCloud(ibs_part.cpu().detach().numpy(),
                                                   colors=finger_colors[self.inv_remap_finger_name[name]])
                pc.append(pc_finger)

            obj_mesh = self.object_params['mesh']
            if hand_mesh:
                scene = trimesh.Scene([obj_mesh, hand_mesh, *pc])
            else:
                pc_hand = trimesh.PointCloud(hand_vertices[0].cpu().detach().numpy(), colors=(0, 25, 255))
                scene = trimesh.Scene([obj_mesh, pc_hand, *pc])
            scene.show()

        return ibs_results, ibs_normals_results, sdf_obj

    def compute_ibs_loss(self, ibs_mano, ibs_hand):
        loss = 0.0
        for name, value in ibs_mano.items():
            if len(ibs_hand[self.remap_finger_name[name]]) > 0:
                if name == 'thumb':
                    weight = 200
                else:
                    weight = 100
                dist_forward, dist_backward, _, _, _, _ = point2point_signed(ibs_mano[name].reshape(self.bs, -1, 3), ibs_hand[self.remap_finger_name[name]].reshape(self.bs, -1, 3))
                loss += (dist_forward.sum()/dist_forward.shape[1] + dist_backward.sum()/dist_backward.shape[1]) * weight
        return loss

    def forward(self, iteration=0, obstacle=None, debug=False):
        """
        Implement function to be optimised.
        """
        pose = self.pose.clone()
        if self.use_quat:
            quat = F.normalize(self.wrist_rot)
            pose[:, :3, :3] = roma.unitquat_to_rotmat(quat)
        else:
            pose[:, :3, :3] = robust_compute_rotation_matrix_from_ortho6d(self.wrist_rot)

        pose[:, :3, 3] = self.wrist_tsl
        theta = self.decode_theta()

        pred_vertices, pred_normals = self.hand_layer.get_forward_vertices(pose, theta)

        pred = {'vertices': pred_vertices, 'normals': pred_normals}

        loss_collision_obstacle = 0
        if obstacle is not None:
            _, h2o_signed, _, _, _, _ = point2point_signed(
                pred['vertices'], obstacle['points'].repeat(self.bs, 1, 1), pred['normals'],
                obstacle['normals'].repeat(self.bs, 1, 1))

            h2o_dist_neg = torch.logical_and(h2o_signed.abs() < 0.05, h2o_signed < 0.0)
            loss_collision_obstacle = torch.sum(h2o_signed * h2o_dist_neg, dim=1) * -20
        # torch.cuda.synchronize()
        # time_start = time.time()
        # hand object collision
        o2h_signed, h2o_signed, _, obj_near_idx, o2h_vec, h2o_vec = point2point_signed(
            pred['vertices'], self.object_params['points'].repeat(self.bs, 1, 1),
            pred['normals'], self.object_params['normals'].repeat(self.bs, 1, 1),
        )

        o2h_dist_neg = torch.logical_and(o2h_signed.abs() < 0.005, o2h_signed < 0.0)
        h2o_dist_neg = torch.logical_and(h2o_signed.abs() < 0.005, h2o_signed < 0.0)

        loss_collision_h2o = torch.sum(h2o_signed * h2o_dist_neg, dim=1)
        loss_collision_o2h = torch.sum(o2h_signed * o2h_dist_neg, dim=1)

        hand_obj_collision = -20 * (1*loss_collision_h2o + 10 * loss_collision_o2h)  # 75
        # hand_obj_collision = -200 * loss_collision_o2h  # 75
        # torch.cuda.synchronize()
        # time_cost = time.time() - time_start
        # print('time cost', time_cost)
        # hand self collision
        if self.hand_name == 'parallel_hand':  # there is no self collision with parallel jaw gripper
            hand_self_collision = 0
        else:
            hand_self_collision = -100 * self.compute_self_collision(pred)  # 60 as default

        if iteration == 499:
            flag = True
        else:
            flag = False

        # ibs loss
        ibs_hand = self.compute_ibs_hand(pred_vertices, pred_normals,
                                         self.object_params['points'].repeat(self.bs, 1, 1),
                                         self.object_params['normals'].repeat(self.bs, 1, 1),
                                         self.voxel_points, vis=flag
                                         )

        ibs_loss = self.compute_ibs_loss(self.ibs_mano, ibs_hand) * 10

        # fingertip_loss = self.compute_fingertip_loss(mano_fingertip, pred_fingertip)

        # hand rot loss
        if self.use_quat:
            hand_rot_loss = (1 - (quat * self.init_wrist_rot).sum(-1) ** 2)
        else:
            hand_rot_loss = roma.rotmat_geodesic_distance(pose[:, :3, :3], self.pose[:, :3, :3]) * 0.2

        # abnormal joint angle loss  (hand specific loss)
        angle_loss = self.hand_layer.compute_abnormal_joint_loss(theta)

        total_cost = (hand_obj_collision + hand_self_collision + hand_rot_loss + loss_collision_obstacle + angle_loss + ibs_loss)
        return total_cost

    def inference(self, return_anchors=False):
        with torch.no_grad():
            pose = torch.from_numpy(np.identity(4)).to(self.device).reshape(-1, 4, 4).float()
            if self.use_quat:
                wrist_quat = F.normalize(self.best_wrist_rot)
            else:
                wrist_rot_tmp = robust_compute_rotation_matrix_from_ortho6d(self.best_wrist_rot)
                wrist_quat = roma.rotmat_to_unitquat(wrist_rot_tmp)

            pose[0, :3, :3] = roma.unitquat_to_rotmat(wrist_quat)
            pose[0, :3, 3] = self.best_wrist_tsl
            theta_decode = self.best_joint_angles
            pred_vertices, _ = self.hand_layer.get_forward_vertices(pose, theta_decode)
            pred_anchors = self.hand_anchor_layer(pred_vertices)
        if return_anchors:
            return pred_vertices, pred_anchors
        return pred_vertices

    def best_grasp_configuration(self, save_real=False):
        # get best hand parameters
        assert self.best_wrist_rot is not None
        assert self.best_wrist_tsl is not None
        assert self.best_joint_angles is not None

        if self.use_quat:
            wrist_quat = F.normalize(self.best_wrist_rot)
        else:
            wrist_rot_tmp = robust_compute_rotation_matrix_from_ortho6d(self.best_wrist_rot)
            wrist_quat = roma.rotmat_to_unitquat(wrist_rot_tmp)
        wrist_tsl = self.best_wrist_tsl.clone()
        if save_real:
            if self.use_mano_frame:
                pose_mano_frame = torch.eye(4).reshape(-1, 4, 4).float().repeat(self.bs, 1, 1).to(self.device)
                pose_mano_frame[:, :3, :3] = roma.unitquat_to_rotmat(wrist_quat)
                pose_mano_frame[:, :3, 3] = self.best_wrist_tsl.clone()
                pose_hand_frame = torch.matmul(pose_mano_frame, self.hand_layer.base_2_world)
                wrist_quat_xyzw = roma.rotmat_to_unitquat(pose_hand_frame[:, :3, :3])
                wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_quat_xyzw)
                wrist_tsl = pose_hand_frame[:, :3, 3]
            else:
                wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_quat)
        else:
            wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_quat)

        joint_angles = self.best_joint_angles.cpu().numpy()

        grasp_dict = {'wrist_quat': wrist_quat_wxyz.cpu().numpy(),
                      'wrist_tsl': wrist_tsl.cpu().numpy(),
                      'joint_angles': joint_angles,
                      "obj_scale": self.object_params['scale'],
                      'filepath': self.object_params['filepath']
                      }
        return grasp_dict

    def last_grasp_configuration(self, save_real=False):
        # get current hand parameters
        if self.use_quat:
            wrist_quat = F.normalize(self.best_wrist_rot)
        else:
            wrist_rot_tmp = robust_compute_rotation_matrix_from_ortho6d(self.best_wrist_rot)
            wrist_quat = roma.rotmat_to_unitquat(wrist_rot_tmp)
        wrist_tsl = self.wrist_tsl.detach()
        joint_angles = self.decode_theta(with_limit=False).detach()

        if save_real:
            if self.use_mano_frame:
                pose_mano_frame = torch.eye(4).reshape(-1, 4, 4).float().repeat(self.bs, 1, 1).to(self.device)
                pose_mano_frame[:, :3, :3] = roma.unitquat_to_rotmat(wrist_quat)
                pose_mano_frame[:, :3, 3] = wrist_tsl.clone()
                pose_hand_frame = torch.matmul(pose_mano_frame, self.hand_layer.base_2_world)
                wrist_quat_xyzw = roma.rotmat_to_unitquat(pose_hand_frame[:, :3, :3])
                wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_quat_xyzw)
                wrist_tsl = pose_hand_frame[:, :3, 3]
            else:
                wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_quat)
        else:
            wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_quat)

        grasp_dict = {'wrist_quat': wrist_quat_wxyz.cpu().numpy(),
                      'wrist_tsl': wrist_tsl.cpu().numpy(),
                      'joint_angles': joint_angles.cpu().numpy(),
                      "obj_scale": self.object_params['scale'],
                      'filepath': self.object_params['filepath']
                      }
        return grasp_dict

    def optimize(self, obstacle=None, mano_config={}, n_iters=1000):
        query_range = np.array([
            [-0.3, 0.3],
            [-0.3, 0.3],
            [-0.15, 0.15],
        ])
        coords, pts = pv.get_coordinates_and_points_in_grid(self.vox_size, query_range, device=self.device)

        # make the center of the pts lie in the center of hand and object
        mano_config['vertices'] -= torch.from_numpy(self.object_params['offset_center']).to(self.device).float()
        mano_config['mesh'].vertices -= self.object_params['offset_center']
        print(mano_config['vertices'].shape)
        pc = trimesh.PointCloud(mano_config['vertices'].cpu().numpy().squeeze())
        scene = trimesh.Scene([pc, mano_config['mesh']])
        scene.show()


        self.voxel_points = pts.reshape(1, -1, 3)

        self.ibs_mano = self.compute_ibs_hand(mano_config['vertices'], mano_config['normals'],
                                              self.object_params['points'],
                                              self.object_params['normals'],
                                              self.voxel_points, hand_mesh=mano_config['mesh'],
                                              mano=True, vis=True)

        min_loss = 1e8
        for iter_step in tqdm(range(n_iters + 1), desc='hand optimize process'):
            loss = self.forward(iter_step, obstacle)

            if iter_step >= 0:
                loss_mask = loss < min_loss
                min_loss = torch.where(loss_mask, loss, min_loss)
                nonzero_idx = torch.nonzero(loss_mask, as_tuple=True)[0]
                if not torch.numel(nonzero_idx) == 0:
                    with torch.no_grad():
                        if self.use_quat:
                            self.best_wrist_rot[nonzero_idx] = F.normalize(self.wrist_rot[nonzero_idx], dim=1).clone().detach()  # .cpu().squeeze().numpy()
                        else:
                            self.best_wrist_rot[nonzero_idx] = self.wrist_rot[nonzero_idx].clone().detach()
                        self.best_wrist_tsl[nonzero_idx] = self.wrist_tsl[nonzero_idx].clone().detach()  # .cpu().squeeze().numpy()
                        self.best_joint_angles[nonzero_idx] = self.decode_theta(
                            with_limit=False)[nonzero_idx].clone().detach()  # .cpu().squeeze().numpy()
            self.optimizer.zero_grad()
            loss.mean().backward()

            self.optimizer.step()
            self.scheduler.step()

            print('{}-th iter: {}'.format(iter_step, loss.mean().item()))
