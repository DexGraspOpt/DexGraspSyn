import os

import torch
import numpy as np
import sys
import roma
import trimesh
import glob
sys.path.append('../')

from hand_layers.allegro_hand_layer.allegro_layer import AllegroHandLayer
from hand_layers.shadow_hand_layer.shadow_layer import ShadowHandLayer
from hand_layers.svh_kinematics.svh_layer import SvhHandLayer



category = 'spray_bottle'
intention = 'TU'
hand_name = 'shadow_hand'  # ['shadow_hand', 'allegro_hand', 'svh_hand']
if hand_name == 'allegro_hand':
    hand_layer = AllegroHandLayer(to_mano_frame=True, show_mesh=True, device='cpu')
elif hand_name == 'shadow_hand':
    hand_layer = ShadowHandLayer(to_mano_frame=True, show_mesh=True, device='cpu')
elif hand_name == 'svh_hand':
    hand_layer = SvhHandLayer(to_mano_frame=True, show_mesh=True).to('cpu')
else:
    # Only above three hands are supported now (BarrettHand and HIT DLR II hand are not supported now)
    raise ValueError(f'hand_name {hand_name} not recognized')

DEXFG_DATA_DIR = '/home/v-wewei/data/dexfg_grasp_dataset_new/'  # PATH to the dexfg dataset
Grasp_DIR = os.path.join(DEXFG_DATA_DIR, category, intention, hand_name)

files = glob.glob(Grasp_DIR+'/*.npy')
for filepath in files:
    grasp_filename = filepath.split('/')[-1]
    grasp = np.load(filepath, allow_pickle=True).item()

    if intention == 'TU':
        object_name = grasp_filename[:-4] + '.obj'
    else:
        object_name = grasp_filename[:-6] + '.obj'
    object_filepath = os.path.join(DEXFG_DATA_DIR, category, object_name)
    assert os.path.isfile(object_filepath), object_filepath

    obj_mesh = trimesh.load(object_filepath, force='mesh')
    scale = grasp['obj_scale']
    obj_mesh.vertices *= scale

    pose = torch.eye(4).reshape(1, 4, 4)
    pose[:, :3, :3] = roma.unitquat_to_rotmat(torch.from_numpy(grasp['hand_quat']).to(torch.float))
    pose[:, :3, 3] = torch.from_numpy(grasp['hand_tsl']).to(torch.float)

    theta = torch.from_numpy(grasp['joint_angles']).to(torch.float).reshape(1, -1)

    hand_mesh = hand_layer.get_forward_hand_mesh(pose, theta)[0]
    hand_mesh.visual.face_colors = np.array([0, 255, 255])

    (hand_mesh + obj_mesh).show()