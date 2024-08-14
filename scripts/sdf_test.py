import pytorch_volumetric as pv
import trimesh
import coacd
import numpy as np
import torch
from xml_processing.write_obj_xml import write_convex_obj_file

# mesh = trimesh.load('../test_data/meshes/tmph0h9jpn1.obj')

# coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
# # parts = coacd.run_coacd(coacd_mesh, threshold=0.05, max_convex_hull=12)
# parts = coacd.run_coacd(coacd_mesh, threshold=0.1)
#
# meshes = []
# for part in parts:
#     mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])
#     meshes.append(mesh)
# new_mesh = trimesh.boolean.boolean_manifold(meshes, 'union')
# new_mesh.export('tmph0h9jpn1_vhacd.obj')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
obj = pv.MeshObjectFactory("../test_data/meshes/tmph0h9jpn1_simplified.obj")
sdf = pv.MeshSDF(obj)
# caching the SDF via a voxel grid to accelerate queries
cached_sdf = pv.CachedSDF('drill', resolution=0.001, range_per_dim=obj.bounding_box(padding=0.1), gt_sdf=sdf, device=device)


# get points in a grid in the object frame
query_range = np.array([
    [-0.2, 0.2],
    [-0.2, 0.2],
    [-0.2, 0.2],
])

import time

coords, pts = pv.get_coordinates_and_points_in_grid(0.001, query_range)
print(pts.shape)
pts = pts.to(device)

# time_start = time.time()
# a = (pts - torch.tensor([0.1, 0.1, 0.1], device=device)) // 0.005
#
# time_cost = time.time() - time_start
# print('time cost', time_cost)
# print(pts.device)
# exit()


time_start = time.time()
# we can also query with batched points B x N x 3, B can be any number of batch dimensions
sdf_val, sdf_grad = cached_sdf(pts)
time_cost = time.time() - time_start
print('time cost', time_cost)


# sdf_val is N, or B x N, the SDF value in meters
# sdf_grad is N x 3 or B x N x 3, the normalized SDF gradient (points along steepest increase in SDF)