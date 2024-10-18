import os
os.environ['PATH'] = os.environ['HOME']+'/code:' + os.environ['PATH']
import trimesh
import numpy as np
import point_cloud_utils as pcu
from mesh_to_sdf import get_surface_point_cloud
from scipy.spatial import KDTree


def get_stable_pose(mesh):
    # NOTE: watertight object mesh is needed for compute stable pose
    if mesh.is_watertight:
        pass
    else:
        pitch = mesh.extents.max() / 128  # size
        if pitch < 0.002:
            pitch = 0.002
        vox = mesh.voxelized(pitch)
        vox.fill()
        bounds = vox.bounds

        mesh = vox.marching_cubes
        mesh.vertices -= mesh.bounds[0]
        mesh.vertices *= pitch
        mesh.vertices += bounds[0]

    poses = mesh.compute_stable_poses(n_samples=1)[0]
    return poses


def get_object_params(mesh_filepath, vox_size=0.006, scale=1.0, vis=False, watertight_process=True, **kwargs):
    mesh = trimesh.load(mesh_filepath, force='mesh', process=False, skip_material=True)

    # scale the object mesh
    mesh.vertices *= scale

    # normalize to the center
    bbmin = mesh.vertices.min(0)
    bbmax = mesh.vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    mesh.vertices -= center  # center
    mesh_orin = mesh.copy()
    if watertight_process:
        if mesh.is_watertight:
            pass
        else:
            pitch = mesh.extents.max() / 128  # size
            if pitch < 0.002:
                pitch = 0.002
            use_binvox = False
            if mesh.faces.shape[0] > 100:
                use_binvox = True
                # change it to binvox method for better and speed up
                vox = mesh.voxelized(pitch, 'binvox')
            else:
                vox = mesh.voxelized(pitch)
            vox.fill()
            bounds = vox.bounds

            mesh = vox.marching_cubes
            if use_binvox:
                mesh.vertices -= (mesh.bounds[0]-0)  # 0.5
            else:
                mesh.vertices -= mesh.bounds[0]
            mesh.vertices *= pitch
            mesh.vertices += bounds[0]
            # vw, fw = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, resolution=50000)
            # mesh = trimesh.Trimesh(vertices=vw, faces=fw)

    # points, face_index = trimesh.sample.sample_surface(mesh, 50000)
    # normals = mesh.face_normals[face_index]
    # v = np.array(points).astype(np.float32)
    # n = np.array(normals).astype(np.float32)
    point_cloud = get_surface_point_cloud(mesh, scan_count=25, scan_resolution=150, sample_point_count=50000)
    v = point_cloud.points.astype(np.float32)
    n = point_cloud.normals.astype(np.float32)

    sizeof_voxel = vox_size
    # Downsample a point cloud on a voxel grid so there is at most one point per voxel.
    # Any arguments after the points are treated as attribute arrays and get averaged within each voxel
    v_sampled = pcu.downsample_point_cloud_on_voxel_grid(sizeof_voxel, v)
    v_sampled = pcu.downsample_point_cloud_on_voxel_grid(sizeof_voxel, v_sampled)

    kdtree = KDTree(data=v)
    dist, index = kdtree.query(v_sampled, k=1)
    v_sampled = v[index]
    n_sampled = n[index]

    if vis:
        print('points shape is :', v_sampled.shape)
        import open3d as o3d
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(v_sampled)
        o3d_pc.normals = o3d.utility.Vector3dVector(n_sampled)
        o3d_mesh = mesh.as_open3d
        o3d_mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([o3d_pc, o3d_mesh])

    write_ply = True
    if write_ply:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(v_sampled)
        CUR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        o3d.io.write_point_cloud('{}/test_data/ply/{}.ply'.format(CUR_DIR, mesh_filepath.split('/')[-1].split('.')[0]), pc)

    object_params = {'points': v_sampled,
                     'normals': n_sampled,
                     'filepath': mesh_filepath,
                     'name': mesh_filepath.split('/')[-1].split('.')[0],
                     'mesh': mesh.copy(),  # scaled mesh
                     'scale': scale,
                     }
    return object_params


def create_table_points(lx, ly, lz, dx=0, dy=0, dz=0, grid_size=0.01):
    '''
    **Input:**
    - lx:
    - ly:
    - lz:
    **Output:**
    - numpy array of the points with shape (-1, 3).
    '''
    xmap = np.linspace(0, lx, int(lx/grid_size))
    ymap = np.linspace(0, ly, int(ly/grid_size))
    zmap = np.linspace(0, lz, int(lz/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, ymap, zmap], axis=-1)
    points = points.reshape([-1, 3])
    return points


def create_plane_points_with_normal(lx, ly, grid_size):
    points = create_table_points(lx, ly, grid_size, dx=-lx/2, dy=-ly/2, dz=0, grid_size=grid_size)
    normals = np.ones_like(points)
    normals[:, :2] = 0
    return np.concatenate([points, normals], axis=1)


if __name__ == '__main__':
    # plane_points = create_plane_points_with_normal(0.15, 0.15, 0.01)
    # pc = trimesh.PointCloud(plane_points[:, :3], colors=(0, 255, 255))
    # pc.show()
    import objaverse
    objects = objaverse.load_objects(uids=['917f8aaadff04833a8601caaa2b76d95'])
    for uid, obj_filepath in objects.items():
        get_object_params(
            mesh_filepath=obj_filepath,
            scale=0.05851385052149179)
        # obj_mesh = trimesh.load(obj_filepath, force='mesh')
        # face_count = obj_mesh.faces.shape[0]
        # print(face_count)

        # obj_mesh.apply_scale(0.05851385052149179)
        # obj_mesh.show()
        # exit()


    get_object_params(mesh_filepath='/media/v-wewei/T01/objaverse/hf-objaverse-v1/glbs/000-121/a9e49d03467c47a0bf39931a2a8ac6aa.glb',
                      scale=0.003811418377331815)
