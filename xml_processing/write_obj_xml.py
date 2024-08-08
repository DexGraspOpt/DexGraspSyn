from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import trimesh
# import pybullet as op1
# from pybullet_utils import bullet_client
import multiprocessing as mp
import shutil
import coacd
import open3d as o3d
import numpy as np


def make_obj_dir(root_dir, dst_dir, object_scale_dict=None):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if not file_name.split('.')[1] == 'obj':
                continue
            if '_simplified' in file_name:
                continue
            print(file_name)
            filepath = os.path.abspath(os.path.join(root, file_name))
            folder_name = file_name.split('/')[-1].split('.')[0]
            folder_path = os.path.join(dst_dir, folder_name)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            if object_scale_dict is not None:
                mesh = trimesh.load(filepath, force='mesh')
                mesh.vertices *= object_scale_dict[file_name.split('.')[0]]
                mesh.export(os.path.join(folder_path, file_name))
            else:
                shutil.copy(filepath, folder_path)

def write_convex_obj_file(path_to_obj_file, parts):
    objFile = open(path_to_obj_file, 'w')
    idx = 0
    bais = 0
    for verts, faces in parts:
        objFile.write("o convex_{} \n".format(idx))
        for vert in verts:
            objFile.write("v ")
            objFile.write(str(vert[0]))
            objFile.write(" ")
            objFile.write(str(vert[1]))
            objFile.write(" ")
            objFile.write(str(vert[2]))
            objFile.write("\n")
        for face in faces:
            objFile.write("f ")
            objFile.write(str(face[0] + 1+bais))
            objFile.write(" ")
            objFile.write(str(face[1] + 1+bais))
            objFile.write(" ")
            objFile.write(str(face[2] + 1+bais))
            objFile.write("\n")
        bais += len(verts)
        idx += 1
    objFile.close()


def coacd_decomposition(name_in, name_out, name_log):
    mesh = trimesh.load(name_in, force="mesh")
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    # parts = coacd.run_coacd(coacd_mesh, threshold=0.05, max_convex_hull=12)
    parts = coacd.run_coacd(coacd_mesh)
    for idx, part in enumerate(parts):
        name = name_out.replace('_vhacd.obj', '_cvx_{}.stl'.format(idx))
        verts, faces = part
        tmp_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        tmp_mesh.export(name)

    # parts = coacd.run_coacd(coacd_mesh, threshold=0.03)
    write_convex_obj_file(path_to_obj_file=name_out, parts=parts)


def vhacd_to_piece(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.split('.')[1] == 'obj' and '_vhacd.obj' in file_name and '_vox_' not in file_name:
                file_path = os.path.join(root, file_name)
                meshes = trimesh.load_mesh(file_path)
                mesh_list = meshes.split()
                for i, mesh in enumerate(mesh_list):
                    new_file_path = file_path.replace('vhacd.obj', 'cvx_{}.stl'.format(i))
                    mesh.export(new_file_path)


def create_xml(file_dir_name, root_dir):
    # create the file structure
    data = ET.Element('mujoco')
    data.set('model', 'OBJ')
    compiler = ET.SubElement(data, 'compiler')
    size = ET.SubElement(data, 'size')
    compiler.set('angle', 'radian')
    size.set('njmax', '500')
    size.set('nconmax', '100')
    item_asset = ET.SubElement(data,'asset')

    item_worldbody = ET.SubElement(data,'worldbody')
    item_body = ET.SubElement(item_worldbody, 'body')
    # item_body.set('name', file_dir_name)
    item_body.set('name', 'object')
    item_body.set('pos','0 0 0')
    item_body.set('euler','0 0 0')

    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            # create visual attribute
            if '_cvx_' not in file_name and '_vhacd' not in file_name and file_name.split('.')[-1] == 'obj':
                file_name_ = file_name.split('.')[0]
                if abs_path:
                    file_path = os.path.abspath(os.path.join(root, file_name))
                else:
                    file_path = '../{}/{}'.format(file_name.split('.')[0], file_name)

                item_mesh = ET.SubElement(item_asset, 'mesh')
                item_mesh.set('name', file_name_)
                item_mesh.set('file', file_path)

                item_geom = ET.SubElement(item_body, 'geom')
                item_geom.set('type', 'mesh')
                item_geom.set('density', '0')
                item_geom.set('mesh', file_name_)
                item_geom.set('name', file_name_)
                item_geom.set('contype','0')
                item_geom.set('conaffinity','0')
                item_geom.set('group','1')

            # create collision attribute
            if '.stl' in file_name and 'cvx' in file_name and file_name.split('.')[0].split('_')[-2] == 'cvx':
                file_name_ = file_name.split('.')[0]
                if abs_path:
                    file_path = os.path.abspath(os.path.join(root, file_name))
                else:
                    file_path = '../{}/{}'.format(file_name.split('_')[0], file_name)

                item_mesh = ET.SubElement(item_asset, 'mesh')
                item_mesh.set('name', file_name_)

                item_mesh.set('file', file_path)

                item_geom = ET.SubElement(item_body, 'geom')
                item_geom.set('type','mesh')
                item_geom.set('density','1500')
                item_geom.set('mesh',file_name_)
                item_geom.set('name', file_name_)
                item_geom.set('group', '0')
                item_geom.set('condim', '4')
                item_geom.set('friction', '0.5 0.005 0.0001')
                # item_geom.set('solref', "0.02 1.0")
                # item_geom.set('friction', '10')

    et = ET.ElementTree(data)

    dst_dir = os.path.join(root_dir, '../obj_xml_static')
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    fname = os.path.join(root_dir, '../obj_xml_static/{}.xml'.format(file_dir_name))
    et.write(fname, encoding='utf-8', xml_declaration=True)
    x = minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent='  '))


if __name__ == '__main__':
    src_dir = '../test_data/meshes/'
    dst_dir = '../test_data/xmls/'
    import json
    scale_dict_path = '../test_data/object_scale.json'
    obj_scale_list = json.load(open(scale_dict_path, 'r'))
    obj_scale_dict = {}
    for item in obj_scale_list:
        for key, value in item.items():
            obj_scale_dict[key.split('/')[-1].split('.')[0]] = value
    print(obj_scale_dict)

    # step1: object to folder
    make_obj_dir(src_dir, dst_dir, obj_scale_dict)

    # step2: Mesh Decomposition for collision shapes used by mujoco
    to_vhacd = True
    if to_vhacd:
        pool = mp.Pool(int(mp.cpu_count() / 2))
        for root, dirs, files in os.walk(dst_dir):
            for file_name in files:
                if file_name.endswith('.obj') and '_vox_' not in file_name and "_vhacd" not in file_name:
                    name_in = os.path.join(root, file_name)
                    name_out = name_in.replace('.obj', '_vhacd.obj')
                    name_log = "log.txt"
                    if not os.path.exists(name_out):
                        pool.apply_async(coacd_decomposition, args=(name_in, name_out, name_log,))
        pool.close()
        pool.join()

    # step3: create xml file for mujoco simulation
    abs_path = False
    for root, dirs, files in os.walk(dst_dir):
        for dir_name in dirs:
            src = os.path.join(root, dir_name)
            create_xml(dir_name, src)
            print('{} is ok'.format(src))