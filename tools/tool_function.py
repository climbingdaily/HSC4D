import pandas as pd  
from scipy.spatial.transform import Rotation as R
import numpy as np
import sys
import os
import time
import functools
import pickle as pkl
from multiprocessing import Pool
import open3d as o3d
import torch
import json

sys.path.append(os.path.dirname(os.path.split(os.path.abspath( __file__))[0]))
from smpl.skele2smpl import get_pose_from_bvh
from smpl.generate_ply import save_ply
import shutil

TRANSFORM_MOCAP = np.array([
    [-1, 0, 0],
    [0, 0, 1], 
    [0, 1, 0]])

def toRt(r, t):
    '''
    transfrom 3*3 rotation matrix to 4*4
    '''
    share_vector = np.array([0,0,0,1], dtype=float)[np.newaxis, :]
    r = np.concatenate((r, t.reshape(-1,1)), axis = 1)
    r = np.concatenate((r, share_vector), axis=0)
    return r

def save_smpl(count, start_idx, smpl_out_dir, smpl_models, label=''):
    ply_save_path = os.path.join(smpl_out_dir, str(int(count + start_idx)) + '_smpl' + label + '.ply')
    save_ply(smpl_models[count], ply_save_path)
    print(f'\rSave ply in {ply_save_path}', end="", flush=True)

def load_scene(dataset_root, data_name):
    scene_file = os.path.join(dataset_root, 'Scenes', data_name + '.pcd')
    print('Loading scene point cloud in: ', scene_file)
    normals_file = os.path.join(dataset_root, 'Scenes', data_name + '_normals.pkl')   
    # sub_scene_file = os.path.join(dataset_root, data_name + '_sub_scene.pcd')
    # sub_normals_file = os.path.join(filedataset_rootpath, data_name + '_sub_scene_normals.txt')   

    scene_point_cloud = o3d.io.read_point_cloud(scene_file)
    # points = np.asarray(scene_point_cloud.points)
    
    kdtree = o3d.geometry.KDTreeFlann(scene_point_cloud)
    if not os.path.exists(normals_file):
        print('Estimating normals...')
        scene_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=40))
        normals = np.asarray(scene_point_cloud.normals)
        with open(normals_file, 'wb') as f:
            pkl.dump(normals, f)
        print('Save scene normals in: ', normals_file)
    else:
        with open(normals_file, 'rb') as f:
            normals = pkl.load(f)
        scene_point_cloud.normals = o3d.utility.Vector3dVector(normals)

    print(scene_point_cloud)
    return scene_point_cloud, kdtree


def multiprocess_save_smpl_model(start_idx, smpl_models, file_path, data_name, step = 1, label=''):
    savedir = os.path.join(file_path, 'SMPL')
    smpl_out_dir = os.path.join(savedir, data_name + '_step_' + str(step) + label)
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(smpl_out_dir, exist_ok=True)
    time1 = time.time()

    with Pool(8) as p:
        p.map(functools.partial(save_smpl, start_idx = start_idx, smpl_out_dir = smpl_out_dir, smpl_models = smpl_models, label=label), np.arange(0, len(smpl_models), step).tolist())
    # pool.join()
    time2 = time.time()

    print(f'\nSMPL saved in: {smpl_out_dir}. Consumed {(time2- time1):.2f} s.')

def save_json_file(file_name, save_dict):
    """
    Saves a dictionary into json file
    Args:
        file_name:
        save_dict:
    Returns:
    """
    with open(file_name, 'w') as fp:
        json.dump(save_dict, fp, indent=4)
        
def read_json_file(file_name):
    """
    Reads a json file
    Args:
        file_name:
    Returns:
    """
    with open(file_name) as f:
        try:
            data = json.load(f)
        except:
            data = None
    return data

def data_from_json(file_path, name='betas'):
    """
    Gets the betas from the file_path
    Args:
        file_path: The name of the data file_path
    Returns:
    """
    beta_file = os.path.join(file_path, 'person.json')
    data = read_json_file(beta_file)
    return data[name]

def mocap_to_smpl_axis(mocap_init, mocap_rots):
    col_file = "/".join(os.path.abspath(__file__).split('/')[:-1]) + '/col_name.txt'
    with open (col_file, 'r') as col:
        csv_columns = col.readlines()
    new_rot_csv = pd.DataFrame(mocap_rots, columns = [col.strip() for col in csv_columns])
    mocap_smpl_rots = np.empty(shape=(0, 72))
    
    for count in range(mocap_rots.shape[0]):
        pp = get_pose_from_bvh(new_rot_csv, count, False).reshape(1,-1)
        mocap_smpl_rots = np.concatenate((mocap_smpl_rots, pp))
    '''
    mocap_smpl_rots = np.empty(shape=(mocap_rots.shape[0], 0))
    RADIAN = 57.2957795131
    for idx in mocap_init["mocap_to_smpl_order"]:
        if idx >= 0:
            mocap_euler = mocap_rots[:, idx:idx+3]
            if idx == mocap_init['LeftShoulder']: #108
                mocap_euler[:, 2] -= 0.3 * RADIAN
            elif idx == mocap_init['RightShoulder']: # 39
                mocap_euler[:, 2] += 0.3 * RADIAN
            elif idx == mocap_init['LeftArm']: # 111
                mocap_euler[:, 2] += 0.3 * RADIAN
            elif idx == mocap_init['RightArm']: # 42
                mocap_euler[:, 2] -= 0.3 * RADIAN

            mocap_euler = R.from_euler(
                'yxz', mocap_euler, degrees=True).as_rotvec()
            mocap_smpl_rots = np.concatenate((mocap_smpl_rots, mocap_euler), axis=1)
        else:
            mocap_smpl_rots = np.concatenate((mocap_smpl_rots, np.zeros((mocap_rots.shape[0], 3))), axis=1)
    '''
    return mocap_smpl_rots

    
def axis_angle_to_quaternion(axis_angle):
    """
    pytroch version
    
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [axis_angle * sin_half_angles_over_angles, torch.cos(half_angles)], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

    
def save_traj_in_dir(save_dir, data, comments):
    save_file = os.path.join(save_dir, comments + '.txt')
    field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
    np.savetxt(save_file, data, fmt=field_fmts)
    print('Save traj in: ', save_file)
    return save_file

def set_lidar_offset(init_json, synced_lidar, synced_mocap_trans):
    meta_init = read_json_file(init_json)
    if('lidar_offset' in meta_init.keys()):
        lidar_offset = np.asarray(meta_init['lidar_offset'])
    else:
        sum_lidar_offset = 0
        for i in range(10):
            # i-th rotation of lidar
            l_rot = R.from_quat(synced_lidar[i,4:8]).as_matrix()    
            
            # the lidar_offset vector from lidar to mocap at world coordinate 
            lidar_offset = synced_mocap_trans[i] - synced_lidar[i,1:4]   

            # the lidar_offset vector from lidar to mocap at lidar coordinate
            lidar_offset = np.matmul(np.linalg.inv(l_rot), lidar_offset)  

            sum_lidar_offset += lidar_offset
        lidar_offset = sum_lidar_offset / 10
        meta_init['lidar_offset'] = lidar_offset.tolist()
        save_json_file(init_json, meta_init)
    
    print('\tLidar offset (in lidar coordinate): ', np.round(lidar_offset, 3))
    return lidar_offset

def save_synced_data(synced_pos, synced_rot, synced_lidar, save_dir):
    save_file = os.path.join(save_dir, 'synced_data_for_optimization.pkl')
    synced_mocap = {
        'synced_pos': synced_pos,
        'synced_rot': synced_rot,
        'synced_lidar': synced_lidar,
        # 'smpl': smpl_models,
    }
    with open(save_file, 'wb') as sp:
        pkl.dump(synced_mocap, sp)
        print('\tsave synced data for optimization data in: ', save_file)

def compute_traj_params(mocap, lidar, init_params = None):
    mocap_traj_length = np.linalg.norm(mocap[1:] - mocap[:-1], axis=1).sum()
    lidar_traj_length = np.linalg.norm(lidar[1:] - lidar[:-1], axis=1).sum()

    mocap_XY_length = np.linalg.norm(mocap[1:, :2] - mocap[:-1, :2], axis=1).sum()
    lidar_XY_length = np.linalg.norm(lidar[1:, :2] - lidar[:-1, :2], axis=1).sum()

    mocap_Z_length = abs(mocap[1:, 2] - mocap[:-1, 2]).sum()
    lidar_Z_length = abs(lidar[1:, 2] - lidar[:-1, 2]).sum()

    print(f'Mocap traj lenght: {mocap_traj_length:.3f} m')
    print(f'LiDAR traj lenght: {lidar_traj_length:.3f} m')
    print(f'Mocap XY lenght: {mocap_XY_length:.3f} m')
    print(f'LiDAR XY lenght: {lidar_XY_length:.3f} m')
    print(f'Mocap Z lenght: {mocap_Z_length:.3f} m')
    print(f'LiDAR Z lenght: {lidar_Z_length:.3f} m')
    
    if init_params:
        init_params['lidar_traj_length'] = lidar_traj_length
        init_params['mocap_traj_length'] = mocap_traj_length
        init_params['lidar_XY_length'] = lidar_XY_length
        init_params['mocap_XY_length'] = mocap_XY_length
        init_params['lidar_Z_height'] = lidar_Z_length
        init_params['mocap_Z_heitht'] = mocap_Z_length

    return mocap_traj_length, lidar_traj_length, mocap_XY_length, lidar_XY_length, init_params

def move_file_to_dir(dst, src_file, label=None):
    if label:
        _src_file = os.path.join(dst, label)
    else:
        _src_file = os.path.join(dst, os.path.split(src_file)[1])
    shutil.move(src_file, _src_file)
    print(f'\tMove to {_src_file}')
    return _src_file

def rigid_transform_3D(A, B):
    # https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector
    # R * A + t = B
    assert A.shape == B.shape
    # =======================
    A = A.copy()
    B = B.copy()
    A[2] = 0  # Just for xy plane
    B[2] = 0
    # =======================
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
        return None, 0

    t = -R @ centroid_A + centroid_B

    return R, t