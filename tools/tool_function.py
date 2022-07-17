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
import shutil

from . import get_pose_from_bvh, save_ply


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
    """
    It saves the trajectory data in a file in the specified directory
    
    Args:
      save_dir: the directory where you want to save the trajectory
      data: the data to be saved
      comments: a string that will be used to name the file
    
    Returns:
      The save_file is being returned.
    """
    save_file = os.path.join(save_dir, comments + '.txt')
    field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
    np.savetxt(save_file, data, fmt=field_fmts)
    print('Save traj in: ', save_file)
    return save_file

def set_lidar_offset(init_json, synced_lidar, synced_mocap_trans):
    """
    > The function `set_lidar_offset` takes in the initial json file, the synced lidar data, and the
    synced mocap data, and returns the lidar offset vector in the lidar coordinate
    
    Args:
      init_json: the path to the initialization json file
      synced_lidar: the lidar data that has been synced with the mocap data
      synced_mocap_trans: the translation of the mocap at each time step
    
    Returns:
      The lidar offset in lidar coordinate
    """
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
    """
    It takes in the synced position, rotation, and lidar data, and saves it to a pickle file
    
    Args:
      synced_pos: a list of numpy arrays, each of shape (num_frames, 3)
      synced_rot: a list of numpy arrays, each of shape (num_frames, 3, 3)
      synced_lidar: a list of numpy arrays, each array is a point cloud of shape (N, 3)
      save_dir: the directory where you want to save the data
    """
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
    """
    It computes the trajectory lengths for both the LiDAR and the mocap data
    
    Args:
      mocap: the trajectory of the mocap system
      lidar: the LiDAR trajectory
      init_params: dictionary of initial parameters for the optimization
    """
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
    """
    > Move a file to a directory, and return the new path
    
    Args:
      dst: the destination directory
      src_file: the file to be moved
      label: the label of the file, if you want to rename it.
    
    Returns:
      The new file path.
    """
    if label:
        _src_file = os.path.join(dst, label)
    else:
        _src_file = os.path.join(dst, os.path.split(src_file)[1])
    shutil.move(src_file, _src_file)
    print(f'\tMove to {_src_file}')
    return _src_file