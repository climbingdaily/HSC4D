import numpy as np
import os
# from scipy.spatial.transform import Rotation as R
# import json

def save_traj_in_dir(save_dir, data, comments):
    save_file = os.path.join(save_dir, comments + '.txt')
    field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
    np.savetxt(save_file, data, fmt=field_fmts)
    print('Save traj in: ', save_file)

def get_overlap(lidar_file, mocap_root_file, 
                lidar_sync, mocap_sync, mocap_frame_time, save_dir=None):
    # 1. load data
    lidar = np.loadtxt(lidar_file, dtype=float)
    mocap_root = np.loadtxt(mocap_root_file, dtype=float)

    # 2. read timestamps
    lidar_time = lidar[:,-1]
    mocap_time = mocap_root[:,-1]
    start = lidar_sync - mocap_sync

    # 3. choose the corresponding frame from the mocap file according to the LiDAR timestamp
    _lidar_time = lidar_time - start
    _lidar = []
    _mocap_id = []
    _mocap_root = []
    for i, l in enumerate(lidar):
        t = _lidar_time[i]
        index = np.where(abs(mocap_time - t) - mocap_frame_time/2 <= 1e-4)[0]
        if(index.shape[0] > 0):
            _mocap_root.append(mocap_root[index[0]])
            _mocap_id.append(index[0])
            _lidar.append(l)

    # 4. save the modified mocap file
    _mocap_root = np.asarray(_mocap_root)
    _mocap_id = np.asarray(_mocap_id)
    _lidar = np.asarray(_lidar)
    if save_dir:
        save_traj_in_dir(save_dir, _mocap_root, 'mocap_trans_synced')    
        save_traj_in_dir(save_dir, _lidar, 'lidar_synced')    
    return _mocap_root, _lidar, _mocap_id

def compute_traj_params(mocap, lidar):
    mocap_traj_length = np.linalg.norm(mocap[1:] - mocap[:-1], axis=1).sum()
    lidar_traj_length = np.linalg.norm(lidar[1:] - lidar[:-1], axis=1).sum()
    print(f'Mocap traj: {mocap_traj_length:.3f}')
    print(f'LiDAR traj: {lidar_traj_length:.3f}')
    return mocap_traj_length, lidar_traj_length