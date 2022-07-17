import pandas as pd  
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import configargparse
from .bvh_tool import Bvh

def load_csv_data(rotpath, pospath):
    """
    It loads the csv files and converts the data into the right format
    
    Args:
      rotpath: path to the csv file containing the rotation data
      pospath: the path to the csv file containing the position data
    
    Returns:
      The position and rotation data of the mocap data.
    """
    pos_data_csv=pd.read_csv(pospath, dtype=np.float32)
    rot_data_csv=pd.read_csv(rotpath, dtype=np.float32)

    pos_data = np.asarray(pos_data_csv) /100 # cm -> m
    mocap_length = pos_data.shape[0]
    pos_data = pos_data[:, 1:].reshape(mocap_length, -1, 3)
    rot_data = np.asarray(rot_data_csv) # degree
    return pos_data, rot_data


def loadjoint(frames, joint_number, frame_time=0.0333333):
    """
    It takes in a list of frames, a joint number, and a frame time, and returns a list of frames with
    the joint number's position and rotation
    
    Args:
      frames: the data of the joint
      joint_number: the number of the joint you want to extract.
      frame_time: the time between two frames, default is 0.0333333
    
    Returns:
      the joint data in the form of a numpy array.
    """
    joint = frames[:, joint_number * 6:joint_number*6+3] / 100
    joint_rot = frames[:, joint_number * 6+3:joint_number*6+6]
    rot = R.from_euler('yxz', joint_rot, degrees=True).as_quat()
    frame_number = np.arange(frames.shape[0])
    frame_number = frame_number.reshape((-1, 1))
    frame_time = frame_number * frame_time

    # from mocap coordinate to LiDAr coordinate, make the Z-axis point up
    rz = R.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix()
    rx = R.from_rotvec(270 * np.array([1, 0, 0]), degrees=True).as_matrix()
    init_rot = np.matmul(rx, rz) #Rotate 180° around Z，and then rotate 270° around X
    joint = np.matmul(init_rot, joint.T).T

    save_joint = np.concatenate((frame_number, joint, rot, frame_time), axis=-1)
    return save_joint

def get_mocap_root(bvh_file, save_dir = None):
    """
    It reads a bvh file and returns the root translation of each frame
    
    Args:
      bvh_file: the path to the bvh file
      save_dir: the directory where you want to save the mocap data.
    
    Returns:
      The root is being returned.
    """
    with open(bvh_file) as f:
        mocap = Bvh(f.read())

    # load data
    frame_time = mocap.frame_time
    frames = mocap.frames
    frames = np.asarray(frames, dtype='float32')

    root = loadjoint(frames, 0, frame_time)  # unit：meter

    # save data
    if save_dir:
        save_file = os.path.join(save_dir, 'mocap_trans.txt')
        field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
        np.savetxt(save_file, root, fmt=field_fmts)
        print('Save mocap translation in: ', save_file)
    return root

def save_traj_in_dir(save_dir, data, comments):
    save_file = os.path.join(save_dir, comments + '.txt')
    field_fmts = ['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
    np.savetxt(save_file, data, fmt=field_fmts)
    print('Save traj in: ', save_file)

def get_overlap(lidar_file, mocap_root_file, 
                lidar_sync, mocap_sync, mocap_frame_time, save_dir=None):
    """
    > Given the LiDAR timestamp, find the corresponding frame from the mocap file
    
    Args:
      lidar_file: the path to the LiDAR file
      mocap_root_file: the file containing the mocap data
      lidar_sync: the timestamp of the first LiDAR point
      mocap_sync: the timestamp of the first frame in the mocap file
      mocap_frame_time: the time between two consecutive frames in the mocap file
      save_dir: the directory to save the modified mocap file
    """
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


if __name__ == '__main__':
    parser = configargparse()
    print('GET BVH ROOT POSITIONS......')
    parser.add_argument("-B", "--bvh_file", type=str, default=None)
    args = parser.parse_args()
    bvh_file = args.bvh_file
    save_dir = os.path.join(os.path.dirname(bvh_file), os.path.basename(bvh_file).split('.')[0] + '_data')
    os.makedirs(save_dir, exist_ok=True)
    print('Bvh file: ', bvh_file)

    root = get_mocap_root(bvh_file, save_dir=save_dir)