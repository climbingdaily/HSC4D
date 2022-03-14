import numpy as np
import os
from scipy.spatial.transform import Rotation as R

from process_data.get_mocap_trans import get_mocap_root
from process_data.sync_mocap_to_lidar import get_overlap
from process_data.mocap_to_SMPL import load_csv_data

from tools.compute_foot_moving import detect_jump
from tools.tool_function import read_json_file, save_json_file, toRt, set_lidar_offset, save_synced_data, compute_traj_params, save_traj_in_dir, move_file_to_dir
from tools.filter_traj import filterTraj


MOCAP_RATE = 100
LIDAR_RATE = 20

def make_parser():
    import configargparse    
    parser = configargparse.ArgumentParser()

    parser.add_argument('--dataset_root', type=str,
                        default='/your/dataset/root', help="data folder's directory path")
    parser.add_argument("-fn", "--data_name", type=str,
                        default="campus", help="data folder's name")

    parser.add_argument("-D", "--dist_thresh", type=float,
                        default=0.04, help='distance threshold. If > D, a point is considered as an outlier.')
    parser.add_argument("-S", "--save_type", type=str,
                        default='a', help='a: Keep only non-outliers | b: both non-outliers and fitted value')
    parser.add_argument("--time_interp", type=bool,
                        default=True, help='time interpolation of trajectories')
    return parser.parse_args()


if __name__ == '__main__':

    args = make_parser()
    args.dataset_root = os.path.join(args.dataset_root, args.data_name) 
    print('File path: ', args.dataset_root)
    print('File name: ', args.data_name)

    # print('=======================================')
    print('1. Setting file path...')
    # print('=======================================')

    save_dir = os.path.join(args.dataset_root, args.data_name + '_data')
    os.makedirs(save_dir, exist_ok=True)

    mocap_trans_file = os.path.join(save_dir, 'mocap_trans.txt')
    init_json = os.path.join(save_dir, args.data_name + '_init.json')

    lidar_file = os.path.join(args.dataset_root, args.data_name + '_lidar_trajectory.txt')
    bvh_file = os.path.join(args.dataset_root, args.data_name + '.bvh')
    pospath = os.path.join(args.dataset_root, args.data_name + '_pos.csv')
    rotpath = os.path.join(args.dataset_root, args.data_name + '_rot.csv')

    if not os.path.exists(init_json):
        save_json_file(init_json, {})
    init_params = read_json_file(init_json)

    # print('=======================================')
    print('2. GET BVH ROOT POSITIONS...')
    # print('=======================================')
    print('\tBvh file: ', bvh_file)

    if os.path.exists(mocap_trans_file):
        mocap_root = np.loadtxt(mocap_trans_file, dtype=float)
    else:
        mocap_root = get_mocap_root(bvh_file, save_dir=save_dir)

    # print('=======================================')
    print('3. Detect jumps to sync')
    # print('=======================================')

    lidar_root = np.loadtxt(lidar_file, dtype=float)

    if 'mocap_sync' not in init_params.keys():
        _, _, mocap_jumps = detect_jump(
            mocap_root[:, 3], mocap_root[:, 3], width=50)
        init_params['mocap_sync'] = (np.asarray(mocap_jumps)/100).tolist()
        print('\tMocap sync time: ', init_params['mocap_sync'])

    if 'lidar_sync' not in init_params.keys():
        _, _, lidar_jumps = detect_jump(
            lidar_root[:, 3], lidar_root[:, 3], width=10)
        init_params['lidar_sync'] = lidar_root[lidar_jumps, -1].tolist()
        print('\tLidar sync time: ', init_params['lidar_sync'])

    if 'mocap_framerate' not in init_params.keys():
        init_params['mocap_framerate'] = MOCAP_RATE
    if 'lidar_framerate' not in init_params.keys():
        init_params['lidar_framerate'] = LIDAR_RATE
    if 'transfrom_lidar' not in init_params.keys():
        init_params['transfrom_lidar'] = toRt(R.from_quat(
            lidar_root[0, 4:8]).as_matrix(), lidar_root[0, 1:4]).tolist()

    # print('=======================================')
    print('4. Remove lidar outliers')
    # print('=======================================')
    _, filted_lidar_file, _ = filterTraj(lidar_file, frame_time = 1./LIDAR_RATE, segment = 15,
               dist_thresh=args.dist_thresh, save_type=args.save_type, time_interp = args.time_interp)
    _filted_lidar_file = move_file_to_dir(save_dir, filted_lidar_file, args.data_name + '_lidar_filt.txt')

    # print('=======================================')
    print('5. Sync mocap and lidar...')
    # print('=======================================')

    mocap_sync = float(init_params['mocap_sync'][0])    # first jumping in IMU
    lidar_sync = float(init_params['lidar_sync'][0])    # first jumping in LiDAR

    # Compute overlap trajs based on the first synced jump time
    synced_mocap, synced_lidar, mocap_sync_id = get_overlap(
        _filted_lidar_file, mocap_trans_file, lidar_sync, mocap_sync, 1/MOCAP_RATE)
    save_traj_in_dir(save_dir, synced_mocap, 'mocap_trans_synced')    
    save_traj_in_dir(save_dir, synced_lidar, 'lidar_filt_synced')    

    # Compute trajectroy lenght
    _,_,_,_, init_params = compute_traj_params(
        synced_mocap[:, 1:4], synced_lidar[:, 1:4], init_params)

    # print('=======================================')
    print('6. Calculate lidar offset ')
    # print('=======================================')

    lidar_offset = set_lidar_offset(
        init_json, synced_lidar, synced_mocap[:, 1:4])
    trans_with_offset = np.matmul(R.from_quat(
        synced_lidar[:, 4:8]).as_matrix(), lidar_offset)
    synced_lidar[:, 1:4] += trans_with_offset
    save_traj_file = save_traj_in_dir(save_dir, synced_lidar, args.data_name + '_lidar_filt_synced_offset')

    # print('=======================================')
    print('7. Saving data')
    # print('=======================================')

    _, pos_data, rot_data = load_csv_data(lidar_file, rotpath, pospath)
    save_synced_data(pos_data[mocap_sync_id, 0],
                     rot_data[mocap_sync_id], synced_lidar, save_dir)

    init_params['lidar_offset_traj_length'] = np.linalg.norm(
        synced_lidar[1:, 1:4] - synced_lidar[:-1, 1:4], axis=1).sum()

    init_params['lidar_offset'] = lidar_offset.tolist()
    save_json_file(init_json, init_params)

    print('=======================Data prepared!!!==========================')
