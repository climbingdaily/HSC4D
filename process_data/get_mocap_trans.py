import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def loadjoint(frames, joint_number, frame_time=0.0333333):
    joint = frames[:, joint_number * 6:joint_number*6+3] / 100
    joint_rot = frames[:, joint_number * 6+3:joint_number*6+6]
    rot = R.from_euler('yxz', joint_rot, degrees=True).as_quat()
    frame_number = np.arange(frames.shape[0])
    frame_number = frame_number.reshape((-1, 1))
    frame_time = frame_number * frame_time

    '''
    from mocap coordinate to Velodyne coordinate, make the Z-axis point up
    '''
    rz = R.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix()
    rx = R.from_rotvec(270 * np.array([1, 0, 0]), degrees=True).as_matrix()
    init_rot = np.matmul(rx, rz) #Rotate 180° around Z，and then rotate 270° around X
    joint = np.matmul(init_rot, joint.T).T

    save_joint = np.concatenate((frame_number, joint, rot, frame_time), axis=-1)
    return save_joint

from ._import_ import bvh_tool
def get_mocap_root(bvh_file, save_dir = None):
    with open(bvh_file) as f:
        mocap = bvh_tool.Bvh(f.read())

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

if __name__ == '__main__':
    from _import_ import config_parser
    parser = config_parser()
    print('GET BVH ROOT POSITIONS......')
    parser.add_argument("-B", "--bvh_file", type=str, default=None)
    args = parser.parse_args()


    if args.bvh_file:
        bvh_file = args.bvh_file
    else:
        bvh_file = os.path.join(args.dataset_root, args.data_name + '.bvh')

    save_dir = os.path.join(os.path.dirname(bvh_file), bvh_file.split('/')[-1].split('.')[0] + '_data')
    os.makedirs(save_dir, exist_ok=True)
    print('Bvh file: ', bvh_file)

    root = get_mocap_root(bvh_file, save_dir=save_dir)