from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
import sys
import os
import configargparse


def save_in_same_dir(file_path, data, ss):
    dirname = os.path.dirname(file_path)
    file_name = Path(file_path).stem
    save_file = os.path.join(dirname, file_name + ss + '.txt')
    field_fmts = ['%d', '%.6f', '%.6f', '%.6f',
                  '%.6f', '%.6f', '%.6f', '%.6f', '%.3f']
    np.savetxt(save_file, data, fmt=field_fmts)
    print('save traj in: ', save_file)
    return save_file


def filterTraj(lidar_file, frame_time=0.05, segment=20 , dist_thresh=0.03, save_type='b', time_interp = False):
    """
    The function is used to filter the trajectory of the lidar. The trajectory is filtered by the time
    interval of the trajectory. The trajectory is interpolated by the time interval of the trajectory.
    
    Args:
      lidar_file: the path to the trajectory file to be filtered
      frame_time: the time interval between two frames
      segment: the number of frames to be fitted. Defaults to 20
      dist_thresh: the distance threshold between the original trajectory and the fitted trajectory.
      save_type: . Defaults to b
      time_interp: If true, the frame id will be rearranged, otherwise, the frame id will be recorded as
    -1. Defaults to False
    """
    # 1. read data
    lidar = np.loadtxt(lidar_file, dtype=float)

    # 2. Spherical Linear Interpolation of Rotations.
    from scipy.spatial.transform import Slerp
    from scipy.spatial.transform import RotationSpline
    times = lidar[:, -1].astype(np.float64)
    time_gap = times[1:] - times[:-1]
    fit_list = np.arange(
        times.shape[0] - 1)[time_gap > frame_time * 1.5]  # Find frame's time gap > 1.5 standards frame time  

    fit_time = []
    for i, t in enumerate(times):
        fit_time.append(t)
        if i in fit_list:
            for j in range(int(time_gap[i]//frame_time)):
                if time_gap[i] - j * frame_time >= frame_time * 1.5:
                    fit_time.append(fit_time[-1] + frame_time)
    fit_time = np.asarray(fit_time)
    R_quats = R.from_quat(lidar[:, 4: 8])
    quats = R_quats.as_quat()
    spline = RotationSpline(times, R_quats)
    quats_plot = spline(fit_time).as_quat()

    trajs = lidar[:, 1:4]  # Trajectory to be fitted
    trajs_plot = []  # Trajectory after  being fitted
    for i in range(0, lidar.shape[0], segment):
        s = i-1   # start index
        e = i+segment   # end index
        if lidar.shape[0] < e:
            s = lidar.shape[0] - segment
            e = lidar.shape[0]
        if s < 0:
            s = 0

        ps = s - segment//2  # filter start index
        pe = e + segment//2  # filter end index
        if ps < 0:
            ps = 0
            pe += segment//2
        if pe > lidar.shape[0]:
            ps -= segment//2
            pe = lidar.shape[0]

        fp = np.polyfit(times[ps:pe],
                        trajs[ps:pe], 3)  
        if s == 0:
            fs = np.where(fit_time == times[0])[0][0]  
        else:
            fs = np.where(fit_time == times[i - 1])[0][0] 

        fe = np.where(fit_time == times[e-1])[0][0]  # 拟合轨迹到结束坐标

        if e == lidar.shape[0]:
            fe += 1
        for j in fit_time[fs: fe]:
            trajs_plot.append(np.polyval(fp, j))

    trajs_plot = np.asarray(trajs_plot)
    frame_id = -1 * np.ones(trajs_plot.shape[0]).reshape(-1, 1)
    valid_idx = []

    for i, t in enumerate(times):
        old_id = np.where(fit_time == t)[0][0]
        if np.linalg.norm(trajs_plot[old_id] - trajs[i]) < dist_thresh:

            # distance between fitted value and the original value < threshold, remain the original value
            trajs_plot[old_id] = trajs[i]
            quats_plot[old_id] = quats[i]

            # distance > threshold, the frame id is recorded as -1
            if save_type == 'a':
                frame_id[old_id] = lidar[i, 0]
                valid_idx.append(old_id)

        # all record the original frame id, no -1 
        if save_type == 'b':
            frame_id[old_id] = lidar[i, 0]
            valid_idx.append(old_id)


    if time_interp:
        interp_idx = np.where(frame_id == -1)[0] # Outlier ID
        frame_id[:,0] = np.arange(trajs_plot.shape[0]) # rearrange the frame id
        valid_idx = np.arange(trajs_plot.shape[0]).astype(np.int64)
    else:
        interp_idx = np.where(frame_id[valid_idx] == -1)[0] # Line ID of the outlier

    fitLidar = np.concatenate(
        (frame_id[valid_idx], trajs_plot[valid_idx], quats_plot[valid_idx], fit_time[valid_idx].reshape(-1, 1)), axis=1)

    # 4. Save trajectory
    save_file = save_in_same_dir(lidar_file, fitLidar, '_filt')  # Save valid trajectory
    np.savetxt(save_file.split('.')[0] + '_lineID.txt', interp_idx, fmt='%d')
    return fitLidar, save_file, interp_idx


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument("-F", "--traj_file", type=str, default="lidar_trajectory.txt")

    parser.add_argument("-T", "--frame_time", type=float, default=0.05)

    parser.add_argument("-D", "--dist_thresh", type=float,
                        default=0.03, help='distance threshold. If > D, a point is considered as an outlier.')
    parser.add_argument("-S", "--save_type", type=str,
                        default='a', help='a: Keep only non-outliers | b: both non-outliers and fitted value')
    parser.add_argument("--time_interp", type=bool,
                        default=True, help='time interpolation of trajectories')
    args = parser.parse_args()
    print('Filtering...')
    print(args)

    # Fit the curve with 20 points, only optimize 10 points, the sliding window is 10
    filter_window = int(1/args.frame_time)

    _, fitLidar, _ = filterTraj(args.traj_file, frame_time=args.frame_time,
                             segment=filter_window, dist_thresh=args.dist_thresh, save_type=args.save_type, time_interp=args.time_interp)
