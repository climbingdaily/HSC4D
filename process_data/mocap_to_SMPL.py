import pandas as pd  
import numpy as np

def load_csv_data(lidar_file, rotpath, pospath):
    lidar_data = np.loadtxt(lidar_file, dtype=float)
    pos_data_csv=pd.read_csv(pospath, dtype=np.float32)
    rot_data_csv=pd.read_csv(rotpath, dtype=np.float32)

    pos_data = np.asarray(pos_data_csv) /100 # cm -> m
    mocap_length = pos_data.shape[0]
    pos_data = pos_data[:, 1:].reshape(mocap_length, -1, 3)
    rot_data = np.asarray(rot_data_csv) # degree
    return lidar_data, pos_data, rot_data