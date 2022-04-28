# HSC4D: Human-centered 4D Scene Capture in Large-scale Indoor-outdoor Space Using Wearable IMUs and LiDAR. CVPR 2022
[[Project page](http://www.lidarhumanmotion.net/hsc4d/) | [Video](https://www.youtube.com/watch?v=IY9FikM__i8)]

<!-- <div align=center>
<img src = "https://github.com/climbingdaily/HSC4D/blob/main/images/logo.png" width=85%/> </div>
<br> -->
<div align=center>
<img src = "https://climbingdaily.github.io/images/overview.png"/></div>



# Getting start
## Dataset (Click [here](https://drive.google.com/drive/folders/1c6iGtqcAhPmzSsoep-WB-g_kJQjMZl-t?usp=sharing) to download)
<div align=center>
<img src = "https://github.com/climbingdaily/HSC4D/blob/main/images/hsc4d_dataset.gif"/>
</div>
<div style="color:orange; border-bottom: 0px solid #d9d9d9;
display: inline-block;
color: #999;
padding: -2px;">The large indoor and outdoor scenes in our dataset. <strong>Left</strong>: a climbing gym (1200 m<sup>2</sup>). <strong>Middle</strong>: a lab building with an outside courtyard 4000 m<sup>2</sup>. <strong>Right</strong>: a loop road scene 4600 m<sup>2</sup> </div>

### Data structure
```terminal
Dataset root/
├── [Place_holder]/
|  ├── [Place_holder].bvh     # MoCap data from Noitom Axis Studio (PNStudio)
|  ├── [Place_holder]_pos.csv # Every joint's roration, generated from `*_bvh`
|  ├── [Place_holder]_rot.csv # Every joint's translation, generated from `*_bvh`
|  ├── [Place_holder].pcap    # Raw data from the LiDAR
|  └── [Place_holder]_lidar_trajectory.txt  # N×9 format file
├── ...
|
└── scenes/
   ├── [Place_holder].pcd
   ├── [Place_holder]_ground.pcd
   ├── ...
   └── ...
```
  1. Place_holder can be replaced to `campus_raod`, `climbing_gym`, and `lab_building`.
  2. `*_lidar_trajectory.txt` is generated by our Mapping method and manually calibrated with corresponding scenes. <br>
  3. `*_bvh` and `*_pcap` are raw data from sensors. They will not be used in the following steps.
  4. You can test your SLAM algorithm by using `*_pcap` captured from Ouster1-64 with 1024×20Hz. 

### Preparation
- Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` and put it in `smpl` directory.
- Downloat the dataset and modify `dataset_root` and `data_name` in `configs/sample.cfg`.
``` bash
dataset_root = /your/path/to/datasets
data_name = campus_road # or lab_building, climbing_gym
```

## Requirement
  Our code is tested under:
  - Ubuntu: 18.04
  - Python: 3.8
  - CUDA:   11.0
  - Pytorch: 1.7.0

## Installation
  ``` python
  conda create -n hsc4d python=3.8
  conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
  pip install open3d chumpy scipy configargparse matplotlib pathlib pandas opencv-python torchgeometry tensorboardx
  ```
  - ***Note***: For mask conversion compatibility in PyTorch 1.7.0, you need to manually edit the source file in torchgeometry. Follow the [guide here](https://stackoverflow.com/questions/65637222/runtimeerror-subtraction-the-operator-with-a-bool-tensor-is-not-supported)
  ```bash
    $ vi /home/dyd/software/anaconda3/envs/hsc4d/lib/python3.8/site-packages/torchgeometry/core/conversions.py

    # mask_c1 = mask_d2 * (1 - mask_d0_d1)
    # mask_c2 = (1 - mask_d2) * mask_d0_nd1
    # mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
    mask_c1 = mask_d2 * ~(mask_d0_d1)
    mask_c2 = ~(mask_d2) * mask_d0_nd1
    mask_c3 = ~(mask_d2) * ~(mask_d0_nd1)
  ```
  - ***Note***: When nvcc fatal error occurs.
  ``` bash
  export TORCH_CUDA_ARCH_LIST="8.0" #nvcc complier error. nvcc fatal: Unsupported gpu architecture 
  ```
## Preprocess
- ### Transfer Mocap data [Optional, data provided]
    ```bash
    pip install bvhtoolbox # https://github.com/OlafHaag/bvh-toolbox
    bvh2csv /your/path/to/campus_road.bvh
    ```
  - **Output**: `campus_road_pos.csv`, `campus_road_rot.csv`


- ### LiDAR mapping [Optional, data provided]
  - Process pcap file
    ```bash
    cd initialize
    pip install ouster-sdk 
    python ouster_pcap_to_txt.py -P /your/path/to/campus_road.pcap [-S start_frame] [-E end_frame]
    ```
  - Output: campus_road__lidar_frames/[time_stamp].txt
  - Run your Mapping/SLAM algorithm.
  
  - Coordinate alignment (About 5 degree error after this step)
  
    1. The human stands as an A-pose before capture, and the human's face direction is regarded as scene's $Y$-axis direction. 
    2. Rotate the scene cloud to make its $Z$-axis perpendicular to the starting position's ground. 
    3. Translate the scene to make its origin to the first SMPL model's origin on the ground. 
    4. LiDAR's ego motion $T^W$ and $R^W$ are translated and rotated as the scene does. 
  - Output: `campus_road_lidar_trajectory.txt`, `scenes/campus_road.pcd`
    
- ### Data preprocessing for optimization. 
  ```bash
  python preprocess.py --dataset_root /your/path/to/datasets -fn campus_road -D 0.1
  ```

## Data fusion
***To be added***

## Data optimization
```bash
python main.py --config configs/sample.cfg
```


## Visualization
***To be added***




## Copyright
The HSC4D dataset is published under the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/).You must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. Contact us if you are interested in commercial usage.



## Bibtex
```
@misc{dai2022hsc4d,
    title={HSC4D: Human-centered 4D Scene Capture in Large-scale Indoor-outdoor Space Using Wearable IMUs and LiDAR},
    author={Yudi Dai and Yitai Lin and Chenglu Wen and Siqi Shen and Lan Xu and Jingyi Yu and Yuexin Ma and Cheng Wang},
    year={2022},
    eprint={2203.09215},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
