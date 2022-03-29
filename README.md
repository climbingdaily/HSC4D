# HSC4D: Human-centered 4D Scene Capture in Large-scale Indoor-outdoor Space Using Wearable IMUs and LiDAR. CVPR 202 <br>[[Project page](https://climbingdaily.github.io/hsc4d/) | [Video](https://www.youtube.com/watch?v=IY9FikM__i8)]

<div style="align: center">
<img src = "https://github.com/climbingdaily/HSC4D/images/hsc4d_dataset.gif"/>
</div>
<div style="color:orange; border-bottom: 0px solid #d9d9d9;
display: inline-block;
color: #999;
padding: -2px;">The large indoor and outdoor scenes in our dataset. <strong>Left</strong>: a climbing gym (1200 m<sup>2</sup>). <strong>Middle</strong>: a lab building with an outside courtyard 4000 m<sup>2</sup>. <strong>Right</strong>: a loop road scene 4600 m<sup>2</sup> </div>

## Dataset
- sequence1: [campus_raod](https://drive.google.com/file/d/1fznVjBwezkJyRoTTEjxNBp7uJBaPgAJB/view?usp=sharing)
- sequence2: [climbing_gym]
- sequence3: [lab_building]
- More data is coming ...

## Data structure
```terminal
Dataset root
├── climbing_gym
|  ├── climbing_gym.bvh
|  ├── climbing_gym_pos.csv
|  ├── climbing_gym_rot.csv
|  ├── climbing_gym.pcap
|  └── climbing_gym_lidar_trajectory.txt 
| 
├── lab_building
|  ├── lab_building.bvh
|  ├── lab_building_pos.csv
|  ├── lab_building_rot.csv
|  ├── lab_building.pcap
|  └── lab_building_lidar_trajectory.txt 
|  
├── campus_road
|  ├── campus_road.bvh
|  ├── campus_road_pos.csv
|  ├── campus_road_rot.csv
|  ├── campus_road.pcap
|  └── campus_road_lidar_trajectory.txt 
└── scenes
   ├── climbing_gym.pcd
   ├── climbing_gym_ground.pcd
   ├── lab_building.pcd
   ├── lab_building_ground.pcd
   ├── campus_road.pcd
   └── campus_road_ground.pcd
```
1. The `csv` files are generated from the MoCap format data `bvh`. <br>
2. The `lidar_trajectory.txt` is provided by our own method from the LiDAR point cloud data `pcap` and coordinates aligned with corresponding scenes. <br>
3. `bvh` and `pcap` will not be used in the following processing and optimization steps.
4. By the way, you can test your SLAM algorithm with the `pcap` file. `pcap` file can be transferred into txt files with file [`initialize/ouster`](/initialize/ouster_pcap_to_txt.py)

## Preprocess
- Transfer Mocap data  (Optional, data provided)
  - **Input**: `campus_road.bvh`
  - **Command**: 
      ```python
      pip install bvhtoolbox # https://github.com/OlafHaag/bvh-toolbox
      bvh2csv campus_road.bvh
      ```
  - **Output**: `campus_road__pos.csv`, `campus_road__rot.csv`


- Point cloud mapping (Optional, data provided)
  - Write the binary pcap into txt files. Each frame is stored in a file.
    ```python
    pip install ouster-sdk 
    python ouster_pcap_to_txt.py campus_road.pcap [frame_num]
    ```
  - Run a SLAM program.
    - **Input**: The frame files in the last step.
    - **Output**: `campus_road_lidar_trajectory.txt`, `campus_road.pcd` 
  
  - Coordinate alignment (About 5 degree error after this step)
  
    1. The human stands as an A-pose before capture, and the human's face direction is regarded as scene's $Y$-axis direction. 
    2. Rotate the scene cloud $Z$-axis perpendicular to the starting position's ground. 
    3. Translate the scene to make its origin to the first SMPL model's origin on the ground. LiDAR's ego motion $T^W$ and $R^W$ are translated and rotated as the scene does. 
    
- Generate necessary files for optimization. For example, you can run the following code to preprocess the `campus_road` sequence.
  ```python
  python preprocess.py --fn campus_road
  ```

## Data fusion
***To be added***

## Data optimization
***To be added***

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
