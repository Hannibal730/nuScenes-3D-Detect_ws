![Visitor Badge](https://visitor-badge.laobi.icu/badge?page_id=Hannibal730.nuScenes-3D-Detect-Track-Predict_ws)

# Real-Time 3D Traffic Object Detection, Tracking, Future Position Prediction in ROS

This repository delivers a comprehensive solution for the [**nuScenes traffic dataset**](https://www.nuscenes.org/nuscenes) using the Velodyne VLP‑16 LiDAR sensor on ROS platform.

The system is optimized for real‑time performance. I converted [**VoxelNeXt**](https://github.com/dvlab-research/VoxelNeXt) to compute 3D spatial coordinates and confidence scores for each detected object in real time. And integrated it with [**SORT**](https://github.com/abewley/sort) tracking algorithm to achieve multi‑object tracking. Also I utilized Kalman filter to predict future position of each tracked objects for handling dynamic objects. 

I recommend this framework to students in autonomous driving competitions or studies who need end‑to‑end 3D detection, tracking, and prediction capabilities for Traffic data.
Especially those preparing for autonomous driving competitions who need to detect traffic cones using LiDAR sensors, **“This repository is strongly recommended.”**


<br>

* ## **3D Object Detection**
    <img src="https://github.com/user-attachments/assets/b3e6ead5-7de8-4ca2-a9ff-8d5cd2fac275" width="500" alt="Example Image" />


    ![Image](https://github.com/user-attachments/assets/2f300088-fca3-4832-b1d4-dda7380f6602)

<br> 

* ## **Multi‑Object Tracking**  
    ![Image](https://github.com/user-attachments/assets/171fd1f2-1c68-4c72-8ec0-8a0b285541b4)

<br>

* ## **Predicting Future-Position of tracked objects with Kalman Filter**  

    ![Image](https://github.com/user-attachments/assets/99a2b9df-4cf4-4fdb-9738-3ba760bf70b9)
    

    ![Image](https://github.com/user-attachments/assets/38ec7578-6955-4c19-8250-30364f4f99ec)

---

<br><br>


## Requirements
- This repository requires following environment.
    - Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
    - Python 3.6+
    - PyTorch 1.1 or higher (tested on PyTorch 1.1, 1,3, 1,5~1.10)
    - CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+)
    - [**spconv v1.0 (commit 8da6f96)**](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) or [**spconv v1.2**](https://github.com/traveller59/spconv) or [**spconv v2.x**](https://github.com/traveller59/spconv) <br>
    Before downloading spconv, please double-check your CUDA version to ensure compatibility with your chosen spconv version.<br><br>

- The following instructions were carried out using the setup detailed below:
    | Component         | Specification                                   |
    |-------------------|-------------------------------------------------|
    | **OS**            | Ubuntu 20.04 LTS                                |
    | **CPU**           | Intel® Core™ i7-9750H CPU @ 2.60GHz             |
    | **GPU**           | NVIDIA GTX 2060 Mobile (TU106M)                 |
    | **NVIDIA Driver** | 535                                             |
    | **CUDA**          | 11.7                                            |
    | **ROS**           | ROS Noetic                                      |
    | **PyTorch**       | 2.3.1                                           |
    | **TorchVision**   | 0.18.1                                          |
    | **TorchAudio**    | 2.3.1                                           |
    | **Lidar Sensor**  | Velodyne Lidar Vlp 16                           |



<br><br>

## Installation

1. Clone this repository
    ```shell
    git clone https://github.com/Hannibal730/nuScenes-3D-Detect-Track-Predict_ws.git
    ```

2. Install the OpenPCDet environment by following the [**OpenPCDet installation guide**](src/voxelnext_pkg/docs/INSTALL.md).

3. [**Click Here**](https://drive.google.com/file/d/1IV7e7G9X-61KXSjMGtQo579pzDNbhwvf/view?usp=share_link) to download the pre-trained weights file. Then, place the file on [**src/voxelnext_pkg/checkpoints**](src/voxelnext_pkg/checkpoints)

4. Build ROS packages
    ```shell
    cd ~/catkin_ws
    catkin_make -DCMAKE_BUILD_TYPE=Release
    ```
    
<br><br>

## How to Run

![Image](https://github.com/user-attachments/assets/972171be-7725-410f-b4cb-3d4fa6900e35)  


* **3D Object Detection**

    ```shell
    cd ~/catkin_ws
    source devel/setup.bash
    roslaunch voxelnext_pkg 3D_object_detect.launch
    ```

    ![Image](https://github.com/user-attachments/assets/c8390759-ee41-4fc2-9943-0cf7cc1b2b85)



* **Multi‑Object Tracking & Predict future position**

    ```shell
    cd ~/catkin_ws
    source devel/setup.bash
    roslaunch sort_ros_pkg tracking_and_predict_trajectory.launch
    ```

    ![Image](https://github.com/user-attachments/assets/2b821d3a-f4c1-4b49-8c73-6366fff282c1)


* **Tip**

    - If you’re on a laptop, i recommend you to make sure it’s plugged into its charger—this can dramatically reduce lag and frame drops.

    - After you run the launch/rosrun command, wait until you see the message as shown below. When you check that messages, start playing your rosbag file or powering the Velodyne LiDAR sensor.

    ![Image](https://github.com/user-attachments/assets/bdfb6806-c118-488d-8daa-7359dc128b4a)

<br><br>

## Key features in voxelnext_pkg
I focused on converting [**VoxelNeXt**](https://github.com/dvlab-research/VoxelNeXt) into a real‑time, ROS‑based package.
To achieve this, It was essential to convert the raw point cloud data generated by the Velodyne LiDAR sensor into a format compatible with the VoxelNeXt model. 
These following are the main features of the process.


### [1. live_lidar_dataset.py](https://github.com/Hannibal730/nuScenes-3D-Detect-Track-Predict_ws/blob/master/src/voxelnext_pkg/pcdet/datasets/live_lidar_dataset.py)


-  LiveLidarDataset class calculates the grid size by dividing the point cloud range by the voxel size and sets additional attributes required for the 3D detector.

- It has load_lidar_data method to convert NumPy-based LiDAR data into Torch tensors for GPU processing.



### [2. cbgs_voxel0075_voxelnext.yaml](https://github.com/Hannibal730/nuScenes-3D-Detect-Track-Predict_ws/blob/master/src/voxelnext_pkg/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml)

- This YAML specifies the POINT_CLOUD_RANGE to establish the ROI of the LiDAR data.

- It specifies crucial data processing parameters like VOXEL_SIZE, MAX_POINTS_PER_VOXEL, and MAX_NUMBER_OF_VOXELS.

- It configures the overall model architecture by defining components like NMS configuration and loss weights to optimize real-time object detection.


### [3. voxelnext_load.py](https://github.com/Hannibal730/nuScenes-3D-Detect-Track-Predict_ws/blob/master/src/voxelnext_pkg/voxelnext_load.py)

- This module loads the YAML configuration file using the cfg_from_yaml_file function.

- It creates an instance of the LiveLidarDataset class (an object that pre-processes LiDAR data and performs voxel conversion).

- It creates an instance of the VoxelNeXt model by providing the model configuration, the number of classes, and the dataset instance.

- It loads the model parameters from the checkpoint folder and moves the model to the GPU.

- It switches the model to evaluation mode and returns both the LiveLidarDataset instance and the VoxelNeXt model instance.


### [4. 3D_object_detection.py](https://github.com/Hannibal730/nuScenes-3D-Detect-Track-Predict_ws/blob/main/src/voxelnext_pkg/scripts/3D_object_detect.py)

- This script subscribes to live LiDAR data and uses the pointcloud2_to_numpy function to convert data into a format compatible with model input. Specifically the 4‑dimensional vectors (x, y, z, intensity) is extended to the 5‑dimensional NumPy tensor (x, y, z, intensity, 0.0) by adding a constant timestamp; 0.0

- It performs voxel-based 3D object detection, using the loaded VoxelNeXt model with the converted data

- The script publishes detected objects as 3D Bbounding Box marker in the name of /detected_3D_Box. And it also publishes each deteected objects' class name and score of credibailty as /detected_class


<br><br>

## Key features in sort_ros_pkg

I focused on integrating the bounding boxes published by voxelnext_pkg with the [**SORT-ros from Hyun-je**](https://github.com/Hyun-je/SORT-ros) tracking package. SORT tracking system performs tracking by comparing 2D bounding boxes frame by frame.
Thus, to feed them into SORT, I had to convert the 3D bounding boxes from voxelnext_pkg into 2D bounding boxes.
And I utilized the Kalman filter built into the SORT algorithm to predict each object’s future position.


### [1. SortRos.cpp](https://github.com/Hannibal730/nuScenes-3D-Detect-Track-Predict_ws/blob/main/src/sort_ros_pkg/src/SortRos.cpp)


- This script subscribes to 2D bounding boxes from voxelnext_pkg and converts them into SortRect objects for tracking.

- For each track, it publishes a 3D bounding box marker as /tracked_3D_Box. And publishes sphere markers meaning the tracked object’s center in the name of /tracked_center.


### [2. PredictTrajectory.cpp](https://github.com/Hannibal730/nuScenes-3D-Detect-Track-Predict_ws/blob/main/src/sort_ros_pkg/src/PredictTrajectory.cpp)


- This script initialize a state variable and sets the Kalman filter parameters, step count (iteratively applied count; frame) to compute future states.
- Kalman filter parameters: centerX, centerY, area, aspectRatio, vx, vy, area_change_rate

- It assemble a line‐strip marker connecting the current state and predicted state, and publish it as /predicted_trajectory.
  
- It creates a sphere marker at the final position (or the current one if no predictions) to denote the endpoint. And publish it as /predicted_trajectory_endpoint.

  
### [3. SharedColorMap.cpp](https://github.com/Hannibal730/nuScenes-3D-Detect-Track-Predict_ws/blob/main/src/sort_ros_pkg/src/SharedColorMap.cpp)

- These Following use the same color for each object to indicate tracking.
- /tracked_3D_Box, /tracked_center, /predicted_trajectory, /predicted_trajectory_endpoint
- To achieve this, the script assigns a unique color to each object based on its ID.



<br><br><br>

## Questions or Issues
If you have any questions or issues, feel free to ask them in the [**Issues**](https://github.com/Hannibal730/nuScenes-3D-Detect-Track-Predict_ws/issues) section.

<br><br>

## Refernce
- This project is released under the [**AGPL-3.0 license**](LICENSE). <br>
- This project is developed with reference to [**VoxelNeXt from dvlabresearch**](https://github.com/dvlab-research/VoxelNeXt) under the Apache License 2.0 <br>
- This project is developed with reference to [**SORT-ros from Hyun-je**](https://github.com/Hyun-je/SORT-ros) under the AGPL-3.0 license
- VoxelNeXt pre-trained weights file is referenced to [**VoxelNeXt from dvlabresearch**](https://github.com/dvlab-research/VoxelNeXt)
