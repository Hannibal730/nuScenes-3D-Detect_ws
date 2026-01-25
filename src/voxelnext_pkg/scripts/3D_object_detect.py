#!/usr/bin/env python3

import sys
import os
import rclpy
from rclpy.node import Node
import torch
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2 as pc2

script_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(script_path))
sys.path.insert(0, os.path.dirname(os.path.dirname(script_path)))

from model import load_voxelnext
import math

from builtin_interfaces.msg import Duration

# -------------------------
# Define colors for each class
# -------------------------
color_map = {
    'car': [1, 0.5, 0.5], # Light Red
    'truck': [0, 1, 0], # Green
    'construction_vehicle': [0, 0, 1], # Blue
    'bus': [1, 1, 0], # Yellow
    'trailer': [1, 0, 1], # Magenta
    'barrier': [0, 1, 1], # Cyan
    'motorcycle': [0.5, 0.5, 0.5], # Gray
    'bicycle': [1, 0.5, 0], # Orange
    'pedestrian': [0.5, 0, 0.5], # Purple
    'traffic_cone': [1, 0, 0],  # 빨강
}
default_color = [0, 0, 0]   # Default: Black


# -------------------------
# Define class number for each class
# -------------------------
nuscenes_class_names = [
    'car',                   # 1
    'truck',                 # 2
    'construction_vehicle',  # 3
    'bus',                   # 4
    'trailer',               # 5
    'barrier',               # 6
    'motorcycle',            # 7
    'bicycle',               # 8
    'pedestrian',            # 9
    'traffic_cone'           # 10
]

# -------------------------
# Function: Convert PointCloud2 message to a NumPy array in (N, 5) format
# -------------------------
def pointcloud2_to_numpy(msg):
    """
    Convert a ROS PointCloud2 message to a NumPy array with shape (N, 5).
    - Format: [x, y, z, intensity, timestamp]
    - Extracts only points from the region of interest (ROI).
    """
    points = np.array(
        list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))),
        dtype=np.float32)
    # Add timestamp column (fixed at 0.0 in this case)
    timestamp = np.full((points.shape[0], 1), 0.0, dtype=np.float32)
    points_with_timestamp = np.hstack((points, timestamp))
    
    return points_with_timestamp
class VoxelNeXt3DDetect(Node):
    def __init__(self):
        super().__init__('VoxelNeXt_3D_object_detect')
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Define the project root directory (assumed to be one level up from the current script)
        project_dir = os.path.abspath(os.path.join(script_dir, '..'))

        # Change the working directory to the project root (important for resolving relative paths)
        os.chdir(project_dir)

        # Define absolute paths for the configuration file and the model checkpoint
        config_path = os.path.join(project_dir, 'tools', 'cfgs', 'nuscenes_models', 'cbgs_voxel0075_voxelnext.yaml')
        model_checkpoint = os.path.join(project_dir, 'checkpoints', 'voxelnext_nuscenes_kernel1.pth')

        self.get_logger().info(f"Config Path: {config_path}")
        self.get_logger().info(f"Model Checkpoint Path: {model_checkpoint}")
        self.get_logger().info(f"Config file exists: {os.path.exists(config_path)}")
        self.get_logger().info(f"Model checkpoint exists: {os.path.exists(model_checkpoint)}")

        # Exit if configuration or model checkpoint is not found
        if not os.path.exists(config_path):
            self.get_logger().error(f"Config file not found: {config_path}")
            sys.exit(1)
        if not os.path.exists(model_checkpoint):
            self.get_logger().error(f"Model checkpoint not found: {model_checkpoint}")
            sys.exit(1)

        # Load the VoxelNeXt model and associated lidar dataset
        self.voxelnext_model, self.lidar_dataset = load_voxelnext(config_path, model_checkpoint)
        self.voxelnext_model.eval() # Set the model to evaluation mode
        self.get_logger().info("✅ VoxelNeXt model load completed")

        # Create a ROS publisher for detected objects (bounding box markers)
        self.pub_detected_objects = self.create_publisher(MarkerArray, '/detected_3D_Box', 10)
        self.get_logger().info("✅ Publishers for /detected_3D_Box created")

        self.pub_detected_class   = self.create_publisher(MarkerArray, '/detected_class', 10)
        self.get_logger().info("✅ Publishers for /detected_class created")

        # Create a ROS subscriber to receive PointCloud2 messages from the LiDAR sensor
        self.subscription = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.lidar_callback,
            1) # QoS profile 1 for compatibility with buff_size
        self.get_logger().info("✅ Subscriber for '/velodyne_points' created")
        self.get_logger().info("🚀 Now everything is ready. Run the rosbag file or launch the Velodyne LiDAR")

    def lidar_callback(self, msg):
        self.get_logger().info("📡 Receiving LiDAR data...")

        try:
            points = pointcloud2_to_numpy(msg)
        except Exception as e:
            self.get_logger().error(f"❌ Error converting PointCloud2: {e}")
            return

        if points.shape[1] != 5:
            self.get_logger().warn(f"❌ Incorrect point format! Expected (N,5), got: {points.shape}")
            return

        try:
            output_dicts = self.detect_objects(points, self.voxelnext_model, self.lidar_dataset)
            self.publish_markers(output_dicts, self.pub_detected_objects, self.pub_detected_class, self.voxelnext_model.class_names)
        except Exception as e:
            self.get_logger().error(f"❌ Error during object detection/publishing: {e}")

    def detect_objects(self, points, voxelnext_model, lidar_dataset):
        self.get_logger().info("📡 Converting LiDAR data...")
        data_dict = {"points": points}

        # Perform point feature encoding
        data_dict = lidar_dataset.point_feature_encoder.forward(data_dict)

        # Process data using each processor in the dataset configuration
        for processor in lidar_dataset.dataset_cfg.DATA_PROCESSOR:
            if processor["NAME"] == "transform_points_to_voxels":
                voxels, coords, num_points_per_voxel = lidar_dataset.voxel_generator.generate(data_dict["points"])
                data_dict["voxels"] = voxels
                data_dict["voxel_coords"] = coords
                data_dict["voxel_num_points"] = num_points_per_voxel

        # Prepare tensors for model inference
        device = next(voxelnext_model.parameters()).device
        voxel_coords_tensor = torch.from_numpy(data_dict["voxel_coords"]).int().to(device)

        with torch.no_grad():
            batch_dict = {
                "batch_size": 1,
                "points": torch.from_numpy(data_dict["points"]).to(device),
                "voxels": torch.from_numpy(data_dict["voxels"]).to(device),
                "voxel_coords": voxel_coords_tensor,
                "voxel_num_points": torch.from_numpy(data_dict["voxel_num_points"]).to(device),
            }
            self.get_logger().debug("📦 Batch Dict ready")
            output_dicts, _ = voxelnext_model(batch_dict)
        return output_dicts

    def publish_markers(self, output_dicts, pub_detected_objects, pub_detected_class, class_names):
        self.get_logger().info("📡 publishing /detected_3D_Box..")

        box_markers  = MarkerArray()
        text_markers = MarkerArray()

        for i, output in enumerate(output_dicts):
            for j, (box, label, score) in enumerate(zip(output["pred_boxes"], output["pred_labels"], output["pred_scores"])):
                label = label.cpu().item()
                score = score.cpu().item()

                # Extract x, y and z box's informations
                x_center = box[0].cpu().item()
                y_center = box[1].cpu().item()
                z_center = box[2].cpu().item()
                x_length = box[3].cpu().item()
                y_length = box[4].cpu().item()
                z_length = box[5].cpu().item()
                heading = box[6].cpu().item()

                qz = math.sin(heading / 2.0)
                qw = math.cos(heading / 2.0)

                class_name = class_names[label - 1] # Adjust the class label index (assuming class indices start from 1)
                color = color_map.get(class_name, default_color)
                self.get_logger().info(f"🔍 Object {j+1},  Class: {class_name},  Score: {score:.2f},  Position: ({x_center:.2f}, {y_center:.2f}, {z_center:.2f})")


                # 1. Create Box marker 
                marker = Marker()
                marker.header = Header()
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.header.frame_id = "velodyne"
                marker.ns = "detected_3D_Box"
                marker.id = i * 1000 + j
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                marker.pose.position.x = x_center
                marker.pose.position.y = y_center
                marker.pose.position.z = z_center
                marker.scale.x = x_length
                marker.scale.y = y_length
                marker.scale.z = z_length

                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = qz
                marker.pose.orientation.w = qw

                marker.color.a = 0.5
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.lifetime = Duration(sec=0, nanosec=200000000)
                box_markers.markers.append(marker)


                # 2. # Create txt marker 
                text = Marker()
                text.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="velodyne")
                text.ns     = "detected_class"
                text.id     = i * 1000 + j
                text.type   = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose.position.x = x_center
                text.pose.position.y = y_center
                text.pose.position.z = z_center + z_length / 2 + 0.2
                text.text    = f"{class_name}: {score:.2f}"
                text.scale.z = 0.2
                text.color.r = 1.0
                text.color.g = 1.0
                text.color.b = 1.0
                text.color.a = 1.0
                text.lifetime = Duration(sec=0, nanosec=200000000)
                text_markers.markers.append(text)

        # publish two topics
        pub_detected_objects.publish(box_markers)
        pub_detected_class.publish(text_markers)
def main(args=None):
    rclpy.init(args=args)
    voxelnext_3d_detect_node = VoxelNeXt3DDetect()
    try:
        rclpy.spin(voxelnext_3d_detect_node)
    except KeyboardInterrupt:
        pass
    finally:
        voxelnext_3d_detect_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
