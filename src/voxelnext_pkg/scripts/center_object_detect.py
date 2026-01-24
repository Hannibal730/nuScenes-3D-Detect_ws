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
import sensor_msgs.point_cloud2 as pc2
from voxelnext_load import load_voxelnext_model
from builtin_interfaces.msg import Duration

# Color map for classes
color_map = {
    'car': [1, 0.5, 0.5], 'truck': [0, 1, 0], 'construction_vehicle': [0, 0, 1],
    'bus': [1, 1, 0], 'trailer': [1, 0, 1], 'barrier': [0, 1, 1],
    'motorcycle': [0.5, 0.5, 0.5], 'bicycle': [1, 0.5, 0], 'pedestrian': [0.5, 0, 0.5],
    'traffic_cone': [1, 0, 0],
}
default_color = [0, 0, 0]

nuscenes_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

def pointcloud2_to_numpy(msg):
    points = np.array(
        list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))),
        dtype=np.float32)
    timestamp = np.full((points.shape[0], 1), 0.0, dtype=np.float32)
    return np.hstack((points, timestamp))

class CenterObjectDetect(Node):
    def __init__(self):
        super().__init__('VoxelNeXt_center_object_detect')
        
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_dir = os.path.abspath(os.path.join(script_dir, '..'))
        os.chdir(project_dir)

        config_path = os.path.join(project_dir, 'tools', 'cfgs', 'nuscenes_models', 'cbgs_voxel0075_voxelnext.yaml')
        model_checkpoint = os.path.join(project_dir, 'checkpoints', 'voxelnext_nuscenes_kernel1.pth')

        self.get_logger().info(f"Config Path: {config_path}")
        self.get_logger().info(f"Model Checkpoint Path: {model_checkpoint}")

        if not os.path.exists(config_path) or not os.path.exists(model_checkpoint):
            self.get_logger().error("Configuration or model checkpoint not found.")
            sys.exit(1)

        self.voxelnext_model, self.lidar_dataset = load_voxelnext_model(config_path, model_checkpoint)
        self.voxelnext_model.eval()
        self.get_logger().info("✅ VoxelNeXt model load completed")

        self.pub_detected_objects = self.create_publisher(MarkerArray, '/detected_center', 10)
        self.pub_detected_class = self.create_publisher(MarkerArray, '/detected_class', 10)
        self.get_logger().info("✅ Publishers created")

        self.subscription = self.create_subscription(
            PointCloud2, '/velodyne_points', self.lidar_callback, 1)
        self.get_logger().info("🚀 Now everything is ready.")

    def lidar_callback(self, msg):
        self.get_logger().info("📡 Receiving LiDAR data...")
        try:
            points = pointcloud2_to_numpy(msg)
            if points.shape[1] != 5:
                self.get_logger().warn(f"❌ Incorrect point format! Expected (N,5), got: {points.shape}")
                return
            output_dicts = self.detect_objects(points)
            self.publish_markers(output_dicts)
        except Exception as e:
            self.get_logger().error(f"❌ Error during processing: {e}")

    def detect_objects(self, points):
        data_dict = {"points": points}
        data_dict = self.lidar_dataset.point_feature_encoder.forward(data_dict)

        for processor in self.lidar_dataset.dataset_cfg.DATA_PROCESSOR:
            if processor["NAME"] == "transform_points_to_voxels":
                voxels, coords, num_points_per_voxel = self.lidar_dataset.voxel_generator.generate(data_dict["points"])
                data_dict.update({"voxels": voxels, "voxel_coords": coords, "voxel_num_points": num_points_per_voxel})

        device = next(self.voxelnext_model.parameters()).device
        with torch.no_grad():
            batch_dict = {
                "batch_size": 1,
                "points": torch.from_numpy(data_dict["points"]).to(device),
                "voxels": torch.from_numpy(data_dict["voxels"]).to(device),
                "voxel_coords": torch.from_numpy(data_dict["voxel_coords"]).int().to(device),
                "voxel_num_points": torch.from_numpy(data_dict["voxel_num_points"]).to(device),
            }
            output_dicts, _ = self.voxelnext_model(batch_dict)
        return output_dicts

    def publish_markers(self, output_dicts):
        self.get_logger().info("📡 publishing /detected_center..")
        center_markers = MarkerArray()
        text_markers = MarkerArray()

        for i, output in enumerate(output_dicts):
            for j, (box, label, score) in enumerate(zip(output["pred_boxes"], output["pred_labels"], output["pred_scores"])):
                x_center, y_center, z_center, _, _, z_length, _ = box.cpu().numpy()
                class_name = self.voxelnext_model.class_names[label.cpu().item() - 1]
                score_val = score.cpu().item()
                
                self.get_logger().info(f"🔍 Object {j+1},  Class: {class_name},  Score: {score_val:.2f}")

                # Center marker
                marker = Marker()
                marker.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="velodyne")
                marker.ns, marker.id = "detected_center", i * 1000 + j
                marker.type, marker.action = Marker.SPHERE, Marker.ADD
                marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = float(x_center), float(y_center), float(z_center)
                marker.pose.orientation.w = 1.0
                marker.scale.x, marker.scale.y, marker.scale.z = 0.3, 0.3, 0.3
                marker.color.a, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
                marker.lifetime = Duration(sec=0, nanosec=200000000)
                center_markers.markers.append(marker)
                
                # Text marker
                text = Marker()
                text.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="velodyne")
                text.ns, text.id = "detected_class", i * 1000 + j
                text.type, text.action = Marker.TEXT_VIEW_FACING, Marker.ADD
                text.pose.position.x, text.pose.position.y, text.pose.position.z = float(x_center), float(y_center), float(z_center) + float(z_length) / 2 + 0.2
                text.text = f"{class_name}: {score_val:.2f}"
                text.scale.z = 0.2
                text.color.a, text.color.r, text.color.g, text.color.b = 1.0, 1.0, 1.0, 1.0
                text.lifetime = Duration(sec=0, nanosec=200000000)
                text_markers.markers.append(text)

        self.pub_detected_objects.publish(center_markers)
        self.pub_detected_class.publish(text_markers)

def main(args=None):
    rclpy.init(args=args)
    node = CenterObjectDetect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
