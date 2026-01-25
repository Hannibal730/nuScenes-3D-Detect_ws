#!/usr/bin/env python3

import sys
import os
import rclpy
from rclpy.node import Node
import torch
import numpy as np
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2 as pc2
from model import load_voxelnext
import math
from builtin_interfaces.msg import Duration

# Color map and class names
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

class ObjectDetect(Node):
    def __init__(self):
        super().__init__('VoxelNeXt_object_detect')
        
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

        self.voxelnext_model, self.lidar_dataset = load_voxelnext(config_path, model_checkpoint)
        self.voxelnext_model.eval()
        self.get_logger().info("✅ VoxelNeXt model load completed")

        self.pub_2d = self.create_publisher(MarkerArray, '/detected_2D_Box', 10)
        self.pub_3d = self.create_publisher(MarkerArray, '/detected_3D_Box', 10)
        self.pub_center = self.create_publisher(MarkerArray, '/detected_center', 10)
        self.pub_class = self.create_publisher(MarkerArray, '/detected_class', 10)
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
            batch_dict = {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v for k, v in data_dict.items() if k in ['points', 'voxels', 'voxel_num_points']}
            batch_dict["batch_size"] = 1
            batch_dict["voxel_coords"] = torch.from_numpy(data_dict["voxel_coords"]).int().to(device)
            output_dicts, _ = self.voxelnext_model(batch_dict)
        return output_dicts

    def publish_markers(self, output_dicts):
        markers_2d, markers_3d, markers_center, markers_text = MarkerArray(), MarkerArray(), MarkerArray(), MarkerArray()
        
        for i, output in enumerate(output_dicts):
            for j, (box, label, score) in enumerate(zip(output["pred_boxes"], output["pred_labels"], output["pred_scores"])):
                b = box.cpu().numpy()
                class_name = self.voxelnext_model.class_names[label.cpu().item() - 1]
                score_val = score.cpu().item()
                color = color_map.get(class_name, default_color)
                
                self.get_logger().info(f"🔍 Object {j+1}, Class: {class_name}, Score: {score_val:.2f}")

                stamp = self.get_clock().now().to_msg()
                
                # Common properties
                x_c, y_c, z_c, x_l, y_l, z_l, heading = b
                qz = math.sin(heading / 2.0)
                qw = math.cos(heading / 2.0)

                # 2D Box
                m2 = Marker(header=Header(stamp=stamp, frame_id="velodyne"), ns="detected_2D_Box", id=i*1000+j, type=Marker.CUBE, action=Marker.ADD)
                m2.pose.position.x, m2.pose.position.y, m2.pose.position.z = float(x_c), float(y_c), 0.0
                m2.scale.x, m2.scale.y, m2.scale.z = float(x_l), float(y_l), 0.01 # thin box
                m2.pose.orientation.z, m2.pose.orientation.w = float(qz), float(qw)
                m2.color.a, m2.color.r, m2.color.g, m2.color.b = 0.5, float(color[0]), float(color[1]), float(color[2])
                m2.lifetime = Duration(sec=0, nanosec=200000000)
                markers_2d.markers.append(m2)

                # 3D Box
                m3 = Marker(header=Header(stamp=stamp, frame_id="velodyne"), ns="detected_3D_Box", id=i*1000+j, type=Marker.CUBE, action=Marker.ADD)
                m3.pose.position.x, m3.pose.position.y, m3.pose.position.z = float(x_c), float(y_c), float(z_c)
                m3.scale.x, m3.scale.y, m3.scale.z = float(x_l), float(y_l), float(z_l)
                m3.pose.orientation.z, m3.pose.orientation.w = float(qz), float(qw)
                m3.color.a, m3.color.r, m3.color.g, m3.color.b = 0.5, float(color[0]), float(color[1]), float(color[2])
                m3.lifetime = Duration(sec=0, nanosec=200000000)
                markers_3d.markers.append(m3)

                # Center
                mc = Marker(header=Header(stamp=stamp, frame_id="velodyne"), ns="detected_center", id=i*1000+j, type=Marker.SPHERE, action=Marker.ADD)
                mc.pose.position.x, mc.pose.position.y, mc.pose.position.z = float(x_c), float(y_c), float(z_c)
                mc.scale.x, mc.scale.y, mc.scale.z = 0.3, 0.3, 0.3
                mc.pose.orientation.w = 1.0
                mc.color.a, mc.color.g = 1.0, 1.0
                mc.lifetime = Duration(sec=0, nanosec=200000000)
                markers_center.markers.append(mc)

                # Text
                mt = Marker(header=Header(stamp=stamp, frame_id="velodyne"), ns="detected_class", id=i*1000+j, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
                mt.pose.position.x, mt.pose.position.y, mt.pose.position.z = float(x_c), float(y_c), float(z_c) + float(z_l)/2 + 0.2
                mt.text = f"{class_name}: {score_val:.2f}"
                mt.scale.z = 0.4
                mt.color.a, mt.color.r, mt.color.g, mt.color.b = 1.0, 1.0, 1.0, 1.0
                mt.lifetime = Duration(sec=0, nanosec=200000000)
                markers_text.markers.append(mt)

        self.pub_2d.publish(markers_2d)
        self.pub_3d.publish(markers_3d)
        self.pub_center.publish(markers_center)
        self.pub_class.publish(markers_text)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()