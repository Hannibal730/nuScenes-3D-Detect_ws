import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

from launch.actions import ExecuteProcess


def generate_launch_description():

    pkg_share = get_package_share_directory('voxelnext_pkg')
    rviz_config_file = os.path.join(pkg_share, 'launch/object_detect.rviz')

    return LaunchDescription([
        Node(
            package='voxelnext_pkg',
            executable='3D_object_detect.py',
            name='VoxelNeXt_3D_object_detect',
            output='screen',
        ),
        ExecuteProcess(
            cmd=['rviz2', '-d', rviz_config_file],
            output='screen',
            additional_env={'LIBGL_ALWAYS_SOFTWARE': '1'},
        ),
    ])
