import os
from urllib.parse import quote

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def generate_launch_description():
    pkg_share = get_package_share_directory("nmb")
    default_urdf_path = os.path.join(pkg_share, "urdf", "02_SUB_RoboticArm.urdf")
    default_payload_urdf_path = os.path.join(pkg_share, "urdf", "02_SUB_RoboticArm_payload.urdf")
    default_mujoco_model_path = os.path.join(pkg_share, "urdf", "02_SUB_RoboticArm.xml")
    default_arm_controller_params_path = os.path.join(pkg_share, "config", "arm_controller.yaml")
    default_arm_automation_params_path = os.path.join(
        pkg_share, "config", "arm_automation_state_machine.yaml")
    default_trajectory_bspline_params_path = os.path.join(
        pkg_share, "config", "trajectory_bspline_jerk_planner.yaml")
    default_cartesian_ik_mapper_params_path = os.path.join(
        pkg_share, "config", "cartesian_ik_mapper.yaml")
    default_motor_comm_params_path = os.path.join(
        pkg_share, "config", "motor_comm.yaml")
    default_ros_domain_config_path = os.path.join(pkg_share, "config", "ros_domain.yaml")
    default_rviz_config_path = os.path.join(pkg_share, "rviz", "robot_arm.rviz")

    with open(default_ros_domain_config_path, "r") as infp:
        ros_domain_config = yaml.safe_load(infp) or {}
    ros_domain_id = str(ros_domain_config.get("ros", {}).get("domain_id", 0))

    urdf_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value=default_urdf_path,
        description="Absolute path to robot URDF file",
    )

    urdf_payload_arg = DeclareLaunchArgument(
        "urdf_path_payload",
        default_value=default_payload_urdf_path,
        description="Absolute path to payload URDF file",
    )

    use_mujoco_sim_arg = DeclareLaunchArgument(
        "use_mujoco_sim",
        default_value="false",
        description="Whether to run MuJoCo joint simulation node",
    )

    mujoco_model_arg = DeclareLaunchArgument(
        "mujoco_model_path",
        default_value=default_mujoco_model_path,
        description="Absolute path to MuJoCo XML model file",
    )

    mujoco_viewer_arg = DeclareLaunchArgument(
        "mujoco_viewer",
        default_value="true",
        description="Whether to open MuJoCo viewer window",
    )

    arm_controller_params_arg = DeclareLaunchArgument(
        "arm_controller_params",
        default_value=default_arm_controller_params_path,
        description="Absolute path to arm_controller_node parameter YAML file",
    )

    arm_automation_params_arg = DeclareLaunchArgument(
        "arm_automation_params",
        default_value=default_arm_automation_params_path,
        description="Absolute path to arm_automation_state_machine_node parameter YAML file",
    )

    trajectory_bspline_params_arg = DeclareLaunchArgument(
        "trajectory_bspline_params",
        default_value=default_trajectory_bspline_params_path,
        description="Absolute path to trajectory_bspline_jerk_planner_node parameter YAML file",
    )

    cartesian_ik_mapper_params_arg = DeclareLaunchArgument(
        "cartesian_ik_mapper_params",
        default_value=default_cartesian_ik_mapper_params_path,
        description="Absolute path to cartesian_ik_mapper_node parameter YAML file",
    )

    motor_comm_params_arg = DeclareLaunchArgument(
        "motor_comm_params",
        default_value=default_motor_comm_params_path,
        description="Absolute path to motor_comm_node parameter YAML file",
    )

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz_config_path,
        description="Absolute path to RViz config file",
    )

    ros_domain_env = SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id)

    trajectory_bspline_planner = Node(
        package="nmb",
        executable="trajectory_bspline_jerk_planner_node",
        name="trajectory_bspline_jerk_planner_node",
        output="screen",
        parameters=[
            LaunchConfiguration("trajectory_bspline_params"),
        ],
    )

    cartesian_ik_mapper = Node(
        package="nmb",
        executable="cartesian_ik_mapper_node",
        name="cartesian_ik_mapper_node",
        output="screen",
        parameters=[
            LaunchConfiguration("cartesian_ik_mapper_params"),
            {
                "urdf_path_empty": LaunchConfiguration("urdf_path"),
                "urdf_path_payload": LaunchConfiguration("urdf_path_payload"),
                "task_frame_empty": "ee_link",
                "task_frame_payload": "payload_center_link",
            },
        ],
    )

    arm_controller = Node(
        package="nmb",
        executable="arm_controller_node",
        name="arm_controller_node",
        output="screen",
        parameters=[
            LaunchConfiguration("arm_controller_params"),
            {
                "urdf_path_empty": LaunchConfiguration("urdf_path"),
                "urdf_path_payload": LaunchConfiguration("urdf_path_payload"),
                "task_frame_empty": "ee_link",
                "task_frame_payload": "payload_center_link",
            },
        ],
    )

    arm_automation = Node(
        package="nmb",
        executable="arm_automation_state_machine_node",
        name="arm_automation_state_machine_node",
        output="screen",
        parameters=[
            LaunchConfiguration("arm_automation_params"),
        ],
    )

    motor_comm_hw = Node(
        package="nmb",
        executable="motor_comm_node",
        name="motor_comm_node",
        output="screen",
        parameters=[
            LaunchConfiguration("motor_comm_params"),
            {
                "publish_joint_state_from_usb": True,
            },
        ],
        condition=UnlessCondition(LaunchConfiguration("use_mujoco_sim")),
    )

    mujoco_joint_sim = Node(
        package="nmb",
        executable="mujoco_joint_sim_node.py",
        name="mujoco_joint_sim_node",
        output="screen",
        prefix="/usr/bin/python3",
        parameters=[
            {
                "model_path": LaunchConfiguration("mujoco_model_path"),
                "dof": 4,
                "sim_hz": 500.0,
                "joint_state_topic": "/joint_states",
                "torque_topic": "/joint_torque_cmd",
                "payload_attached_topic": "/payload_attached",
                "payload_attached_initial": False,
                "payload_enabled": True,
                "payload_suction_cmd_topic": "/payload_suction_cmd",
                "payload_grasped_topic": "/payload_grasped",
                "publish_payload_grasped": False,
                "payload_initial_pose": [0.30, 0.00, 0.015],
                "payload_initial_yaw_rad": 0.0,
                "payload_attach_distance_threshold_m": 0.03,
                "timing_log_interval_sec": 0.0,
                "enable_viewer": LaunchConfiguration("mujoco_viewer"),
            }
        ],
        condition=IfCondition(LaunchConfiguration("use_mujoco_sim")),
    )
 
    mesh_dir = os.path.join(pkg_share, "urdf")
    mesh_uri_prefix = "file://" + quote(mesh_dir + "/", safe=":/")
    with open(default_urdf_path, 'r') as infp:
        robot_desc = infp.read().replace("package://nmb/urdf/", mesh_uri_prefix)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}],
        remappings=[
            ('robot_description', 'robot_description_internal'),
        ],
    )

    robot_description_switcher = Node(
        package='nmb',
        executable='robot_description_switcher_node',
        name='robot_description_switcher_node',
        output='screen',
        parameters=[
            {
                'urdf_path_empty': LaunchConfiguration('urdf_path'),
                'urdf_path_payload': LaunchConfiguration('urdf_path_payload'),
                'payload_attached_topic': '/payload_attached',
                'payload_attached_initial': False,
                'target_node_name': 'robot_state_publisher',
                'robot_description_topic': '/robot_description',
            }
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
    )

    return LaunchDescription([
        urdf_arg,
        urdf_payload_arg,
        use_mujoco_sim_arg,
        mujoco_model_arg,
        mujoco_viewer_arg,
        arm_controller_params_arg,
        arm_automation_params_arg,
        trajectory_bspline_params_arg,
        cartesian_ik_mapper_params_arg,
        motor_comm_params_arg,
        rviz_config_arg,
        ros_domain_env,
        trajectory_bspline_planner,
        cartesian_ik_mapper,
        motor_comm_hw,
        arm_controller,
        arm_automation,
        mujoco_joint_sim,
        robot_state_publisher_node,
        robot_description_switcher,
        rviz_node
    ])
