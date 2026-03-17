#!/usr/bin/env python3
import os
import time
from typing import List

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray

try:
    import mujoco
except ImportError as exc:
    raise RuntimeError(
        "MuJoCo Python package not found. Please install with: pip install mujoco"
    ) from exc

try:
    import mujoco.viewer as mj_viewer
except ImportError:
    mj_viewer = None


class MujocoJointSimNode(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_joint_sim_node")

        # Core IO parameters.
        self.model_path = str(self.declare_parameter("model_path", "").value)
        self.dof = int(self.declare_parameter("dof", 4).value)
        self.joint_state_topic = str(self.declare_parameter("joint_state_topic", "/joint_states").value)
        self.torque_topic = str(self.declare_parameter("torque_topic", "/joint_torque_cmd").value)
        self.payload_attached_topic = str(self.declare_parameter("payload_attached_topic", "/payload_attached").value)
        self.payload_attached = bool(self.declare_parameter("payload_attached_initial", False).value)
        self.payload_enabled = bool(self.declare_parameter("payload_enabled", True).value)
        self.payload_suction_cmd_topic = str(
            self.declare_parameter("payload_suction_cmd_topic", "/payload_suction_cmd").value)
        self.payload_grasped_topic = str(
            self.declare_parameter("payload_grasped_topic", "/payload_grasped").value)
        self.publish_payload_grasped = bool(
            self.declare_parameter("publish_payload_grasped", True).value)
        self.enable_viewer = bool(self.declare_parameter("enable_viewer", True).value)

        # Payload grasp/placement parameters.
        payload_initial_pose = self.declare_parameter(
            "payload_initial_pose", [0.30, 0.00, 0.015]).value
        self.payload_initial_pose = np.array(payload_initial_pose[:3], dtype=np.float64)
        self.payload_initial_yaw_rad = float(
            self.declare_parameter("payload_initial_yaw_rad", 0.0).value)
        self.payload_attach_distance_threshold_m = float(
            self.declare_parameter("payload_attach_distance_threshold_m", 0.03).value)

        # Timing parameters.
        self.sim_hz = float(self.declare_parameter("sim_hz", 500.0).value)
        self.max_substeps_per_tick = int(self.declare_parameter("max_substeps_per_tick", 32).value)
        self.max_elapsed_per_tick_sec = float(self.declare_parameter("max_elapsed_per_tick_sec", 0.1).value)
        self.viewer_sync_hz = float(self.declare_parameter("viewer_sync_hz", 60.0).value)
        self.timing_log_interval_sec = float(self.declare_parameter("timing_log_interval_sec", 0.0).value)
        self.startup_delay_sec = float(self.declare_parameter("startup_delay_sec", 1.0).value)

        self._validate_parameters()

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.model_dt = float(self.model.opt.timestep)
        if self.model_dt <= 0.0:
            raise ValueError("MuJoCo model timestep must be > 0")

        self.dof = max(1, min(self.dof, int(min(self.model.nq, self.model.nv))))
        self.joint_names = self._build_joint_names(self.dof)

        self.latest_tau = np.zeros(self.dof, dtype=np.float64)
        self.applied_tau = np.zeros(self.dof, dtype=np.float64)

        self.payload_body_id = -1
        self.payload_geom_id = -1
        self.payload_joint_id = -1
        self.payload_weld_eq_id = -1
        self.payload_weld_uses_sites = False
        self.ee_body_id = -1
        self.ee_payload_site_id = -1
        self.payload_grasp_site_id = -1

        self.payload_mass_enabled = None
        self.payload_inertia_enabled = None
        self.payload_mass_disabled = 0.0
        self.payload_inertia_disabled = np.zeros(3, dtype=np.float64)
        self.payload_rgba_enabled = None
        self.payload_rgba_disabled = None
        self.payload_contype_enabled = None
        self.payload_conaffinity_enabled = None
        self.payload_contype_disabled = 0
        self.payload_conaffinity_disabled = 0
        self.payload_qpos_adr = -1
        self.payload_dof_adr = -1
        self.payload_grasp_site_attached_local = None
        self.payload_grasped = False
        self.payload_suction_enabled = False

        self._configure_payload_handles()
        self._initialize_payload_pose()
        self._apply_payload_enabled_state(force=True)
        self._apply_payload_grasp_state(force=True)

        self.timer_period = 1.0 / self.sim_hz
        self.viewer_sync_period = 0.0 if self.viewer_sync_hz <= 0.0 else (1.0 / self.viewer_sync_hz)
        self.substep_accumulator = 0.0

        now_wall = time.perf_counter()
        self.start_wall_time = now_wall
        self.last_wall_time = now_wall
        self.last_timing_log_time = now_wall
        self.last_viewer_sync_time = now_wall
        self.steps_since_timing_log = 0
        self.startup_delay_released = False

        self.joint_state_pub = self.create_publisher(JointState, self.joint_state_topic, 20)
        self.payload_grasped_pub = None
        if self.publish_payload_grasped and self.payload_grasped_topic:
            self.payload_grasped_pub = self.create_publisher(Bool, self.payload_grasped_topic, 10)
        self.torque_sub = self.create_subscription(Float64MultiArray, self.torque_topic, self.on_torque_cmd, 20)
        self.payload_sub = self.create_subscription(Bool, self.payload_attached_topic, self.on_payload_attached, 20)
        self.payload_suction_cmd_sub = self.create_subscription(
            Bool, self.payload_suction_cmd_topic, self.on_payload_suction_cmd, 20)
        self.sim_timer = self.create_timer(self.timer_period, self.sim_step)

        self.viewer = None
        if self.enable_viewer:
            self._start_viewer()

        self._publish_payload_grasped()

        nominal_substeps = self.timer_period / self.model_dt
        self.get_logger().info(
            "mujoco_joint_sim_node started. "
            f"model={self.model_path} dof={self.dof} "
            f"sim_hz={self.sim_hz:.1f} timer_dt={self.timer_period:.6f} "
            f"model_dt={self.model_dt:.6f} nominal_substeps={nominal_substeps:.3f} "
            f"viewer_sync_hz={self.viewer_sync_hz:.1f} "
            f"startup_delay_sec={self.startup_delay_sec:.3f} "
            f"payload_topic={self.payload_attached_topic} "
            f"payload_enabled={self.payload_enabled} "
            f"payload_grasped_topic={self.payload_grasped_topic if self.payload_grasped_pub else 'disabled'} "
            f"payload_initial_pose={self.payload_initial_pose.tolist()}"
        )
        if abs(self.timer_period - self.model_dt) > 1e-6:
            self.get_logger().warn(
                "sim_hz and model timestep differ: "
                f"timer_dt={self.timer_period:.6f}s model_dt={self.model_dt:.6f}s"
            )

    def _validate_parameters(self) -> None:
        if not self.model_path:
            raise ValueError("Parameter 'model_path' is required")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(self.model_path)
        if self.sim_hz <= 0.0:
            raise ValueError("Parameter 'sim_hz' must be > 0")
        if self.max_substeps_per_tick <= 0:
            raise ValueError("Parameter 'max_substeps_per_tick' must be > 0")
        if self.max_elapsed_per_tick_sec <= 0.0:
            raise ValueError("Parameter 'max_elapsed_per_tick_sec' must be > 0")
        if self.startup_delay_sec < 0.0:
            raise ValueError("Parameter 'startup_delay_sec' must be >= 0")
        if self.payload_initial_pose.shape[0] != 3:
            raise ValueError("Parameter 'payload_initial_pose' must have 3 values")
        if self.payload_attach_distance_threshold_m <= 0.0:
            raise ValueError("Parameter 'payload_attach_distance_threshold_m' must be > 0")

    def _build_joint_names(self, dof: int) -> List[str]:
        names: List[str] = []
        for joint_id in range(int(self.model.njnt)):
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            if dof_adr < 0 or dof_adr >= dof:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            names.append(name if name else f"joint_{dof_adr + 1}")

        while len(names) < dof:
            names.append(f"joint_{len(names) + 1}")
        return names[:dof]

    def _yaw_to_quat(self, yaw_rad: float) -> np.ndarray:
        half = 0.5 * yaw_rad
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)

    def _quat_multiply(self, q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q_left
        w2, x2, y2, z2 = q_right
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float64)

    def _payload_initial_quat(self) -> np.ndarray:
        # Rotate the free payload so its grasp face (+Y in payload frame)
        # starts as the world top face, matching "approach from above".
        base_half = 0.25 * np.pi
        base_quat = np.array([np.cos(base_half), np.sin(base_half), 0.0, 0.0], dtype=np.float64)
        yaw_quat = self._yaw_to_quat(self.payload_initial_yaw_rad)
        quat = self._quat_multiply(yaw_quat, base_quat)
        return quat / np.linalg.norm(quat)

    def _payload_attach_site_local(self) -> np.ndarray | None:
        if self.payload_geom_id < 0:
            if self.payload_grasp_site_id >= 0:
                return np.array(self.model.site_pos[self.payload_grasp_site_id], dtype=np.float64)
            return None

        center = self.data.geom_xpos[self.payload_geom_id]
        rotation = self.data.geom_xmat[self.payload_geom_id].reshape(3, 3)
        half_extents = self.model.geom_size[self.payload_geom_id][:3]

        face_centers = []
        local_offsets = []
        for axis_idx in range(3):
            local_offset = np.zeros(3, dtype=np.float64)
            local_offset[axis_idx] = half_extents[axis_idx]
            world_offset = rotation @ local_offset
            face_centers.append(center + world_offset)
            local_offsets.append(local_offset)
            face_centers.append(center - world_offset)
            local_offsets.append(-local_offset)

        best_idx = max(range(len(face_centers)), key=lambda i: face_centers[i][2])
        return local_offsets[best_idx]

    def _sync_payload_grasp_site(self) -> np.ndarray | None:
        if self.payload_grasp_site_id < 0:
            return None

        local_offset = self._payload_attach_site_local()
        if local_offset is None:
            return None

        if not np.allclose(self.model.site_pos[self.payload_grasp_site_id], local_offset, atol=1e-12, rtol=0.0):
            self.model.site_pos[self.payload_grasp_site_id] = local_offset
            mujoco.mj_forward(self.model, self.data)

        return np.array(self.data.site_xpos[self.payload_grasp_site_id], dtype=np.float64)

    def _payload_attach_point_world(self) -> np.ndarray | None:
        return self._sync_payload_grasp_site()

    def _update_payload_weld_relpose_from_current_pose(self) -> None:
        if self.payload_weld_uses_sites:
            return

        if self.payload_weld_eq_id < 0 or self.ee_body_id < 0 or self.payload_body_id < 0:
            return

        ee_pos = np.array(self.data.xpos[self.ee_body_id], dtype=np.float64)
        ee_rot = np.array(self.data.xmat[self.ee_body_id], dtype=np.float64).reshape(3, 3)
        payload_pos = np.array(self.data.xpos[self.payload_body_id], dtype=np.float64)
        payload_rot = np.array(self.data.xmat[self.payload_body_id], dtype=np.float64).reshape(3, 3)

        rel_pos = ee_rot.T @ (payload_pos - ee_pos)
        rel_rot = ee_rot.T @ payload_rot
        rel_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(rel_quat, rel_rot.reshape(-1))

        # Weld eq_data is [anchor(3), relpos(3), relquat(4), torquescale(1)].
        if self.ee_payload_site_id >= 0 and int(self.model.site_bodyid[self.ee_payload_site_id]) == self.ee_body_id:
            self.model.eq_data[self.payload_weld_eq_id, 0:3] = self.model.site_pos[self.ee_payload_site_id]
        self.model.eq_data[self.payload_weld_eq_id, 3:6] = rel_pos
        self.model.eq_data[self.payload_weld_eq_id, 6:10] = rel_quat

    def _align_payload_to_ee_attach_pose(self) -> None:
        if (
            self.payload_qpos_adr < 0
            or self.ee_payload_site_id < 0
            or self.payload_body_id < 0
        ):
            return

        mujoco.mj_forward(self.model, self.data)
        local_offset = self._payload_attach_site_local()
        if local_offset is None:
            local_offset = self.payload_grasp_site_attached_local
        if local_offset is None:
            return

        ee_pos = np.array(self.data.site_xpos[self.ee_payload_site_id], dtype=np.float64)
        ee_rotation = np.array(self.data.site_xmat[self.ee_payload_site_id], dtype=np.float64).reshape(3, 3)
        target_payload_rotation = ee_rotation
        if self.payload_grasp_site_id >= 0:
            payload_site_local_quat = np.array(self.model.site_quat[self.payload_grasp_site_id], dtype=np.float64)
            payload_site_local_rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(payload_site_local_rotation, payload_site_local_quat)
            target_payload_rotation = ee_rotation @ payload_site_local_rotation.reshape(3, 3).T
            self.model.site_pos[self.payload_grasp_site_id] = local_offset

        target_payload_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(target_payload_quat, target_payload_rotation.reshape(-1))
        target_payload_pos = ee_pos - target_payload_rotation @ local_offset
        self.data.qpos[self.payload_qpos_adr: self.payload_qpos_adr + 3] = target_payload_pos
        self.data.qpos[self.payload_qpos_adr + 3: self.payload_qpos_adr + 7] = target_payload_quat
        if self.payload_dof_adr >= 0:
            self.data.qvel[self.payload_dof_adr: self.payload_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._update_payload_weld_relpose_from_current_pose()

    def _configure_payload_handles(self) -> None:
        self.payload_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "payload_center_link"
        )
        self.payload_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "payload_box_geom"
        )
        self.payload_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "payload_freejoint"
        )
        self.payload_weld_eq_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "payload_weld"
        )
        if self.payload_weld_eq_id >= 0:
            self.payload_weld_uses_sites = (
                int(self.model.eq_objtype[self.payload_weld_eq_id]) == int(mujoco.mjtObj.mjOBJ_SITE)
            )
        self.ee_payload_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_payload_center_site"
        )
        if self.ee_payload_site_id >= 0:
            self.ee_body_id = int(self.model.site_bodyid[self.ee_payload_site_id])
        self.payload_grasp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "payload_grasp_site"
        )
        if self.payload_grasp_site_id >= 0:
            self.payload_grasp_site_attached_local = np.array(
                self.model.site_pos[self.payload_grasp_site_id], dtype=np.float64
            )

        if self.payload_body_id < 0:
            self.get_logger().warn(
                "MuJoCo model has no body named 'payload_center_link'; payload features disabled"
            )
            return

        if self.payload_joint_id >= 0:
            self.payload_qpos_adr = int(self.model.jnt_qposadr[self.payload_joint_id])
            self.payload_dof_adr = int(self.model.jnt_dofadr[self.payload_joint_id])
        else:
            self.get_logger().warn("MuJoCo model has no freejoint named 'payload_freejoint'")

        self.payload_mass_enabled = float(self.model.body_mass[self.payload_body_id])
        self.payload_inertia_enabled = self.model.body_inertia[self.payload_body_id].copy()

        if self.payload_geom_id >= 0:
            self.payload_rgba_enabled = self.model.geom_rgba[self.payload_geom_id].copy()
            self.payload_rgba_disabled = self.payload_rgba_enabled.copy()
            self.payload_rgba_disabled[3] = 0.0
            self.payload_contype_enabled = int(self.model.geom_contype[self.payload_geom_id])
            self.payload_conaffinity_enabled = int(self.model.geom_conaffinity[self.payload_geom_id])
        else:
            self.get_logger().warn(
                "MuJoCo model has no geom named 'payload_box_geom'; payload visibility/collision disabled"
            )

    def _initialize_payload_pose(self) -> None:
        if self.payload_joint_id < 0 or self.payload_qpos_adr < 0:
            return

        quat = self._payload_initial_quat()
        self.data.qpos[self.payload_qpos_adr: self.payload_qpos_adr + 3] = self.payload_initial_pose
        self.data.qpos[self.payload_qpos_adr + 3: self.payload_qpos_adr + 7] = quat
        if self.payload_dof_adr >= 0:
            self.data.qvel[self.payload_dof_adr: self.payload_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        attach_point = self._payload_attach_point_world()
        if attach_point is not None:
            self.get_logger().info(
                f"Initialized payload attach point world position={np.array2string(attach_point, precision=6)} "
                f"quat={np.array2string(quat, precision=6)}"
            )

    def _refresh_model_constants_preserving_state(self) -> None:
        # mj_setConst refreshes model constants from qpos0 and overwrites the current
        # simulation state, so preserve the live state before calling it.
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        saved_act = self.data.act.copy()
        saved_ctrl = self.data.ctrl.copy()
        saved_qacc_warmstart = self.data.qacc_warmstart.copy()
        saved_qfrc_applied = self.data.qfrc_applied.copy()
        saved_xfrc_applied = self.data.xfrc_applied.copy()
        saved_eq_active = self.data.eq_active.copy()
        saved_mocap_pos = self.data.mocap_pos.copy()
        saved_mocap_quat = self.data.mocap_quat.copy()
        saved_time = float(self.data.time)

        mujoco.mj_setConst(self.model, self.data)

        self.data.time = saved_time
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.data.qacc_warmstart[:] = saved_qacc_warmstart
        self.data.qfrc_applied[:] = saved_qfrc_applied
        self.data.xfrc_applied[:] = saved_xfrc_applied
        self.data.eq_active[:] = saved_eq_active
        if saved_act.size > 0:
            self.data.act[:] = saved_act
        if saved_ctrl.size > 0:
            self.data.ctrl[:] = saved_ctrl
        if saved_mocap_pos.size > 0:
            self.data.mocap_pos[:] = saved_mocap_pos
            self.data.mocap_quat[:] = saved_mocap_quat

        mujoco.mj_forward(self.model, self.data)

    def _apply_payload_enabled_state(self, force: bool = False) -> None:
        if self.payload_body_id < 0 or self.payload_mass_enabled is None or self.payload_inertia_enabled is None:
            return

        target_mass = self.payload_mass_enabled if self.payload_enabled else self.payload_mass_disabled
        target_inertia = self.payload_inertia_enabled if self.payload_enabled else self.payload_inertia_disabled

        same_mass = abs(float(self.model.body_mass[self.payload_body_id]) - target_mass) <= 1e-12
        same_inertia = np.allclose(
            self.model.body_inertia[self.payload_body_id], target_inertia, atol=1e-12, rtol=0.0)

        same_rgba = True
        same_collision = True
        if self.payload_geom_id >= 0 and self.payload_rgba_enabled is not None and self.payload_rgba_disabled is not None:
            target_rgba = self.payload_rgba_enabled if self.payload_enabled else self.payload_rgba_disabled
            same_rgba = np.allclose(
                self.model.geom_rgba[self.payload_geom_id], target_rgba, atol=1e-12, rtol=0.0)
        if self.payload_geom_id >= 0 and self.payload_contype_enabled is not None and self.payload_conaffinity_enabled is not None:
            collisions_enabled = self.payload_enabled
            target_contype = self.payload_contype_enabled if collisions_enabled else self.payload_contype_disabled
            target_conaffinity = self.payload_conaffinity_enabled if collisions_enabled else self.payload_conaffinity_disabled
            same_collision = (
                int(self.model.geom_contype[self.payload_geom_id]) == target_contype
                and int(self.model.geom_conaffinity[self.payload_geom_id]) == target_conaffinity
            )

        if not force and same_mass and same_inertia and same_rgba and same_collision:
            return

        self.model.body_mass[self.payload_body_id] = target_mass
        self.model.body_inertia[self.payload_body_id] = target_inertia

        if self.payload_geom_id >= 0 and self.payload_rgba_enabled is not None and self.payload_rgba_disabled is not None:
            self.model.geom_rgba[self.payload_geom_id] = (
                self.payload_rgba_enabled if self.payload_enabled else self.payload_rgba_disabled
            )
        if self.payload_geom_id >= 0 and self.payload_contype_enabled is not None and self.payload_conaffinity_enabled is not None:
            collisions_enabled = self.payload_enabled
            self.model.geom_contype[self.payload_geom_id] = (
                self.payload_contype_enabled if collisions_enabled else self.payload_contype_disabled
            )
            self.model.geom_conaffinity[self.payload_geom_id] = (
                self.payload_conaffinity_enabled if collisions_enabled else self.payload_conaffinity_disabled
            )

        self._refresh_model_constants_preserving_state()

    def _apply_payload_grasp_state(self, force: bool = False) -> None:
        if self.payload_weld_eq_id < 0:
            return

        target_active = bool(self.payload_enabled and self.payload_grasped)
        current_active = bool(self.data.eq_active[self.payload_weld_eq_id])
        if not force and current_active == target_active:
            return

        if target_active and not current_active:
            # Make the attach frames coincide before engaging the weld, so the
            # constraint starts from a zero-error configuration.
            self._align_payload_to_ee_attach_pose()

        self.data.eq_active[self.payload_weld_eq_id] = 1 if target_active else 0
        if self.payload_dof_adr >= 0:
            self.data.qvel[self.payload_dof_adr: self.payload_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if target_active and self.ee_payload_site_id >= 0 and self.payload_grasp_site_id >= 0:
            site_error = float(np.linalg.norm(
                self.data.site_xpos[self.ee_payload_site_id] - self.data.site_xpos[self.payload_grasp_site_id]
            ))
            if site_error > 1e-3:
                self.get_logger().warn(
                    f"Payload weld engaged with residual site error {site_error:.6f} m"
                )

    def _publish_payload_grasped(self) -> None:
        if self.payload_grasped_pub is None:
            return
        msg = Bool()
        msg.data = bool(self.payload_enabled and self.payload_grasped)
        self.payload_grasped_pub.publish(msg)

    def _set_payload_grasped(self, grasped: bool, reason: str) -> None:
        target_grasped = bool(grasped and self.payload_enabled)
        if self.payload_grasped == target_grasped and self.payload_enabled:
            return

        if target_grasped:
            self._apply_payload_enabled_state(force=True)

        self.payload_grasped = target_grasped
        self._apply_payload_grasp_state(force=True)
        self._publish_payload_grasped()
        self.get_logger().info(
            f"MuJoCo payload grasp state changed to {'grasped' if self.payload_grasped else 'released'} ({reason})"
        )

    def _can_attach_payload(self, log_reject: bool = True) -> bool:
        if not self.payload_enabled:
            return False
        if self.ee_payload_site_id < 0:
            self.get_logger().warn("EE attach site missing; cannot evaluate suction attach")
            return False

        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.site_xpos[self.ee_payload_site_id]
        payload_pos = self._payload_attach_point_world()
        if payload_pos is None:
            self.get_logger().warn("Payload attach point unavailable; cannot evaluate suction attach")
            return False
        distance = float(np.linalg.norm(ee_pos - payload_pos))
        self.get_logger().info(
            "Evaluating payload attach: "
            f"ee_pos={np.array2string(ee_pos, precision=6)} "
            f"payload_pos={np.array2string(payload_pos, precision=6)} "
            f"distance={distance:.4f} m"
        )
        if distance <= self.payload_attach_distance_threshold_m:
            return True

        if log_reject:
            self.get_logger().warn(
                "Payload suction attach rejected: "
                f"ee_pos={np.array2string(ee_pos, precision=6)} "
                f"payload_pos={np.array2string(payload_pos, precision=6)} "
                f"distance={distance:.4f} m "
                f"threshold={self.payload_attach_distance_threshold_m:.4f} m"
            )
        return False

    def _start_viewer(self) -> None:
        if mj_viewer is None:
            self.get_logger().warn("mujoco.viewer unavailable, running without viewer")
            return
        try:
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
        except Exception as exc:  # pragma: no cover
            self.viewer = None
            self.get_logger().warn(f"Failed to start MuJoCo viewer: {exc}")

    def on_torque_cmd(self, msg: Float64MultiArray) -> None:
        if not msg.data:
            return
        n = min(self.dof, len(msg.data))
        self.latest_tau[:n] = np.asarray(msg.data[:n], dtype=np.float64)
        if n < self.dof:
            self.latest_tau[n:] = 0.0

    def on_payload_attached(self, msg: Bool) -> None:
        self.payload_attached = bool(msg.data)

    def on_payload_suction_cmd(self, msg: Bool) -> None:
        if not self.payload_enabled:
            self.get_logger().info("Ignoring suction command because simulation payload is disabled")
            self.payload_suction_enabled = False
            self._set_payload_grasped(False, "payload disabled")
            return

        if msg.data:
            self.payload_suction_enabled = True
            if self.payload_grasped:
                return
            if self._can_attach_payload():
                self._set_payload_grasped(True, "suction command")
        else:
            self.payload_suction_enabled = False
            if not self.payload_grasped:
                return
            self._set_payload_grasped(False, "release command")

    def _update_elapsed(self) -> float:
        now_wall = time.perf_counter()
        elapsed = now_wall - self.last_wall_time
        self.last_wall_time = now_wall
        if elapsed < 0.0:
            elapsed = 0.0
        if elapsed > self.max_elapsed_per_tick_sec:
            elapsed = self.max_elapsed_per_tick_sec
        return elapsed

    def _compute_substeps(self, elapsed_wall: float) -> int:
        self.substep_accumulator += elapsed_wall
        substeps = int(self.substep_accumulator / self.model_dt)
        if substeps <= 0:
            return 0

        if substeps > self.max_substeps_per_tick:
            substeps = self.max_substeps_per_tick

        self.substep_accumulator -= substeps * self.model_dt
        if self.substep_accumulator < 0.0:
            self.substep_accumulator = 0.0
        return substeps

    def _step_model(self, substeps: int) -> None:
        if substeps <= 0:
            return
        self.applied_tau[:] = self.latest_tau
        for _ in range(substeps):
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[: self.dof] = self.applied_tau
            mujoco.mj_step(self.model, self.data)
        self.steps_since_timing_log += substeps

    def _maybe_sync_viewer(self, now_wall: float) -> None:
        if self.viewer is None:
            return
        if self.viewer_sync_period > 0.0 and (now_wall - self.last_viewer_sync_time) < self.viewer_sync_period:
            return
        try:
            self.viewer.sync()
            self.last_viewer_sync_time = now_wall
        except Exception:
            self.viewer = None
            self.get_logger().warn("MuJoCo viewer sync failed, disabling viewer")

    def _maybe_log_timing(self, now_wall: float) -> None:
        if self.timing_log_interval_sec <= 0.0:
            return
        wall_dt = now_wall - self.last_timing_log_time
        if wall_dt < self.timing_log_interval_sec:
            return

        step_rate = self.steps_since_timing_log / wall_dt if wall_dt > 1e-9 else 0.0
        sim_dt = self.steps_since_timing_log * self.model_dt
        rtf = sim_dt / wall_dt if wall_dt > 1e-9 else 0.0
        self.get_logger().info(
            "sim_timing "
            f"step_rate={step_rate:.1f}Hz rtf={rtf:.3f} "
            f"backlog={self.substep_accumulator * 1000.0:.3f}ms"
        )
        self.steps_since_timing_log = 0
        self.last_timing_log_time = now_wall

    def sim_step(self) -> None:
        elapsed_wall = self._update_elapsed()
        now_wall = self.last_wall_time

        if (now_wall - self.start_wall_time) < self.startup_delay_sec:
            self.applied_tau[:] = 0.0
            self.publish_joint_state()
            self._maybe_sync_viewer(now_wall)
            self._maybe_log_timing(now_wall)
            return

        if not self.startup_delay_released:
            self.startup_delay_released = True
            self.get_logger().info("startup delay elapsed, simulator stepping enabled")

        substeps = self._compute_substeps(elapsed_wall)
        self._step_model(substeps)
        if self.payload_enabled and not self.payload_grasped:
            self._sync_payload_grasp_site()
            if self.payload_suction_enabled and self._can_attach_payload(log_reject=False):
                self._set_payload_grasped(True, "latched suction")
        self.publish_joint_state()

        self._maybe_sync_viewer(now_wall)
        self._maybe_log_timing(now_wall)

    def publish_joint_state(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.data.qpos[: self.dof].tolist()
        msg.velocity = self.data.qvel[: self.dof].tolist()
        msg.effort = self.applied_tau.tolist()
        self.joint_state_pub.publish(msg)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MujocoJointSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
