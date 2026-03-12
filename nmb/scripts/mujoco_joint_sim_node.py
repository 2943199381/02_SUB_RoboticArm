#!/usr/bin/env python3
import os
import time
from typing import List

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
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
        self.enable_viewer = bool(self.declare_parameter("enable_viewer", True).value)

        # Timing parameters.
        self.sim_hz = float(self.declare_parameter("sim_hz", 500.0).value)
        self.max_substeps_per_tick = int(self.declare_parameter("max_substeps_per_tick", 32).value)
        self.max_elapsed_per_tick_sec = float(self.declare_parameter("max_elapsed_per_tick_sec", 0.1).value)
        self.viewer_sync_hz = float(self.declare_parameter("viewer_sync_hz", 60.0).value)
        self.timing_log_interval_sec = float(self.declare_parameter("timing_log_interval_sec", 2.0).value)
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
        self.torque_sub = self.create_subscription(Float64MultiArray, self.torque_topic, self.on_torque_cmd, 20)
        self.sim_timer = self.create_timer(self.timer_period, self.sim_step)

        self.viewer = None
        if self.enable_viewer:
            self._start_viewer()

        nominal_substeps = self.timer_period / self.model_dt
        self.get_logger().info(
            "mujoco_joint_sim_node started. "
            f"model={self.model_path} dof={self.dof} "
            f"sim_hz={self.sim_hz:.1f} timer_dt={self.timer_period:.6f} "
            f"model_dt={self.model_dt:.6f} nominal_substeps={nominal_substeps:.3f} "
            f"viewer_sync_hz={self.viewer_sync_hz:.1f} "
            f"startup_delay_sec={self.startup_delay_sec:.3f}"
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
            # qfrc_applied is cleared by mj_step(), so write on each substep.
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
        # When the context is externally shutdown (e.g. timeout/launch stop),
        # Humble may raise a backend exception during executor wait.
        if rclpy.ok():
            raise
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
