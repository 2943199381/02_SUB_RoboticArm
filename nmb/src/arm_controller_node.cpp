#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#if __has_include(<pinocchio/algorithm/frames.hpp>) && \
  __has_include(<pinocchio/algorithm/jacobian.hpp>) && \
  __has_include(<pinocchio/algorithm/kinematics.hpp>) && \
  __has_include(<pinocchio/algorithm/rnea.hpp>) && \
  __has_include(<pinocchio/parsers/urdf.hpp>)
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#define NMB_HAS_PINOCCHIO 1
#else
#define NMB_HAS_PINOCCHIO 0
#endif

class ArmControllerNode : public rclcpp::Node
{
public:
  ArmControllerNode()
  : Node("arm_controller_node")
  {
    control_hz_ = declare_parameter<double>("control_hz", 500.0);
    const auto legacy_kp_param = declare_parameter<std::vector<double>>(
      "kp", std::vector<double>{2.0, 2.0, 2.0, 0.0});
    const auto legacy_kd_param = declare_parameter<std::vector<double>>(
      "kd", std::vector<double>{1.0, 1.0, 1.0, 0.0});
    const auto kp_empty_param = declare_parameter<std::vector<double>>("kp_empty", legacy_kp_param);
    const auto kd_empty_param = declare_parameter<std::vector<double>>("kd_empty", legacy_kd_param);
    const auto kp_payload_param = declare_parameter<std::vector<double>>("kp_payload", kp_empty_param);
    const auto kd_payload_param = declare_parameter<std::vector<double>>("kd_payload", kd_empty_param);
    const auto initial_hold_q_param = declare_parameter<std::vector<double>>(
      "initial_hold_q", std::vector<double>{0.0, 0.0, 0.0, 0.0});
    const auto armature_param = declare_parameter<std::vector<double>>("armature", std::vector<double>{0.0, 0.0, 0.0, 0.0});
    kp_empty_ = vector_param_to_eigen4(kp_empty_param, 20.0);
    kd_empty_ = vector_param_to_eigen4(kd_empty_param, 1.0);
    kp_payload_ = vector_param_to_eigen4(kp_payload_param, 20.0);
    kd_payload_ = vector_param_to_eigen4(kd_payload_param, 1.0);
    initial_hold_q_ = vector_param_to_eigen4(initial_hold_q_param, 0.0);
    armature_values = vector_param_to_eigen4(armature_param, 0.0);

    const std::string legacy_urdf_path = declare_parameter<std::string>("urdf_path", "");
    const std::string legacy_task_frame = declare_parameter<std::string>("ee_frame", "ee_link");
    urdf_path_empty_ = declare_parameter<std::string>("urdf_path_empty", legacy_urdf_path);
    urdf_path_payload_ = declare_parameter<std::string>("urdf_path_payload", urdf_path_empty_);
    task_frame_empty_ = declare_parameter<std::string>("task_frame_empty", legacy_task_frame);
    task_frame_payload_ = declare_parameter<std::string>("task_frame_payload", task_frame_empty_);
    payload_attached_topic_ = declare_parameter<std::string>("payload_attached_topic", "/payload_attached");
    payload_attached_ = declare_parameter<bool>("payload_attached_initial", false);

    model_transition_cmd_topic_ = declare_parameter<std::string>(
      "model_transition_cmd_topic", "/arm_model_transition_cmd");
    model_blend_status_topic_ = declare_parameter<std::string>(
      "model_blend_status_topic", "/arm_model_blend_status");
    pd_task_wrench_topic_ = declare_parameter<std::string>(
      "pd_task_wrench_topic", "/arm_pd_task_wrench");
    model_blend_duration_sec_ = std::max(
      1e-3, declare_parameter<double>("model_blend_duration_sec", 0.50));
    wrench_damping_ = std::max(1e-9, declare_parameter<double>("wrench_damping", 1e-3));

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      std::bind(&ArmControllerNode::on_joint_state, this, std::placeholders::_1));

    planned_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/planned_joint_state", 10,
      std::bind(&ArmControllerNode::on_planned_joint_state, this, std::placeholders::_1));

    payload_state_sub_ = create_subscription<std_msgs::msg::Bool>(
      payload_attached_topic_, 10,
      std::bind(&ArmControllerNode::on_payload_attached, this, std::placeholders::_1));

    model_transition_cmd_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      model_transition_cmd_topic_, 10,
      std::bind(&ArmControllerNode::on_model_transition_cmd, this, std::placeholders::_1));

    torque_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/joint_torque_cmd", 10);
    current_task_pose_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/current_ee_pose_pitch", 10);
    pd_task_wrench_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(pd_task_wrench_topic_, 10);
    model_blend_status_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(model_blend_status_topic_, 10);

#if NMB_HAS_PINOCCHIO
    load_pinocchio_context(empty_context_, "empty", urdf_path_empty_, task_frame_empty_);
    load_pinocchio_context(payload_context_, "payload", urdf_path_payload_, task_frame_payload_);
    log_active_context();
#else
    RCLCPP_WARN(
      get_logger(),
      "Pinocchio headers not found at compile time. Running without model-based inverse dynamics, pose, or wrench estimation.");
#endif

    blend_start_time_ = now();
    blend_start_alpha_ = payload_attached_ ? 1.0 : 0.0;
    blend_target_alpha_ = blend_start_alpha_;
    blend_duration_sec_ = model_blend_duration_sec_;

    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, control_hz_));
    control_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&ArmControllerNode::control_loop, this));

    RCLCPP_INFO(
      get_logger(),
      "arm_controller_node started. dof=%d control_hz=%.1f payload_topic=%s transition_cmd=%s blend_status=%s wrench_topic=%s initial_hold_q=[%.3f %.3f %.3f %.3f]",
      kDof, control_hz_, payload_attached_topic_.c_str(), model_transition_cmd_topic_.c_str(),
      model_blend_status_topic_.c_str(), pd_task_wrench_topic_.c_str(),
      initial_hold_q_(0), initial_hold_q_(1), initial_hold_q_(2), initial_hold_q_(3));
  }

private:
  static constexpr int kDof = 4;

  struct WrenchEstimate
  {
    Eigen::Vector4d wrench = Eigen::Vector4d::Zero();
    bool valid {false};
  };

  static Eigen::Vector4d vector_param_to_eigen4(const std::vector<double> & src, double fill)
  {
    Eigen::Vector4d out = Eigen::Vector4d::Constant(fill);
    const size_t n = std::min(static_cast<size_t>(kDof), src.size());
    for (size_t i = 0; i < n; ++i) {
      out(static_cast<Eigen::Index>(i)) = src[i];
    }
    return out;
  }

  static double wrap_to_pi(double a)
  {
    while (a > M_PI) {
      a -= 2.0 * M_PI;
    }
    while (a < -M_PI) {
      a += 2.0 * M_PI;
    }
    return a;
  }

  void refresh_blend_state()
  {
    if (!blend_active_) {
      return;
    }

    const double elapsed = (now() - blend_start_time_).seconds();
    if (elapsed >= blend_duration_sec_) {
      blend_active_ = false;
      blend_start_alpha_ = blend_target_alpha_;
    }
  }

  double current_payload_alpha() const
  {
    if (!blend_active_) {
      return blend_target_alpha_;
    }

    const double progress = std::clamp(
      (now() - blend_start_time_).seconds() / std::max(1e-6, blend_duration_sec_), 0.0, 1.0);
    return blend_start_alpha_ + progress * (blend_target_alpha_ - blend_start_alpha_);
  }

  Eigen::Vector4d current_kp() const
  {
    const double payload_alpha = current_payload_alpha();
    return (1.0 - payload_alpha) * kp_empty_ + payload_alpha * kp_payload_;
  }

  Eigen::Vector4d current_kd() const
  {
    const double payload_alpha = current_payload_alpha();
    return (1.0 - payload_alpha) * kd_empty_ + payload_alpha * kd_payload_;
  }

  void publish_blend_status()
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.data = {
      current_payload_alpha(),
      blend_active_ ? 1.0 : 0.0,
      blend_target_alpha_
    };
    model_blend_status_pub_->publish(msg);
  }

  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    recieve_q_ = true;
    for (size_t i = 0; i < 4; ++i) {
      q_(i) = msg->position[i];
      dq_(i) = msg->velocity[i];
    }
    if (!has_recorded_hold_q_ ) {
      recorded_hold_q_ = q_;
      desired_q_ = recorded_hold_q_;
      desired_dq_.setZero();
      desired_ddq_.setZero();
      has_recorded_hold_q_ = true;
      RCLCPP_INFO(
        get_logger(),
        "Recorded controller hold origin from joint state: [%.3f %.3f %.3f %.3f]",
        recorded_hold_q_(0), recorded_hold_q_(1), recorded_hold_q_(2), recorded_hold_q_(3));
    }
  }

  void on_planned_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {

    for (size_t i = 0; i < 4; ++i) {
      desired_q_(i) = msg->position[i];
      desired_dq_(i) = msg->velocity[i];
      desired_ddq_(i) = msg->effort[i];
    }

    has_planned_state_ = true;
  }

  void on_payload_attached(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (payload_attached_ == msg->data && !blend_active_) {
      return;
    }

    payload_attached_ = msg->data;
    if (!blend_active_) {
      blend_start_alpha_ = payload_attached_ ? 1.0 : 0.0;
      blend_target_alpha_ = blend_start_alpha_;
    }

#if NMB_HAS_PINOCCHIO
    log_active_context();
#endif
  }

  void on_model_transition_cmd(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.empty()) {
      RCLCPP_WARN(get_logger(), "Ignoring empty model transition command.");
      return;
    }

    refresh_blend_state();

    const double target_alpha = std::clamp(msg->data[0], 0.0, 1.0);
    const double duration_sec = msg->data.size() >= 2U ?
      std::max(1e-3, msg->data[1]) : model_blend_duration_sec_;
    const double start_alpha = current_payload_alpha();

    blend_start_alpha_ = start_alpha;
    blend_target_alpha_ = target_alpha;
    blend_duration_sec_ = duration_sec;
    blend_start_time_ = now();
    blend_active_ = std::abs(blend_target_alpha_ - blend_start_alpha_) > 1e-6;

    if (!blend_active_) {
      blend_start_alpha_ = blend_target_alpha_;
    }

    RCLCPP_INFO(
      get_logger(),
      "Received model transition command: start_alpha=%.3f target_alpha=%.3f duration=%.3f active=%s",
      blend_start_alpha_, blend_target_alpha_, blend_duration_sec_, blend_active_ ? "true" : "false");
  }

#if NMB_HAS_PINOCCHIO
  struct ModelContext
  {
    std::string label;
    std::string urdf_path;
    std::string task_frame;
    pinocchio::Model model;
    mutable std::unique_ptr<pinocchio::Data> data;
    pinocchio::FrameIndex frame_id {0};
    bool loaded {false};
    double pitch_reference_rad {0.0};
  };

  static double raw_pitch_from_rotation(const Eigen::Matrix3d & r)
  {
    return std::atan2(r(2, 1), r(2, 2));
  }

  bool load_pinocchio_context(
    ModelContext & ctx,
    const std::string & label,
    const std::string & urdf_path,
    const std::string & task_frame)
  {
    ctx.label = label;
    ctx.urdf_path = urdf_path;
    ctx.task_frame = task_frame;
    ctx.loaded = false;
    ctx.data.reset();

    if (ctx.urdf_path.empty()) {
      RCLCPP_WARN(get_logger(), "Controller context '%s' has empty urdf path.", label.c_str());
      return false;
    }

    try {
      pinocchio::urdf::buildModel(ctx.urdf_path, ctx.model);
      ctx.model.armature = armature_values;
      ctx.data = std::make_unique<pinocchio::Data>(ctx.model);
      ctx.frame_id = ctx.model.getFrameId(ctx.task_frame);
      if (ctx.frame_id >= ctx.model.frames.size()) {
        RCLCPP_ERROR(
          get_logger(), "Controller context '%s' frame '%s' not found in URDF '%s'.",
          label.c_str(), ctx.task_frame.c_str(), ctx.urdf_path.c_str());
        return false;
      }

      if (ctx.model.nq != kDof || ctx.model.nv != kDof) {
        RCLCPP_ERROR(
          get_logger(),
          "Controller context '%s' requires nq=nv=dof=4, got nq=%d nv=%d.",
          label.c_str(), ctx.model.nq, ctx.model.nv);
        return false;
      }

      const Eigen::Vector4d q_zero = Eigen::Vector4d::Zero();
      pinocchio::forwardKinematics(ctx.model, *ctx.data, q_zero);
      pinocchio::updateFramePlacements(ctx.model, *ctx.data);
      ctx.pitch_reference_rad =
        raw_pitch_from_rotation(ctx.data->oMf[ctx.frame_id].rotation());
      ctx.loaded = true;

      RCLCPP_INFO(
        get_logger(), "Controller context '%s' loaded. urdf=%s task_frame=%s",
        ctx.label.c_str(), ctx.urdf_path.c_str(), ctx.task_frame.c_str());
      return true;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        get_logger(), "Failed to load controller context '%s': %s",
        label.c_str(), e.what());
      return false;
    }
  }

  const ModelContext * context_for_payload_flag(bool payload_flag) const
  {
    if (payload_flag && payload_context_.loaded) {
      return &payload_context_;
    }
    if (!payload_flag && empty_context_.loaded) {
      return &empty_context_;
    }
    if (empty_context_.loaded) {
      return &empty_context_;
    }
    if (payload_context_.loaded) {
      return &payload_context_;
    }
    return nullptr;
  }

  const ModelContext * active_context() const
  {
    return context_for_payload_flag(payload_attached_);
  }

  void log_active_context() const
  {
    const auto * ctx = active_context();
    if (!ctx) {
      RCLCPP_WARN(get_logger(), "No valid controller context loaded.");
      return;
    }
    RCLCPP_INFO(
      get_logger(), "Controller active context=%s task_frame=%s payload_attached=%s",
      ctx->label.c_str(), ctx->task_frame.c_str(), payload_attached_ ? "true" : "false");
  }

  bool compute_task_pose(
    const ModelContext & ctx,
    const Eigen::Vector4d & q_vec,
    Eigen::Vector4d & out_pose) const
  {
    if (!ctx.loaded || !ctx.data) {
      return false;
    }

    pinocchio::forwardKinematics(ctx.model, *ctx.data, q_vec);
    pinocchio::updateFramePlacements(ctx.model, *ctx.data);
    const auto & frame = ctx.data->oMf[ctx.frame_id];

    out_pose(0) = frame.translation().x();
    out_pose(1) = frame.translation().y();
    out_pose(2) = frame.translation().z();
    out_pose(3) = wrap_to_pi(raw_pitch_from_rotation(frame.rotation()) - ctx.pitch_reference_rad);
    return out_pose.allFinite();
  }

  bool compute_task_jacobian(
    const ModelContext & ctx,
    const Eigen::Vector4d & q_vec,
    Eigen::Matrix4d & J) const
  {
    if (!ctx.loaded || !ctx.data) {
      return false;
    }

    pinocchio::forwardKinematics(ctx.model, *ctx.data, q_vec);
    pinocchio::updateFramePlacements(ctx.model, *ctx.data);
    pinocchio::computeJointJacobians(ctx.model, *ctx.data, q_vec);

    Eigen::Matrix<double, 6, 4> J6_world_aligned = Eigen::Matrix<double, 6, 4>::Zero();
    Eigen::Matrix<double, 6, 4> J6_local = Eigen::Matrix<double, 6, 4>::Zero();
    pinocchio::getFrameJacobian(
      ctx.model, *ctx.data, ctx.frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J6_world_aligned);
    pinocchio::getFrameJacobian(
      ctx.model, *ctx.data, ctx.frame_id, pinocchio::LOCAL, J6_local);

    J.setZero();
    J.topRows<3>() = J6_world_aligned.topRows<3>();
    J.row(3) = J6_local.bottomRows<3>().row(0);
    return J.allFinite();
  }

  Eigen::Vector4d compute_inverse_dynamics(const ModelContext & ctx, const Eigen::Vector4d & ddq_des) const
  {
    if (!ctx.loaded || !ctx.data) {
      return Eigen::Vector4d::Zero();
    }

    const Eigen::VectorXd tau_full = pinocchio::rnea(ctx.model, *ctx.data, q_, dq_, ddq_des);
    Eigen::Vector4d tau = Eigen::Vector4d::Zero();
    const Eigen::Index n = std::min<Eigen::Index>(tau_full.size(), kDof);
    for (Eigen::Index i = 0; i < n; ++i) {
      tau(i) = tau_full(i);
    }
    return tau;
  }

  Eigen::Vector4d compute_blended_inverse_dynamics(const Eigen::Vector4d & ddq_des) const
  {
    const bool have_empty = empty_context_.loaded && empty_context_.data;
    const bool have_payload = payload_context_.loaded && payload_context_.data;
    if (!have_empty && !have_payload) {
      return Eigen::Vector4d::Zero();
    }

    const double payload_alpha = current_payload_alpha();
    if (have_empty && have_payload) {
      return (1.0 - payload_alpha) * compute_inverse_dynamics(empty_context_, ddq_des) +
             payload_alpha * compute_inverse_dynamics(payload_context_, ddq_des);
    }
    if (payload_alpha >= 0.5 && have_payload) {
      return compute_inverse_dynamics(payload_context_, ddq_des);
    }
    if (have_empty) {
      return compute_inverse_dynamics(empty_context_, ddq_des);
    }
    return compute_inverse_dynamics(payload_context_, ddq_des);
  }

  void publish_current_task_pose(const ModelContext & ctx)
  {
    Eigen::Vector4d pose = Eigen::Vector4d::Zero();
    if (!compute_task_pose(ctx, q_, pose)) {
      return;
    }

    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(4);
    for (int i = 0; i < kDof; ++i) {
      msg.data[static_cast<size_t>(i)] = pose(i);
    }
    current_task_pose_pub_->publish(msg);
  }

  WrenchEstimate estimate_pd_task_wrench(
    const ModelContext & ctx,
    const Eigen::Vector4d & tau_pd) const
  {
    WrenchEstimate out;
    if (!ctx.loaded || !ctx.data) {
      return out;
    }

    pinocchio::forwardKinematics(ctx.model, *ctx.data, q_);
    pinocchio::updateFramePlacements(ctx.model, *ctx.data);
    pinocchio::computeJointJacobians(ctx.model, *ctx.data, q_);

    Eigen::Matrix<double, 6, 4> J6_world_aligned = Eigen::Matrix<double, 6, 4>::Zero();
    Eigen::Matrix<double, 6, 4> J6_local = Eigen::Matrix<double, 6, 4>::Zero();
    pinocchio::getFrameJacobian(
      ctx.model, *ctx.data, ctx.frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J6_world_aligned);
    pinocchio::getFrameJacobian(
      ctx.model, *ctx.data, ctx.frame_id, pinocchio::LOCAL, J6_local);

    // Reconstruct the task-space feedback wrench from the Jacobian transpose
    // relation tau_feedback = J^T * wrench. We intentionally use tau_pd
    // instead of the total torque so pickup verification reflects the
    // unmodelled external load rather than nominal inverse-dynamics feedforward.
    const Eigen::Matrix<double, 3, 4> J_linear = J6_world_aligned.topRows<3>();
    Eigen::Matrix3d H_linear = J_linear * J_linear.transpose();
    H_linear.diagonal().array() += wrench_damping_;
    out.wrench.setZero();
    out.wrench.head<3>() = H_linear.ldlt().solve(J_linear * tau_pd);

    const Eigen::RowVector4d J_pitch = J6_local.bottomRows<3>().row(0);
    const double pitch_denom = J_pitch.squaredNorm() + wrench_damping_;
    if (pitch_denom > 1e-12) {
      out.wrench(3) = (J_pitch * tau_pd)(0) / pitch_denom;
    }

    out.valid = out.wrench.allFinite();
    if (!out.valid) {
      out.wrench.setZero();
    }
    return out;
  }
#endif

  void publish_pd_task_wrench(const WrenchEstimate & wrench_estimate)
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(5, 0.0);
    for (int i = 0; i < kDof; ++i) {
      msg.data[static_cast<size_t>(i)] = wrench_estimate.wrench(i);
    }
    msg.data[4] = wrench_estimate.valid ? 1.0 : 0.0;
    pd_task_wrench_pub_->publish(msg);
  }

  void control_loop()
  {
    if(!recieve_q_)
    {
      Eigen::Vector4d tau = Eigen::Vector4d::Zero();
      publish_torque(tau);
      return;
    }
    else
    {
      
    refresh_blend_state();

    if (!has_planned_state_) {
      // desired_q_ = has_recorded_hold_q_ ? recorded_hold_q_ : initial_hold_q_;
      desired_q_ = q_;
      desired_dq_.setZero();
      desired_ddq_.setZero();
    }

    Eigen::Vector4d pos_err = Eigen::Vector4d::Zero();
    for (int i = 0; i < kDof; ++i) {
      pos_err(i) = wrap_to_pi(desired_q_(i) - q_(i));
    }
    const Eigen::Vector4d vel_err = desired_dq_ - dq_;
    const Eigen::Vector4d kp = current_kp();
    const Eigen::Vector4d kd = current_kd();
    const Eigen::Vector4d tau_pd = 0.08 * kp.cwiseProduct(pos_err) + 0.06 * kd.cwiseProduct(vel_err);

    Eigen::Vector4d tau_model = Eigen::Vector4d::Zero();
    WrenchEstimate wrench_estimate;
#if NMB_HAS_PINOCCHIO
    const auto * ctx = active_context();
    if (ctx && ctx->loaded && ctx->data) {
      publish_current_task_pose(*ctx);
      wrench_estimate = estimate_pd_task_wrench(*ctx, tau_pd);
    }
    tau_model = compute_blended_inverse_dynamics(desired_ddq_);
#endif

    publish_pd_task_wrench(wrench_estimate);
    publish_blend_status();

    const Eigen::Vector4d tau = tau_pd + tau_model;

    publish_torque(tau);
    }
  }

  void publish_torque(const Eigen::Vector4d & tau)
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(kDof);
    for (int i = 0; i < kDof; ++i) {
      msg.data[static_cast<size_t>(i)] = tau(i);
    }
    torque_pub_->publish(msg);
  }

private:
  double control_hz_ {500.0};
  Eigen::Vector4d kp_empty_ = Eigen::Vector4d::Zero();
  Eigen::Vector4d kd_empty_ = Eigen::Vector4d::Zero();
  Eigen::Vector4d kp_payload_ = Eigen::Vector4d::Zero();
  Eigen::Vector4d kd_payload_ = Eigen::Vector4d::Zero();
  Eigen::Vector4d initial_hold_q_ = Eigen::Vector4d::Zero();
  std::string urdf_path_empty_;
  std::string urdf_path_payload_;
  std::string task_frame_empty_;
  std::string task_frame_payload_;
  std::string payload_attached_topic_;
  bool payload_attached_ {false};

  std::string model_transition_cmd_topic_;
  std::string model_blend_status_topic_;
  std::string pd_task_wrench_topic_;
  double model_blend_duration_sec_ {0.50};
  double wrench_damping_ {1e-3};
  Eigen::Vector4d armature_values = Eigen::Vector4d::Zero();

  Eigen::Vector4d q_ = Eigen::Vector4d::Zero();
  Eigen::Vector4d dq_ = Eigen::Vector4d::Zero();
  bool has_joint_state_ {false};

  Eigen::Vector4d desired_q_ = Eigen::Vector4d::Zero();
  Eigen::Vector4d desired_dq_ = Eigen::Vector4d::Zero();
  Eigen::Vector4d desired_ddq_ = Eigen::Vector4d::Zero();
  bool has_planned_state_ {false};
  Eigen::Vector4d recorded_hold_q_ = Eigen::Vector4d::Zero();
  bool has_recorded_hold_q_ {false};

  double blend_start_alpha_ {0.0};
  double blend_target_alpha_ {0.0};
  double blend_duration_sec_ {0.50};
  bool blend_active_ {false};
  bool recieve_q_ {false};
  rclcpp::Time blend_start_time_;

#if NMB_HAS_PINOCCHIO
  ModelContext empty_context_;
  ModelContext payload_context_;
#endif

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr planned_state_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr payload_state_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr model_transition_cmd_sub_;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr torque_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr current_task_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pd_task_wrench_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr model_blend_status_pub_;

  rclcpp::TimerBase::SharedPtr control_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArmControllerNode>());
  rclcpp::shutdown();
  return 0;
}
