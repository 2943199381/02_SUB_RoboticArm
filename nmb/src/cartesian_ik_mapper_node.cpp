#include <algorithm>
#include <cmath>
#include <limits>
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
  __has_include(<pinocchio/parsers/urdf.hpp>)
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#define NMB_HAS_PINOCCHIO 1
#else
#define NMB_HAS_PINOCCHIO 0
#endif

class CartesianIkMapperNode : public rclcpp::Node
{
public:
  CartesianIkMapperNode()
  : Node("cartesian_ik_mapper_node")
  {
    const std::string legacy_urdf_path = declare_parameter<std::string>("urdf_path", "");
    const std::string legacy_task_frame = declare_parameter<std::string>("ee_frame", "ee_link");
    urdf_path_empty_ = declare_parameter<std::string>("urdf_path_empty", legacy_urdf_path);
    urdf_path_payload_ = declare_parameter<std::string>("urdf_path_payload", urdf_path_empty_);
    task_frame_empty_ = declare_parameter<std::string>("task_frame_empty", legacy_task_frame);
    task_frame_payload_ = declare_parameter<std::string>("task_frame_payload", task_frame_empty_);
    payload_attached_topic_ = declare_parameter<std::string>("payload_attached_topic", "/payload_attached");
    payload_attached_ = declare_parameter<bool>("payload_attached_initial", false);

    ik_hz_ = std::max(1.0, declare_parameter<double>("ik_hz", 500.0));
    dt_ = 1.0 / ik_hz_;

    const auto ik_iters_param = declare_parameter<int64_t>("ik_max_iters", 200);
    ik_max_iters_ = static_cast<int>(std::max<int64_t>(1, ik_iters_param));
    ik_tolerance_ = std::max(1e-6, declare_parameter<double>("ik_tolerance", 1e-3));
    ik_pitch_tolerance_rad_ = std::max(1e-6, declare_parameter<double>("ik_pitch_tolerance_rad", 0.03));
    ik_damping_ = std::max(1e-9, declare_parameter<double>("ik_damping", 1e-3));
    ik_max_step_rad_ = std::max(1e-4, declare_parameter<double>("ik_max_step_rad", 0.20));

    const auto lower_param = declare_parameter<std::vector<double>>(
      "ik_joint_lower_limits",
      std::vector<double>{-kDefaultJointLimitRad, -kDefaultJointLimitRad, -kDefaultJointLimitRad,
        -kDefaultJointLimitRad});
    const auto upper_param = declare_parameter<std::vector<double>>(
      "ik_joint_upper_limits",
      std::vector<double>{kDefaultJointLimitRad, kDefaultJointLimitRad, kDefaultJointLimitRad,
        kDefaultJointLimitRad});
    ik_joint_lower_limits_ = vector_param_to_eigen4(lower_param, -kDefaultJointLimitRad, -kDefaultJointLimitRad);
    ik_joint_upper_limits_ = vector_param_to_eigen4(upper_param, kDefaultJointLimitRad, -kDefaultJointLimitRad);
    for (int i = 0; i < kDof; ++i) {
      if (ik_joint_lower_limits_(i) > ik_joint_upper_limits_(i)) {
        std::swap(ik_joint_lower_limits_(i), ik_joint_upper_limits_(i));
      }
    }

    const auto max_joint_vel_param = declare_parameter<std::vector<double>>(
      "max_joint_velocity", std::vector<double>{1.5, 1.5, 1.5, 1.5});
    const auto max_joint_acc_param = declare_parameter<std::vector<double>>(
      "max_joint_acceleration", std::vector<double>{3.0, 3.0, 3.0, 3.0});
    max_joint_velocity_ = vector_param_to_eigen4(max_joint_vel_param, 1.5, 1e-5);
    max_joint_acceleration_ = vector_param_to_eigen4(max_joint_acc_param, 3.0, 1e-5);

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      std::bind(&CartesianIkMapperNode::on_joint_state, this, std::placeholders::_1));

    cartesian_state_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/planned_cartesian_state", 100,
      std::bind(&CartesianIkMapperNode::on_planned_cartesian_state, this, std::placeholders::_1));

    payload_state_sub_ = create_subscription<std_msgs::msg::Bool>(
      payload_attached_topic_, 10,
      std::bind(&CartesianIkMapperNode::on_payload_attached, this, std::placeholders::_1));

    planned_joint_pub_ = create_publisher<sensor_msgs::msg::JointState>("/planned_joint_state", 100);

#if NMB_HAS_PINOCCHIO
    load_pinocchio_context(empty_context_, "empty", urdf_path_empty_, task_frame_empty_);
    load_pinocchio_context(payload_context_, "payload", urdf_path_payload_, task_frame_payload_);
    log_active_context();
#else
    RCLCPP_WARN(get_logger(), "Pinocchio headers not found. IK mapper disabled.");
#endif

    RCLCPP_INFO(
      get_logger(),
      "cartesian_ik_mapper_node started. input=/planned_cartesian_state output=/planned_joint_state ik_hz=%.1f dt=%.6f payload_topic=%s",
      ik_hz_, dt_, payload_attached_topic_.c_str());
  }

private:
  static constexpr int kDof = 4;
  static constexpr double kDefaultJointLimitRad = 2.0 * M_PI;

  struct CartesianState
  {
    Eigen::Vector4d p = Eigen::Vector4d::Zero();
    Eigen::Vector4d dp = Eigen::Vector4d::Zero();
    Eigen::Vector4d ddp = Eigen::Vector4d::Zero();
  };

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

  static Eigen::Vector4d vector_param_to_eigen4(
    const std::vector<double> & src,
    double fill,
    double min_value)
  {
    Eigen::Vector4d out = Eigen::Vector4d::Constant(fill);
    const size_t n = std::min(static_cast<size_t>(kDof), src.size());
    for (size_t i = 0; i < n; ++i) {
      out(static_cast<Eigen::Index>(i)) = std::max(min_value, src[i]);
    }
    return out;
  }

#if NMB_HAS_PINOCCHIO
  struct ModelContext
  {
    std::string label;
    std::string urdf_path;
    std::string task_frame;
    mutable pinocchio::Model model;
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
      RCLCPP_WARN(get_logger(), "IK context '%s' has empty urdf path.", label.c_str());
      return false;
    }

    try {
      pinocchio::urdf::buildModel(ctx.urdf_path, ctx.model);
      ctx.data = std::make_unique<pinocchio::Data>(ctx.model);
      ctx.frame_id = ctx.model.getFrameId(ctx.task_frame);
      if (ctx.frame_id >= ctx.model.frames.size()) {
        RCLCPP_ERROR(
          get_logger(), "IK context '%s' frame '%s' not found in URDF '%s'.",
          label.c_str(), ctx.task_frame.c_str(), ctx.urdf_path.c_str());
        return false;
      }

      if (ctx.model.nq != kDof || ctx.model.nv != kDof) {
        RCLCPP_ERROR(
          get_logger(),
          "IK context '%s' requires nq=nv=%d, got nq=%d nv=%d.",
          label.c_str(), kDof, ctx.model.nq, ctx.model.nv);
        return false;
      }

      const Eigen::Vector4d q_zero = Eigen::Vector4d::Zero();
      pinocchio::forwardKinematics(ctx.model, *ctx.data, q_zero);
      pinocchio::updateFramePlacements(ctx.model, *ctx.data);
      ctx.pitch_reference_rad =
        raw_pitch_from_rotation(ctx.data->oMf[ctx.frame_id].rotation());
      ctx.loaded = true;

      RCLCPP_INFO(
        get_logger(), "IK context '%s' loaded. urdf=%s task_frame=%s",
        ctx.label.c_str(), ctx.urdf_path.c_str(), ctx.task_frame.c_str());
      return true;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        get_logger(), "Failed to load IK context '%s': %s",
        label.c_str(), e.what());
      return false;
    }
  }

  const ModelContext * active_context() const
  {
    if (payload_attached_ && payload_context_.loaded) {
      return &payload_context_;
    }
    if (empty_context_.loaded) {
      return &empty_context_;
    }
    if (payload_context_.loaded) {
      return &payload_context_;
    }
    return nullptr;
  }

  void log_active_context() const
  {
    const auto * ctx = active_context();
    if (!ctx) {
      RCLCPP_WARN(get_logger(), "No valid IK context loaded.");
      return;
    }
    RCLCPP_INFO(
      get_logger(), "IK active context=%s task_frame=%s payload_attached=%s",
      ctx->label.c_str(), ctx->task_frame.c_str(), payload_attached_ ? "true" : "false");
  }

  bool compute_task_jacobian(const ModelContext & ctx, const Eigen::Vector4d & q, Eigen::Matrix4d & J) const
  {
    pinocchio::forwardKinematics(ctx.model, *ctx.data, q);
    pinocchio::updateFramePlacements(ctx.model, *ctx.data);

    pinocchio::computeJointJacobians(ctx.model, *ctx.data, q);
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

  bool compute_task_bias_acceleration(
    const ModelContext & ctx,
    const Eigen::Vector4d & q,
    const Eigen::Vector4d & qdot,
    Eigen::Vector4d & out_jdot_qdot) const
  {
    const Eigen::Vector4d zero_acc = Eigen::Vector4d::Zero();
    pinocchio::forwardKinematics(ctx.model, *ctx.data, q, qdot, zero_acc);
    pinocchio::updateFramePlacements(ctx.model, *ctx.data);

    const auto bias_world_aligned = pinocchio::getFrameClassicalAcceleration(
      ctx.model, *ctx.data, ctx.frame_id, pinocchio::LOCAL_WORLD_ALIGNED);
    const auto bias_local = pinocchio::getFrameClassicalAcceleration(
      ctx.model, *ctx.data, ctx.frame_id, pinocchio::LOCAL);

    out_jdot_qdot.setZero();
    out_jdot_qdot.head<3>() = bias_world_aligned.linear();
    out_jdot_qdot(3) = bias_local.angular()(0);
    return out_jdot_qdot.allFinite();
  }

  void clamp_to_joint_limits(const ModelContext & ctx, Eigen::Vector4d & q_vec) const
  {
    for (int i = 0; i < kDof; ++i) {
      double lo = ctx.model.lowerPositionLimit(i);
      double hi = ctx.model.upperPositionLimit(i);
      const double lo_cfg = ik_joint_lower_limits_(i);
      const double hi_cfg = ik_joint_upper_limits_(i);

      if (std::isfinite(lo_cfg)) {
        lo = std::isfinite(lo) ? std::max(lo, lo_cfg) : lo_cfg;
      }
      if (std::isfinite(hi_cfg)) {
        hi = std::isfinite(hi) ? std::min(hi, hi_cfg) : hi_cfg;
      }

      if (std::isfinite(lo) && std::isfinite(hi) && lo <= hi) {
        q_vec(i) = std::clamp(static_cast<double>(q_vec(i)), lo, hi);
      } else if (std::isfinite(lo)) {
        q_vec(i) = std::max(static_cast<double>(q_vec(i)), lo);
      } else if (std::isfinite(hi)) {
        q_vec(i) = std::min(static_cast<double>(q_vec(i)), hi);
      }
    }
  }

  bool solve_ik_iterative(
    const ModelContext & ctx,
    const CartesianState & target,
    const Eigen::Vector4d & seed_q,
    Eigen::Vector4d & out_q,
    double & out_pos_err,
    double & out_pitch_err) const
  {
    Eigen::Vector4d q_vec = seed_q;
    clamp_to_joint_limits(ctx, q_vec);

    const Eigen::Vector3d target_pos(target.p(0), target.p(1), target.p(2));
    const double lambda = std::max(1e-9, ik_damping_);

    bool converged = false;
    for (int iter = 0; iter < ik_max_iters_; ++iter) {
      pinocchio::forwardKinematics(ctx.model, *ctx.data, q_vec);
      pinocchio::updateFramePlacements(ctx.model, *ctx.data);

      const auto & frame = ctx.data->oMf[ctx.frame_id];
      const Eigen::Vector3d pos_err_vec = target_pos - frame.translation();
      const double current_pitch =
        wrap_to_pi(raw_pitch_from_rotation(frame.rotation()) - ctx.pitch_reference_rad);
      const double pitch_err = wrap_to_pi(target.p(3) - current_pitch);

      out_pos_err = pos_err_vec.norm();
      out_pitch_err = std::abs(pitch_err);

      if (out_pos_err <= ik_tolerance_ && out_pitch_err <= ik_pitch_tolerance_rad_) {
        converged = true;
        break;
      }

      pinocchio::computeJointJacobians(ctx.model, *ctx.data, q_vec);
      Eigen::Matrix<double, 6, 4> J6_world_aligned = Eigen::Matrix<double, 6, 4>::Zero();
      Eigen::Matrix<double, 6, 4> J6_local = Eigen::Matrix<double, 6, 4>::Zero();
      pinocchio::getFrameJacobian(
        ctx.model, *ctx.data, ctx.frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J6_world_aligned);
      pinocchio::getFrameJacobian(
        ctx.model, *ctx.data, ctx.frame_id, pinocchio::LOCAL, J6_local);

      Eigen::Matrix4d J = Eigen::Matrix4d::Zero();
      J.topRows<3>() = J6_world_aligned.topRows<3>();
      J.row(3) = J6_local.bottomRows<3>().row(0);

      Eigen::Vector4d err;
      err.head<3>() = pos_err_vec;
      err(3) = pitch_err;

      Eigen::Matrix4d H = J.transpose() * J;
      H.diagonal().array() += lambda;
      const Eigen::Vector4d g = J.transpose() * err;
      Eigen::Vector4d dq = H.ldlt().solve(g);
      if (!dq.allFinite()) {
        break;
      }

      const double dq_norm = dq.norm();
      if (dq_norm > ik_max_step_rad_) {
        dq *= (ik_max_step_rad_ / dq_norm);
      }

      q_vec += dq;
      clamp_to_joint_limits(ctx, q_vec);
    }

    out_q = q_vec;
    return converged;
  }
#endif

  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    const size_t npos = std::min(static_cast<size_t>(kDof), msg->position.size());
    for (size_t i = 0; i < npos; ++i) {
      current_q_(static_cast<Eigen::Index>(i)) = msg->position[i];
    }
    has_joint_state_ = true;
  }

  void on_payload_attached(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (payload_attached_ == msg->data) {
      return;
    }
    payload_attached_ = msg->data;
#if NMB_HAS_PINOCCHIO
    if (payload_attached_ && !payload_context_.loaded) {
      RCLCPP_WARN(
        get_logger(),
        "Payload attached requested, but payload IK context is unavailable. Falling back to empty context.");
    }
    log_active_context();
#endif
  }

  static bool parse_cartesian_state(const std_msgs::msg::Float64MultiArray & msg, CartesianState & out)
  {
    if (msg.data.size() < 12U) {
      return false;
    }
    for (int i = 0; i < kDof; ++i) {
      out.p(i) = msg.data[static_cast<size_t>(i)];
      out.dp(i) = msg.data[static_cast<size_t>(4 + i)];
      out.ddp(i) = msg.data[static_cast<size_t>(8 + i)];
    }
    return true;
  }

  void on_planned_cartesian_state(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
#if !NMB_HAS_PINOCCHIO
    (void)msg;
    return;
#else
    const auto * ctx = active_context();
    if (!ctx || !ctx->loaded || !ctx->data) {
      return;
    }

    CartesianState state;
    if (!parse_cartesian_state(*msg, state)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Invalid /planned_cartesian_state payload.");
      return;
    }

    state.p(3) = wrap_to_pi(state.p(3));

    const Eigen::Vector4d seed =
      has_last_solution_ ? last_q_solution_ : (has_joint_state_ ? current_q_ : Eigen::Vector4d::Zero());
    Eigen::Vector4d q_solution = seed;
    double pos_err = std::numeric_limits<double>::infinity();
    double pitch_err = std::numeric_limits<double>::infinity();
    const bool ok = solve_ik_iterative(*ctx, state, seed, q_solution, pos_err, pitch_err);

    if (!ok) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "IK failed in context '%s'. hold last joint command. pos_err=%.5f pitch_err=%.5f",
        ctx->label.c_str(), pos_err, pitch_err);
      return;
    }

    Eigen::Matrix4d J;
    if (!compute_task_jacobian(*ctx, q_solution, J)) {
      return;
    }

    const double lambda = std::max(1e-9, ik_damping_);
    Eigen::Matrix4d H = J.transpose() * J;
    H.diagonal().array() += lambda;
    Eigen::Vector4d qdot = H.ldlt().solve(J.transpose() * state.dp);
    if (!qdot.allFinite()) {
      qdot.setZero();
    }
    for (int i = 0; i < kDof; ++i) {
      qdot(i) = std::clamp(qdot(i), -max_joint_velocity_(i), max_joint_velocity_(i));
    }

    Eigen::Vector4d jdot_qdot = Eigen::Vector4d::Zero();
    if (!compute_task_bias_acceleration(*ctx, q_solution, qdot, jdot_qdot)) {
      jdot_qdot.setZero();
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Failed to compute Jdot*qdot in context '%s'. Falling back to zero bias term.",
        ctx->label.c_str());
    }

    const Eigen::Vector4d corrected_task_ddp = state.ddp - jdot_qdot;
    Eigen::Vector4d qddot = H.ldlt().solve(J.transpose() * corrected_task_ddp);
    if (!qddot.allFinite()) {
      qddot.setZero();
    }

    for (int i = 0; i < kDof; ++i) {
      qddot(i) = std::clamp(qddot(i), -max_joint_acceleration_(i), max_joint_acceleration_(i));
    }

    publish_joint_state(q_solution, qdot, qddot);
    last_q_solution_ = q_solution;
    has_last_solution_ = true;
#endif
  }

  void publish_joint_state(const Eigen::Vector4d & q, const Eigen::Vector4d & dq, const Eigen::Vector4d & ddq)
  {
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = now();
    msg.name = {"j1", "j2", "j3", "j4"};
    msg.position.resize(kDof);
    msg.velocity.resize(kDof);
    msg.effort.resize(kDof);
    for (int i = 0; i < kDof; ++i) {
      msg.position[static_cast<size_t>(i)] = q(i);
      msg.velocity[static_cast<size_t>(i)] = dq(i);
      msg.effort[static_cast<size_t>(i)] = ddq(i);
    }
    planned_joint_pub_->publish(msg);
  }

private:
  double ik_hz_ {500.0};
  double dt_ {1.0 / 500.0};

  std::string urdf_path_empty_;
  std::string urdf_path_payload_;
  std::string task_frame_empty_;
  std::string task_frame_payload_;
  std::string payload_attached_topic_;
  bool payload_attached_ {false};

  int ik_max_iters_ {200};
  double ik_tolerance_ {1e-3};
  double ik_pitch_tolerance_rad_ {0.03};
  double ik_damping_ {1e-3};
  double ik_max_step_rad_ {0.20};

  Eigen::Vector4d ik_joint_lower_limits_ = Eigen::Vector4d::Constant(-kDefaultJointLimitRad);
  Eigen::Vector4d ik_joint_upper_limits_ = Eigen::Vector4d::Constant(kDefaultJointLimitRad);
  Eigen::Vector4d max_joint_velocity_ = Eigen::Vector4d::Constant(1.5);
  Eigen::Vector4d max_joint_acceleration_ = Eigen::Vector4d::Constant(3.0);

  Eigen::Vector4d current_q_ = Eigen::Vector4d::Zero();
  bool has_joint_state_ {false};

  Eigen::Vector4d last_q_solution_ = Eigen::Vector4d::Zero();
  bool has_last_solution_ {false};

#if NMB_HAS_PINOCCHIO
  mutable ModelContext empty_context_;
  mutable ModelContext payload_context_;
#endif

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr cartesian_state_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr payload_state_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr planned_joint_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CartesianIkMapperNode>());
  rclcpp::shutdown();
  return 0;
}
