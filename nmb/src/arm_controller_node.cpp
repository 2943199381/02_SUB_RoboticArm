#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"

#if __has_include(<pinocchio/algorithm/frames.hpp>) && \
  __has_include(<pinocchio/algorithm/kinematics.hpp>) && \
  __has_include(<pinocchio/algorithm/rnea.hpp>) && \
  __has_include(<pinocchio/parsers/urdf.hpp>)
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#define NMB_HAS_PINOCCHIO 1
#else
#define NMB_HAS_PINOCCHIO 0
#endif

using namespace std::chrono_literals;

class ArmControllerNode : public rclcpp::Node
{
public:
  ArmControllerNode()
  : Node("arm_controller_node")
  {
    control_hz_ = declare_parameter<double>("control_hz", 500.0);
    const auto kp_param = declare_parameter<std::vector<double>>("kp", std::vector<double>{2.0, 2.0, 2.0, 0.0});
    const auto kd_param = declare_parameter<std::vector<double>>("kd", std::vector<double>{1.0, 1.0, 1.0, 0.0});
    kp_ = vector_param_to_eigen4(kp_param, 20.0);
    kd_ = vector_param_to_eigen4(kd_param, 1.0);
    urdf_path_ = declare_parameter<std::string>("urdf_path", "");
    ee_frame_ = declare_parameter<std::string>("ee_frame", "ee_link");

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      std::bind(&ArmControllerNode::on_joint_state, this, std::placeholders::_1));

    planned_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/planned_joint_state", 10,
      std::bind(&ArmControllerNode::on_planned_joint_state, this, std::placeholders::_1));

    torque_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/joint_torque_cmd", 10);
    motor_tx_pub_ = create_publisher<std_msgs::msg::UInt8MultiArray>("/motor_tx_packet", 20);
    current_ee_pose_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/current_ee_pose_pitch", 10);

#if NMB_HAS_PINOCCHIO
    try_load_pinocchio_model();
#else
    RCLCPP_WARN(
      get_logger(),
      "Pinocchio headers not found at compile time. Running without model-based inverse dynamics.");
#endif

    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, control_hz_));
    control_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(period),
      std::bind(&ArmControllerNode::control_loop, this));

    RCLCPP_INFO(get_logger(), "arm_controller_node started. dof=%d, control_hz=%.1f", kDof, control_hz_);
  }

private:
  static constexpr int kDof = 4;

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

  static double raw_pitch_from_rotation(const Eigen::Matrix3d & r)
  {
    // Keep pitch definition identical to trajectory_planner_node.
    return std::atan2(r(2, 1), r(2, 2));
  }

  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    const size_t npos = std::min(static_cast<size_t>(kDof), msg->position.size());
    for (size_t i = 0; i < npos; ++i) {
      q_(static_cast<Eigen::Index>(i)) = msg->position[i];
    }

    const size_t nvel = std::min(static_cast<size_t>(kDof), msg->velocity.size());
    for (size_t i = 0; i < nvel; ++i) {
      dq_(static_cast<Eigen::Index>(i)) = msg->velocity[i];
    }
  }

  void on_planned_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    const size_t npos = std::min(static_cast<size_t>(kDof), msg->position.size());
    for (size_t i = 0; i < npos; ++i) {
      desired_q_(static_cast<Eigen::Index>(i)) = msg->position[i];
    }

    const size_t nvel = std::min(static_cast<size_t>(kDof), msg->velocity.size());
    for (size_t i = 0; i < nvel; ++i) {
      desired_dq_(static_cast<Eigen::Index>(i)) = msg->velocity[i];
    }

    const size_t nacc = std::min(static_cast<size_t>(kDof), msg->effort.size());
    for (size_t i = 0; i < nacc; ++i) {
      desired_ddq_(static_cast<Eigen::Index>(i)) = msg->effort[i];
    }

    has_planned_state_ = true;
  }

  void control_loop()
  {
    if (!has_planned_state_) {
      desired_q_ = q_;
      // desired_q_ .setZero();  
      desired_dq_.setZero();
      desired_ddq_.setZero();
    }

    Eigen::Vector4d pos_err = Eigen::Vector4d::Zero();
    for (int i = 0; i < kDof; ++i) {
      pos_err(i) = wrap_to_pi(desired_q_(i) - q_(i));
    }
    const Eigen::Vector4d vel_err = desired_dq_ - dq_;
    
    Eigen::Vector4d tau = 0.08 * kp_.cwiseProduct(pos_err) + 0.06 * kd_.cwiseProduct(vel_err);
    // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "pos: [%.3f, %.3f, %.3f, %.3f], vel: [%.3f, %.3f, %.3f, %.3f]",
    //   q_(0), q_(1), q_(2), q_(3),
    //   dq_(0), dq_(1), dq_(2), dq_(3));
    // Eigen::Vector4d tau = {0, 0, 0, 0};
#if NMB_HAS_PINOCCHIO
    if (pinocchio_model_loaded_) {
      tau += compute_inverse_dynamics(desired_ddq_);

      publish_current_end_effector_pose();
    }
#endif

    publish_torque(tau);
    publish_motor_packet(tau);
  }

#if NMB_HAS_PINOCCHIO
  void try_load_pinocchio_model()
  {
    if (urdf_path_.empty()) {
      RCLCPP_WARN(get_logger(), "Parameter urdf_path is empty, Pinocchio model disabled.");
      return;
    }
    try {
      pinocchio::urdf::buildModel(urdf_path_, pinocchio_model_);
      pinocchio_data_ = std::make_unique<pinocchio::Data>(pinocchio_model_);
      ee_frame_id_ = pinocchio_model_.getFrameId(ee_frame_);
      if (ee_frame_id_ >= pinocchio_model_.frames.size()) {
        pinocchio_model_loaded_ = false;
        RCLCPP_ERROR(get_logger(), "ee_frame '%s' not found in URDF model.", ee_frame_.c_str());
        return;
      }
      if (pinocchio_model_.nq != kDof || pinocchio_model_.nv != kDof) {
        pinocchio_model_loaded_ = false;
        RCLCPP_ERROR(
          get_logger(),
          "This controller requires nq=nv=dof=4, but got nq=%d nv=%d.",
          pinocchio_model_.nq, pinocchio_model_.nv);
        return;
      }

      const Eigen::Vector4d q_zero = Eigen::Vector4d::Zero();
      pinocchio::forwardKinematics(pinocchio_model_, *pinocchio_data_, q_zero);
      pinocchio::updateFramePlacements(pinocchio_model_, *pinocchio_data_);
      ee_pitch_reference_rad_ = raw_pitch_from_rotation(pinocchio_data_->oMf[ee_frame_id_].rotation());

      pinocchio_model_loaded_ = true;
      RCLCPP_INFO(
        get_logger(), "Pinocchio model loaded from: %s, ee_frame=%s",
        urdf_path_.c_str(), ee_frame_.c_str());
    } catch (const std::exception & e) {
      pinocchio_model_loaded_ = false;
      RCLCPP_ERROR(get_logger(), "Failed to load Pinocchio model: %s", e.what());
    }
  }

  Eigen::Vector4d compute_inverse_dynamics(const Eigen::Vector4d & ddq_des)
  {
    if (!pinocchio_model_loaded_ || !pinocchio_data_) {
      return Eigen::Vector4d::Zero();
    }

    const Eigen::VectorXd tau_full = pinocchio::rnea(pinocchio_model_, *pinocchio_data_, q_, dq_, ddq_des);
    Eigen::Vector4d tau = Eigen::Vector4d::Zero();
    const Eigen::Index n = std::min<Eigen::Index>(tau_full.size(), kDof);
    for (Eigen::Index i = 0; i < n; ++i) {
      tau(i) = tau_full(i);
    }
    return tau;
  }

  void publish_current_end_effector_pose()
  {
    if (!pinocchio_model_loaded_ || !pinocchio_data_) {
      return;
    }

    pinocchio::forwardKinematics(pinocchio_model_, *pinocchio_data_, q_);
    pinocchio::updateFramePlacements(pinocchio_model_, *pinocchio_data_);
    const auto & frame = pinocchio_data_->oMf[ee_frame_id_];

    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(4);
    msg.data[0] = frame.translation().x();
    msg.data[1] = frame.translation().y();
    msg.data[2] = frame.translation().z();
    msg.data[3] = wrap_to_pi(raw_pitch_from_rotation(frame.rotation()) - ee_pitch_reference_rad_);
    current_ee_pose_pub_->publish(msg);
  }

  pinocchio::Model pinocchio_model_;
  std::unique_ptr<pinocchio::Data> pinocchio_data_;
  pinocchio::FrameIndex ee_frame_id_ {0};
  bool pinocchio_model_loaded_ {false};
  double ee_pitch_reference_rad_ {0.0};
#endif

  void publish_torque(const Eigen::Vector4d & tau)
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(kDof);
    for (int i = 0; i < kDof; ++i) {
      msg.data[static_cast<size_t>(i)] = tau(i);
    }
    torque_pub_->publish(msg);
  }

  void publish_motor_packet(const Eigen::Vector4d & tau)
  {
    std_msgs::msg::UInt8MultiArray packet;
    packet.data.reserve(static_cast<size_t>(kDof) * 2);
    for (int i = 0; i < kDof; ++i) {
      const double t = tau(i);
      const int16_t scaled = static_cast<int16_t>(std::round(std::clamp(t, -32.0, 32.0) * 1000.0));
      packet.data.push_back(static_cast<uint8_t>(scaled & 0xFF));
      packet.data.push_back(static_cast<uint8_t>((scaled >> 8) & 0xFF));
    }
    motor_tx_pub_->publish(packet);
  }

private:
  double control_hz_ {500.0};
  Eigen::Vector4d kp_ = Eigen::Vector4d::Zero();
  Eigen::Vector4d kd_ = Eigen::Vector4d::Zero();
  std::string urdf_path_;
  std::string ee_frame_;

  Eigen::Vector4d q_ = Eigen::Vector4d::Zero();
  Eigen::Vector4d dq_ = Eigen::Vector4d::Zero();

  Eigen::Vector4d desired_q_ = Eigen::Vector4d::Zero();
  Eigen::Vector4d desired_dq_ = Eigen::Vector4d::Zero();
  Eigen::Vector4d desired_ddq_ = Eigen::Vector4d::Zero();
  bool has_planned_state_ {false};

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr planned_state_sub_;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr torque_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr motor_tx_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr current_ee_pose_pub_;

  rclcpp::TimerBase::SharedPtr control_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArmControllerNode>());
  rclcpp::shutdown();
  return 0;
}
