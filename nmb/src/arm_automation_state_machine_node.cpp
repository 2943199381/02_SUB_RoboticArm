#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class ArmAutomationStateMachineNode : public rclcpp::Node
{
public:
  ArmAutomationStateMachineNode()
  : Node("arm_automation_state_machine_node")
  {
    automation_hz_ = std::max(1.0, declare_parameter<double>("automation_hz", 500.0));
    automation_start_topic_ = declare_parameter<std::string>("automation_start_topic", "/arm_control/start");
    current_pose_topic_ = declare_parameter<std::string>("current_pose_topic", "/current_ee_pose_pitch");
    joint_state_topic_ = declare_parameter<std::string>("joint_state_topic", "/joint_states");
    pd_task_wrench_topic_ = declare_parameter<std::string>("pd_task_wrench_topic", "/arm_pd_task_wrench");
    payload_attached_topic_ = declare_parameter<std::string>("payload_attached_topic", "/payload_attached");
    payload_attached_ = declare_parameter<bool>("payload_attached_initial", false);
    cartesian_path_request_topic_ = declare_parameter<std::string>(
      "cartesian_path_request_topic", "/cartesian_path_request");
    model_transition_cmd_topic_ = declare_parameter<std::string>(
      "model_transition_cmd_topic", "/arm_model_transition_cmd");
    model_blend_status_topic_ = declare_parameter<std::string>(
      "model_blend_status_topic", "/arm_model_blend_status");
    payload_suction_cmd_topic_ = declare_parameter<std::string>(
      "payload_suction_cmd_topic", "/payload_suction_cmd");
    payload_grasped_topic_ = declare_parameter<std::string>(
      "payload_grasped_topic", "/payload_grasped");

    lift_distance_m_ = std::clamp(declare_parameter<double>("lift_distance_m", 0.02), 0.01, 0.03);
    suction_wait_timeout_sec_ = std::max(0.1, declare_parameter<double>("suction_wait_timeout_sec", 3.0));
    pickup_force_check_delay_sec_ = std::max(
      0.0, declare_parameter<double>("pickup_force_check_delay_sec", 0.50));
    retry_wait_sec_ = std::max(0.0, declare_parameter<double>("retry_wait_sec", 1.0));
    model_blend_duration_sec_ = std::max(
      1e-3, declare_parameter<double>("model_blend_duration_sec", 0.50));
    context_switch_settle_sec_ = std::max(
      0.0, declare_parameter<double>("context_switch_settle_sec", 0.05));
    load_settle_sec_ = std::max(0.0, declare_parameter<double>("load_settle_sec", 0.15));
    load_sample_sec_ = std::max(0.05, declare_parameter<double>("load_sample_sec", 0.25));
    motion_timeout_sec_ = std::max(0.20, declare_parameter<double>("motion_timeout_sec", 5.0));
    pose_position_tolerance_m_ = std::max(1e-4, declare_parameter<double>("pose_position_tolerance_m", 0.003));
    pose_pitch_tolerance_rad_ = std::max(1e-4, declare_parameter<double>("pose_pitch_tolerance_rad", 0.03));
    joint_stopped_velocity_tolerance_ = std::max(
      1e-4, declare_parameter<double>("joint_stopped_velocity_tolerance", 0.08));
    expected_payload_mass_kg_ = std::max(
      0.0, declare_parameter<double>("expected_payload_mass_kg", 0.63));
    gravity_mps2_ = std::max(1e-6, declare_parameter<double>("gravity_mps2", 9.81));
    const double legacy_pickup_force_tolerance_ratio = std::clamp(
      declare_parameter<double>("pickup_force_tolerance_ratio", 0.35), 0.01, 1.0);
    const double default_pickup_force_n = expected_payload_force_n();
    pickup_force_min_n_ = std::max(
      0.0,
      declare_parameter<double>(
        "pickup_force_min_n",
        std::max(0.0, default_pickup_force_n * (1.0 - legacy_pickup_force_tolerance_ratio))));
    pickup_force_max_n_ = std::max(
      pickup_force_min_n_,
      declare_parameter<double>(
        "pickup_force_max_n",
        default_pickup_force_n * (1.0 + legacy_pickup_force_tolerance_ratio)));
    release_force_clear_ratio_ = std::clamp(
      declare_parameter<double>("release_force_clear_ratio", 0.35), 0.01, 1.0);
    release_clear_hold_sec_ = std::max(
      0.0, declare_parameter<double>("release_clear_hold_sec", 0.30));

    automation_start_sub_ = create_subscription<std_msgs::msg::Bool>(
      automation_start_topic_, 10,
      std::bind(&ArmAutomationStateMachineNode::on_automation_start, this, std::placeholders::_1));

    current_pose_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      current_pose_topic_, 20,
      std::bind(&ArmAutomationStateMachineNode::on_current_pose, this, std::placeholders::_1));

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic_, 20,
      std::bind(&ArmAutomationStateMachineNode::on_joint_state, this, std::placeholders::_1));

    pd_task_wrench_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      pd_task_wrench_topic_, 20,
      std::bind(&ArmAutomationStateMachineNode::on_pd_task_wrench, this, std::placeholders::_1));

    model_blend_status_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      model_blend_status_topic_, 20,
      std::bind(&ArmAutomationStateMachineNode::on_model_blend_status, this, std::placeholders::_1));

    payload_state_sub_ = create_subscription<std_msgs::msg::Bool>(
      payload_attached_topic_, 20,
      std::bind(&ArmAutomationStateMachineNode::on_payload_attached, this, std::placeholders::_1));
    payload_grasped_sub_ = create_subscription<std_msgs::msg::Bool>(
      payload_grasped_topic_, 20,
      std::bind(&ArmAutomationStateMachineNode::on_payload_grasped, this, std::placeholders::_1));

    cartesian_path_request_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      cartesian_path_request_topic_, 10);
    payload_state_pub_ = create_publisher<std_msgs::msg::Bool>(payload_attached_topic_, 10);
    model_transition_cmd_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      model_transition_cmd_topic_, 10);
    payload_suction_cmd_pub_ = create_publisher<std_msgs::msg::Bool>(payload_suction_cmd_topic_, 10);

    state_enter_time_ = now();
    release_clear_start_time_ = state_enter_time_;

    const auto period = std::chrono::duration<double>(1.0 / automation_hz_);
    state_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&ArmAutomationStateMachineNode::on_state_timer, this));

    RCLCPP_INFO(
      get_logger(),
      "arm_automation_state_machine_node started. start_topic=%s payload_topic=%s transition_cmd=%s blend_status=%s lift=%.3f m",
      automation_start_topic_.c_str(), payload_attached_topic_.c_str(),
      model_transition_cmd_topic_.c_str(), model_blend_status_topic_.c_str(), lift_distance_m_);
  }

private:
  static constexpr int kDof = 4;

  enum class AutomationState
  {
    kIdle,
    kPickupWaitSuction,
    kPickupLifting,
    kPickupVerifying,
    kPickupBlendingToPayload,
    kPickupReturning,
    kPickupRetryWait,
    kPickupRetryApproaching,
    kPlaceBlendingToEmpty,
    kPlaceWaitContextSwitch,
    kPlaceLifting,
    kPlaceVerifyingRelease,
    kPlaceFaultHold,
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

  const char * automation_state_name(AutomationState state) const
  {
    switch (state) {
      case AutomationState::kIdle:
        return "idle";
      case AutomationState::kPickupWaitSuction:
        return "pickup_wait_suction";
      case AutomationState::kPickupLifting:
        return "pickup_lifting";
      case AutomationState::kPickupVerifying:
        return "pickup_verifying";
      case AutomationState::kPickupBlendingToPayload:
        return "pickup_blending_to_payload";
      case AutomationState::kPickupReturning:
        return "pickup_returning";
      case AutomationState::kPickupRetryWait:
        return "pickup_retry_wait";
      case AutomationState::kPickupRetryApproaching:
        return "pickup_retry_approaching";
      case AutomationState::kPlaceBlendingToEmpty:
        return "place_blending_to_empty";
      case AutomationState::kPlaceWaitContextSwitch:
        return "place_wait_context_switch";
      case AutomationState::kPlaceLifting:
        return "place_lifting";
      case AutomationState::kPlaceVerifyingRelease:
        return "place_verifying_release";
      case AutomationState::kPlaceFaultHold:
        return "place_fault_hold";
      default:
        return "unknown";
    }
  }

  void set_automation_state(AutomationState next_state, const std::string & reason)
  {
    automation_state_ = next_state;
    state_enter_time_ = now();
    RCLCPP_INFO(
      get_logger(), "%s -> state=%s", reason.c_str(), automation_state_name(automation_state_));
  }

  double automation_state_elapsed_sec() const
  {
    return (now() - state_enter_time_).seconds();
  }

  double expected_payload_force_n() const
  {
    return expected_payload_mass_kg_ * gravity_mps2_;
  }

  double pickup_force_min_n() const
  {
    return pickup_force_min_n_;
  }

  double pickup_force_max_n() const
  {
    return pickup_force_max_n_;
  }

  double release_force_max_n() const
  {
    return expected_payload_force_n() * release_force_clear_ratio_;
  }

  double current_force_norm_n() const
  {
    return latest_wrench_.head<3>().norm();
  }

  void reset_force_sampling()
  {
    sampled_force_sum_n_ = 0.0;
    sampled_force_count_ = 0;
  }

  void accumulate_force_sample()
  {
    if (!has_valid_wrench_) {
      return;
    }
    sampled_force_sum_n_ += current_force_norm_n();
    ++sampled_force_count_;
  }

  double sampled_force_average_n() const
  {
    if (sampled_force_count_ <= 0) {
      return 0.0;
    }
    return sampled_force_sum_n_ / static_cast<double>(sampled_force_count_);
  }

  bool pose_reached(const Eigen::Vector4d & target_pose) const
  {
    if (!has_current_pose_) {
      return false;
    }

    const double pos_err = (current_pose_.head<3>() - target_pose.head<3>()).norm();
    const double pitch_err = std::abs(wrap_to_pi(current_pose_(3) - target_pose(3)));
    return pos_err <= pose_position_tolerance_m_ &&
           pitch_err <= pose_pitch_tolerance_rad_ &&
           max_joint_speed_abs_ <= joint_stopped_velocity_tolerance_;
  }

  bool blend_reached(double target_alpha) const
  {
    if (!has_blend_status_) {
      return false;
    }
    return !blend_active_ &&
           std::abs(blend_target_alpha_ - target_alpha) <= 1e-3 &&
           std::abs(blend_alpha_ - target_alpha) <= 1e-3;
  }

  Eigen::Vector4d make_lift_target(const Eigen::Vector4d & base_pose) const
  {
    Eigen::Vector4d target = base_pose;
    target(2) += lift_distance_m_;
    return target;
  }

  void request_cartesian_target(const Eigen::Vector4d & target_pose, const std::string & reason)
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(4);
    for (int i = 0; i < kDof; ++i) {
      msg.data[static_cast<size_t>(i)] = target_pose(i);
    }
    cartesian_path_request_pub_->publish(msg);

    motion_target_pose_ = target_pose;
    has_motion_target_ = true;

    RCLCPP_INFO(
      get_logger(),
      "%s: target=[%.4f, %.4f, %.4f, %.4f]",
      reason.c_str(), target_pose(0), target_pose(1), target_pose(2), target_pose(3));
  }

  void request_model_transition(double target_alpha, const std::string & reason)
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.data = {target_alpha, model_blend_duration_sec_};
    model_transition_cmd_pub_->publish(msg);

    RCLCPP_INFO(
      get_logger(), "%s: target_alpha=%.3f duration=%.3f",
      reason.c_str(), target_alpha, model_blend_duration_sec_);
  }

  void publish_payload_state(bool attached)
  {
    payload_attached_ = attached;

    std_msgs::msg::Bool msg;
    msg.data = attached;
    payload_state_pub_->publish(msg);

    RCLCPP_INFO(
      get_logger(), "Published payload state: %s", payload_attached_ ? "attached" : "detached");
  }

  void publish_payload_suction_cmd(bool enable, const std::string & reason)
  {
    std_msgs::msg::Bool msg;
    msg.data = enable;
    payload_suction_cmd_pub_->publish(msg);
    RCLCPP_INFO(
      get_logger(), "%s: suction=%s", reason.c_str(), enable ? "on" : "off");
  }

  void begin_pick_sequence()
  {
    prepare_pose_ = current_pose_;
    has_prepare_pose_ = true;
    request_pickup_lift_from_prepare_pose("Pickup lift request", "Start pickup sequence");
  }

  void request_pickup_lift_from_prepare_pose(
    const std::string & request_reason,
    const std::string & transition_reason)
  {
    if (!has_prepare_pose_) {
      set_automation_state(AutomationState::kIdle, transition_reason + " without prepare pose");
      return;
    }

    request_cartesian_target(make_lift_target(prepare_pose_), request_reason);
    set_automation_state(AutomationState::kPickupLifting, transition_reason);
  }

  void request_pickup_retry_approach()
  {
    if (!has_prepare_pose_) {
      set_automation_state(AutomationState::kIdle, "Retry pickup requested without prepare pose");
      return;
    }

    publish_payload_suction_cmd(true, "Pickup retry suction request");
    request_cartesian_target(prepare_pose_, "Pickup retry grasp pose request");
    set_automation_state(
      AutomationState::kPickupRetryApproaching,
      "Retry pickup by replanning to grasp pose");
  }

  void begin_place_sequence()
  {
    request_model_transition(0.0, "Place detach blend request");
    set_automation_state(AutomationState::kPlaceBlendingToEmpty, "Start place sequence");
  }

  void on_automation_start(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (!msg->data) {
      return;
    }

    if (automation_state_ != AutomationState::kIdle) {
      RCLCPP_WARN(
        get_logger(), "Ignoring automation start while busy in state=%s.",
        automation_state_name(automation_state_));
      return;
    }

    if (!has_current_pose_) {
      RCLCPP_WARN(get_logger(), "Ignoring automation start because current pose is not available yet.");
      return;
    }

    if (payload_attached_) {
      begin_place_sequence();
    } else {
      if (payload_grasped_) {
        begin_pick_sequence();
      } else {
        publish_payload_suction_cmd(true, "Pickup suction request");
        set_automation_state(AutomationState::kPickupWaitSuction, "Waiting for suction grasp");
      }
    }
  }

  void on_current_pose(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < 4U) {
      return;
    }

    for (int i = 0; i < kDof; ++i) {
      current_pose_(i) = msg->data[static_cast<size_t>(i)];
    }
    has_current_pose_ = true;
  }

  void on_joint_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    max_joint_speed_abs_ = 0.0;
    const size_t nvel = std::min(static_cast<size_t>(kDof), msg->velocity.size());
    for (size_t i = 0; i < nvel; ++i) {
      max_joint_speed_abs_ = std::max(max_joint_speed_abs_, std::abs(msg->velocity[i]));
    }
  }

  void on_pd_task_wrench(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < 4U) {
      has_valid_wrench_ = false;
      latest_wrench_.setZero();
      return;
    }

    for (int i = 0; i < kDof; ++i) {
      latest_wrench_(i) = msg->data[static_cast<size_t>(i)];
    }
    has_valid_wrench_ = msg->data.size() >= 5U ? (msg->data[4] > 0.5) : latest_wrench_.allFinite();
    if (!has_valid_wrench_ || !latest_wrench_.allFinite()) {
      latest_wrench_.setZero();
      has_valid_wrench_ = false;
    }
  }

  void on_model_blend_status(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < 2U) {
      return;
    }

    blend_alpha_ = msg->data[0];
    blend_active_ = msg->data[1] > 0.5;
    blend_target_alpha_ = msg->data.size() >= 3U ? msg->data[2] : blend_alpha_;
    has_blend_status_ = true;
  }

  void on_payload_attached(const std_msgs::msg::Bool::SharedPtr msg)
  {
    payload_attached_ = msg->data;
  }

  void on_payload_grasped(const std_msgs::msg::Bool::SharedPtr msg)
  {
    payload_grasped_ = msg->data;
  }

  void on_state_timer()
  {
    switch (automation_state_) {
      case AutomationState::kIdle:
        return;

      case AutomationState::kPickupWaitSuction:
        if (payload_grasped_) {
          begin_pick_sequence();
          return;
        }
        if (automation_state_elapsed_sec() >= std::min(suction_wait_timeout_sec_, pickup_force_check_delay_sec_)) {
          RCLCPP_INFO(
            get_logger(),
            "No pickup grasp feedback observed. Start lift probe and rely on load verification.");
          begin_pick_sequence();
        }
        return;

      case AutomationState::kPickupLifting:
        if (has_motion_target_ ) {
          reset_force_sampling();
          set_automation_state(AutomationState::kPickupVerifying, "Pickup lift reached, start load check");
        } else if (automation_state_elapsed_sec() >= motion_timeout_sec_) {
          RCLCPP_WARN(get_logger(), "Pickup lift timed out. Returning to prepare pose.");
          if (has_prepare_pose_) {
            request_cartesian_target(prepare_pose_, "Pickup return request after lift timeout");
            set_automation_state(AutomationState::kPickupReturning, "Pickup timeout return");
          } else {
            set_automation_state(AutomationState::kIdle, "Pickup timeout without prepare pose");
          }
        }
        return;

      case AutomationState::kPickupVerifying:
        if (automation_state_elapsed_sec() >= load_settle_sec_) {
          accumulate_force_sample();
        }

        if (automation_state_elapsed_sec() < load_settle_sec_ + load_sample_sec_) {
          return;
        }

        if (sampled_force_count_ > 0 &&
          sampled_force_average_n() >= pickup_force_min_n() &&
          sampled_force_average_n() <= pickup_force_max_n())
        {
          RCLCPP_INFO(
            get_logger(),
            "Pickup load detected. avg_force=%.3f N expected=[%.3f, %.3f] N",
            sampled_force_average_n(), pickup_force_min_n(), pickup_force_max_n());
          request_model_transition(1.0, "Pickup attach blend request");
          set_automation_state(AutomationState::kPickupBlendingToPayload, "Pickup verification passed");
          return;
        }

        RCLCPP_WARN(
          get_logger(),
          "Pickup verification failed. avg_force=%.3f N expected=[%.3f, %.3f] N",
          sampled_force_average_n(), pickup_force_min_n(), pickup_force_max_n());
        if (has_prepare_pose_) {
          request_cartesian_target(prepare_pose_, "Pickup return request");
          set_automation_state(AutomationState::kPickupReturning, "Pickup verification failed");
        } else {
          set_automation_state(AutomationState::kPickupRetryWait, "Pickup verification failed without prepare pose");
        }
        return;

      case AutomationState::kPickupBlendingToPayload:
        if (!blend_reached(1.0)) {
          return;
        }
        publish_payload_state(true);
        set_automation_state(AutomationState::kIdle, "Pickup blend complete");
        return;

      case AutomationState::kPickupReturning:
        if (has_prepare_pose_ && pose_reached(prepare_pose_)) {
          set_automation_state(AutomationState::kPickupRetryWait, "Returned to prepare pose");
        } else if (automation_state_elapsed_sec() >= motion_timeout_sec_) {
          RCLCPP_WARN(get_logger(), "Pickup return timed out. Waiting before retry.");
          set_automation_state(AutomationState::kPickupRetryWait, "Pickup return timeout");
        }
        return;

      case AutomationState::kPickupRetryWait:
        if (automation_state_elapsed_sec() < retry_wait_sec_) {
          return;
        }
        request_pickup_retry_approach();
        return;

      case AutomationState::kPickupRetryApproaching:
        if (has_prepare_pose_ && pose_reached(prepare_pose_)) {
          request_pickup_lift_from_prepare_pose(
            "Pickup retry lift request",
            "Retry pickup after grasp replan");
        } else if (automation_state_elapsed_sec() >= motion_timeout_sec_) {
          RCLCPP_WARN(get_logger(), "Pickup retry grasp replan timed out. Waiting before retry.");
          set_automation_state(AutomationState::kPickupRetryWait, "Pickup retry grasp pose timeout");
        }
        return;

      case AutomationState::kPlaceBlendingToEmpty:
        if (!blend_reached(0.0)) {
          return;
        }
        publish_payload_suction_cmd(false, "Place release suction");
        if (payload_attached_) {
          publish_payload_state(false);
        }
        release_clear_active_ = false;
        release_clear_start_time_ = now();
        set_automation_state(AutomationState::kPlaceWaitContextSwitch, "Release requested, waiting for empty context");
        return;

      case AutomationState::kPlaceWaitContextSwitch:
        if (automation_state_elapsed_sec() < context_switch_settle_sec_) {
          return;
        }
        release_clear_active_ = false;
        release_clear_start_time_ = now();
        set_automation_state(AutomationState::kPlaceVerifyingRelease, "Empty context settled, wait for low PD load");
        return;

      case AutomationState::kPlaceLifting:
        if (has_motion_target_ && pose_reached(motion_target_pose_)) {
          set_automation_state(AutomationState::kIdle, "Place retreat complete");
        } else if (automation_state_elapsed_sec() >= motion_timeout_sec_) {
          RCLCPP_WARN(get_logger(), "Place lift timed out. Holding position until load disappears.");
          release_clear_active_ = false;
          release_clear_start_time_ = now();
          set_automation_state(AutomationState::kPlaceFaultHold, "Place lift timeout");
        }
        return;

      case AutomationState::kPlaceVerifyingRelease:
        if (!has_valid_wrench_) {
          return;
        }

        if (current_force_norm_n() <= release_force_max_n()) {
          if (!release_clear_active_) {
            release_clear_active_ = true;
            release_clear_start_time_ = now();
            return;
          }
          if ((now() - release_clear_start_time_).seconds() >= release_clear_hold_sec_) {
            set_automation_state(AutomationState::kIdle, "Place sequence complete");
            return;
          }
        } else {
          release_clear_active_ = false;
        }

        if (automation_state_elapsed_sec() < motion_timeout_sec_) {
          return;
        }

        if (has_valid_wrench_) {
          RCLCPP_INFO(
            get_logger(), "Place waiting for low PD load. force=%.3f N clear_threshold=%.3f N",
            current_force_norm_n(), release_force_max_n());
        }
        return;

      case AutomationState::kPlaceFaultHold:
        if (!has_valid_wrench_) {
          return;
        }

        if (current_force_norm_n() <= release_force_max_n()) {
          if (!release_clear_active_) {
            release_clear_active_ = true;
            release_clear_start_time_ = now();
          } else if ((now() - release_clear_start_time_).seconds() >= release_clear_hold_sec_) {
            RCLCPP_INFO(
              get_logger(), "Observed released payload disappearance. force=%.3f N threshold=%.3f N",
              current_force_norm_n(), release_force_max_n());
            set_automation_state(AutomationState::kIdle, "Place hold finished after payload disappeared");
          }
        } else {
          release_clear_active_ = false;
        }
        return;
    }
  }

private:
  double automation_hz_ {500.0};
  std::string automation_start_topic_;
  std::string current_pose_topic_;
  std::string joint_state_topic_;
  std::string pd_task_wrench_topic_;
  std::string payload_attached_topic_;
  bool payload_attached_ {false};
  std::string payload_suction_cmd_topic_;
  std::string payload_grasped_topic_;
  std::string cartesian_path_request_topic_;
  std::string model_transition_cmd_topic_;
  std::string model_blend_status_topic_;

  double lift_distance_m_ {0.02};
  double suction_wait_timeout_sec_ {3.0};
  double pickup_force_check_delay_sec_ {0.50};
  double retry_wait_sec_ {1.0};
  double model_blend_duration_sec_ {0.50};
  double context_switch_settle_sec_ {0.05};
  double load_settle_sec_ {0.15};
  double load_sample_sec_ {0.25};
  double motion_timeout_sec_ {5.0};
  double pose_position_tolerance_m_ {0.003};
  double pose_pitch_tolerance_rad_ {0.03};
  double joint_stopped_velocity_tolerance_ {0.08};
  double expected_payload_mass_kg_ {0.63};
  double gravity_mps2_ {9.81};
  double pickup_force_min_n_ {0.0};
  double pickup_force_max_n_ {0.0};
  double release_force_clear_ratio_ {0.35};
  double release_clear_hold_sec_ {0.30};

  AutomationState automation_state_ {AutomationState::kIdle};
  rclcpp::Time state_enter_time_;

  Eigen::Vector4d current_pose_ = Eigen::Vector4d::Zero();
  bool has_current_pose_ {false};
  double max_joint_speed_abs_ {0.0};

  Eigen::Vector4d latest_wrench_ = Eigen::Vector4d::Zero();
  bool has_valid_wrench_ {false};
  bool payload_grasped_ {false};

  double blend_alpha_ {0.0};
  bool blend_active_ {false};
  double blend_target_alpha_ {0.0};
  bool has_blend_status_ {false};

  Eigen::Vector4d prepare_pose_ = Eigen::Vector4d::Zero();
  bool has_prepare_pose_ {false};
  Eigen::Vector4d motion_target_pose_ = Eigen::Vector4d::Zero();
  bool has_motion_target_ {false};

  double sampled_force_sum_n_ {0.0};
  int sampled_force_count_ {0};
  bool release_clear_active_ {false};
  rclcpp::Time release_clear_start_time_;

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr automation_start_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr current_pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr pd_task_wrench_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr model_blend_status_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr payload_state_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr payload_grasped_sub_;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cartesian_path_request_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr payload_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr model_transition_cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr payload_suction_cmd_pub_;

  rclcpp::TimerBase::SharedPtr state_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArmAutomationStateMachineNode>());
  rclcpp::shutdown();
  return 0;
}
