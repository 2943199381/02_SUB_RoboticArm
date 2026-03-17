#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "visualization_msgs/msg/marker.hpp"

class TrajectoryBsplineJerkPlannerNode : public rclcpp::Node
{
public:
  TrajectoryBsplineJerkPlannerNode()
  : Node("trajectory_bspline_jerk_planner_node")
  {
    planner_hz_ = std::max(1.0, declare_parameter<double>("planner_hz", 500.0));
    path_frame_ = declare_parameter<std::string>("path_frame", "base_link");
    path_marker_topic_ = declare_parameter<std::string>("path_marker_topic", "/planned_cartesian_curve");
    payload_attached_topic_ = declare_parameter<std::string>("payload_attached_topic", "/payload_attached");
    payload_attached_ = declare_parameter<bool>("payload_attached_initial", false);

    const double legacy_max_cartesian_speed = std::max(
      1e-5, declare_parameter<double>("max_cartesian_speed", 0.3));
    const double legacy_max_cartesian_acceleration_accel = std::max(
      1e-5, declare_parameter<double>("max_cartesian_acceleration_accel", 0.3));
    const double legacy_max_cartesian_acceleration_decel = std::max(
      1e-5, declare_parameter<double>("max_cartesian_acceleration_decel", 0.18));
    const double legacy_max_cartesian_jerk_accel = std::max(
      1e-5, declare_parameter<double>("max_cartesian_jerk_accel", 5.0));
    const double legacy_max_cartesian_jerk_decel = std::max(
      1e-5, declare_parameter<double>("max_cartesian_jerk_decel", 2.5));
    const double legacy_max_pitch_speed = std::max(
      1e-5, declare_parameter<double>("max_pitch_speed", legacy_max_cartesian_speed));
    const double legacy_max_pitch_acceleration_accel = std::max(
      1e-5, declare_parameter<double>(
        "max_pitch_acceleration_accel", legacy_max_cartesian_acceleration_accel));
    const double legacy_max_pitch_acceleration_decel = std::max(
      1e-5, declare_parameter<double>(
        "max_pitch_acceleration_decel", legacy_max_cartesian_acceleration_decel));
    const double legacy_max_pitch_jerk_accel = std::max(
      1e-5, declare_parameter<double>("max_pitch_jerk_accel", legacy_max_cartesian_jerk_accel));
    const double legacy_max_pitch_jerk_decel = std::max(
      1e-5, declare_parameter<double>("max_pitch_jerk_decel", legacy_max_cartesian_jerk_decel));

    max_cartesian_speed_empty_ = std::max(
      1e-5, declare_parameter<double>("max_cartesian_speed_empty", legacy_max_cartesian_speed));
    max_cartesian_speed_payload_ = std::max(
      1e-5, declare_parameter<double>("max_cartesian_speed_payload", max_cartesian_speed_empty_));
    max_cartesian_acceleration_accel_empty_ = std::max(
      1e-5, declare_parameter<double>(
        "max_cartesian_acceleration_accel_empty", legacy_max_cartesian_acceleration_accel));
    max_cartesian_acceleration_accel_payload_ = std::max(
      1e-5, declare_parameter<double>(
        "max_cartesian_acceleration_accel_payload", max_cartesian_acceleration_accel_empty_));
    max_cartesian_acceleration_decel_empty_ = std::max(
      1e-5, declare_parameter<double>(
        "max_cartesian_acceleration_decel_empty", legacy_max_cartesian_acceleration_decel));
    max_cartesian_acceleration_decel_payload_ = std::max(
      1e-5, declare_parameter<double>(
        "max_cartesian_acceleration_decel_payload", max_cartesian_acceleration_decel_empty_));
    max_cartesian_jerk_accel_empty_ = std::max(
      1e-5, declare_parameter<double>("max_cartesian_jerk_accel_empty", legacy_max_cartesian_jerk_accel));
    max_cartesian_jerk_accel_payload_ = std::max(
      1e-5, declare_parameter<double>("max_cartesian_jerk_accel_payload", max_cartesian_jerk_accel_empty_));
    max_cartesian_jerk_decel_empty_ = std::max(
      1e-5, declare_parameter<double>("max_cartesian_jerk_decel_empty", legacy_max_cartesian_jerk_decel));
    max_cartesian_jerk_decel_payload_ = std::max(
      1e-5, declare_parameter<double>("max_cartesian_jerk_decel_payload", max_cartesian_jerk_decel_empty_));

    max_pitch_speed_empty_ = std::max(
      1e-5, declare_parameter<double>("max_pitch_speed_empty", legacy_max_pitch_speed));
    max_pitch_speed_payload_ = std::max(
      1e-5, declare_parameter<double>("max_pitch_speed_payload", max_pitch_speed_empty_));
    max_pitch_acceleration_accel_empty_ = std::max(
      1e-5, declare_parameter<double>(
        "max_pitch_acceleration_accel_empty", legacy_max_pitch_acceleration_accel));
    max_pitch_acceleration_accel_payload_ = std::max(
      1e-5, declare_parameter<double>(
        "max_pitch_acceleration_accel_payload", max_pitch_acceleration_accel_empty_));
    max_pitch_acceleration_decel_empty_ = std::max(
      1e-5, declare_parameter<double>(
        "max_pitch_acceleration_decel_empty", legacy_max_pitch_acceleration_decel));
    max_pitch_acceleration_decel_payload_ = std::max(
      1e-5, declare_parameter<double>(
        "max_pitch_acceleration_decel_payload", max_pitch_acceleration_decel_empty_));
    max_pitch_jerk_accel_empty_ = std::max(
      1e-5, declare_parameter<double>("max_pitch_jerk_accel_empty", legacy_max_pitch_jerk_accel));
    max_pitch_jerk_accel_payload_ = std::max(
      1e-5, declare_parameter<double>("max_pitch_jerk_accel_payload", max_pitch_jerk_accel_empty_));
    max_pitch_jerk_decel_empty_ = std::max(
      1e-5, declare_parameter<double>("max_pitch_jerk_decel_empty", legacy_max_pitch_jerk_decel));
    max_pitch_jerk_decel_payload_ = std::max(
      1e-5, declare_parameter<double>("max_pitch_jerk_decel_payload", max_pitch_jerk_decel_empty_));
    derivative_step_ = std::clamp(declare_parameter<double>("derivative_step", 1e-3), 1e-5, 0.1);

    const auto max_steps_param = declare_parameter<int64_t>("max_planning_steps", 60000);
    max_planning_steps_ = static_cast<int>(std::max<int64_t>(500, max_steps_param));

    // Hard-coded topics by request.
    current_ee_pose_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/current_ee_pose_pitch", 10,
      std::bind(&TrajectoryBsplineJerkPlannerNode::on_current_ee_pose, this, std::placeholders::_1));

    payload_state_sub_ = create_subscription<std_msgs::msg::Bool>(
      payload_attached_topic_, 10,
      std::bind(&TrajectoryBsplineJerkPlannerNode::on_payload_attached, this, std::placeholders::_1));

    cartesian_path_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/cartesian_path_request", 10,
      std::bind(&TrajectoryBsplineJerkPlannerNode::on_cartesian_path_request, this, std::placeholders::_1));

    planned_cartesian_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/planned_cartesian_state", 10);
    path_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(path_marker_topic_, 1);

    plan_dt_ = 1.0 / planner_hz_;
    const auto period = std::chrono::duration<double>(plan_dt_);
    planner_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&TrajectoryBsplineJerkPlannerNode::planner_loop, this));

    RCLCPP_INFO(
      get_logger(),
      "trajectory_bspline_jerk_planner_node started. start=/current_ee_pose_pitch request=/cartesian_path_request output=/planned_cartesian_state hz=%.1f dt=%.6f payload_topic=%s xyz_speed_empty=%.3f xyz_speed_payload=%.3f pitch_speed_empty=%.3f pitch_speed_payload=%.3f",
      planner_hz_, plan_dt_, payload_attached_topic_.c_str(),
      max_cartesian_speed_empty_, max_cartesian_speed_payload_,
      max_pitch_speed_empty_, max_pitch_speed_payload_);
  }

private:
  static constexpr int kDim = 4;
  static constexpr int kDegree = 3;

  enum class MotionPhase
  {
    kAccelerating,
    kDecelerating,
  };

  struct CartesianSample
  {
    Eigen::Vector4d p = Eigen::Vector4d::Zero();
    Eigen::Vector4d dp = Eigen::Vector4d::Zero();
    Eigen::Vector4d ddp = Eigen::Vector4d::Zero();
  };

  struct PathDerivatives
  {
    Eigen::Vector4d p = Eigen::Vector4d::Zero();
    Eigen::Vector4d dp_du = Eigen::Vector4d::Zero();
    Eigen::Vector4d ddp_du2 = Eigen::Vector4d::Zero();
  };

  struct MotionLimits
  {
    double max_cartesian_speed {0.3};
    double max_cartesian_acceleration_accel {0.3};
    double max_cartesian_acceleration_decel {0.18};
    double max_cartesian_jerk_accel {5.0};
    double max_cartesian_jerk_decel {2.5};
    double max_pitch_speed {0.3};
    double max_pitch_acceleration_accel {0.3};
    double max_pitch_acceleration_decel {0.18};
    double max_pitch_jerk_accel {5.0};
    double max_pitch_jerk_decel {2.5};
  };

  class CubicBSplineCurve
  {
  public:
    CubicBSplineCurve(const std::vector<Eigen::Vector4d> & control_points, const std::vector<double> & knots)
    : control_points_(control_points), knots_(knots)
    {}

    Eigen::Vector4d evaluate(double u) const
    {
      if (control_points_.empty()) {
        return Eigen::Vector4d::Zero();
      }

      const double u_clamped = std::clamp(u, 0.0, 1.0);
      if (u_clamped >= 1.0) {
        return control_points_.back();
      }

      const int span = find_span(u_clamped);
      std::array<Eigen::Vector4d, kDegree + 1> d;
      for (int j = 0; j <= kDegree; ++j) {
        d[static_cast<size_t>(j)] = control_points_[static_cast<size_t>(span - kDegree + j)];
      }

      for (int r = 1; r <= kDegree; ++r) {
        for (int j = kDegree; j >= r; --j) {
          const int i = span - kDegree + j;
          const double denom = knots_[static_cast<size_t>(i + kDegree - r + 1)] - knots_[static_cast<size_t>(i)];
          double alpha = 0.0;
          if (std::abs(denom) > 1e-12) {
            alpha = (u_clamped - knots_[static_cast<size_t>(i)]) / denom;
          }
          d[static_cast<size_t>(j)] = (1.0 - alpha) * d[static_cast<size_t>(j - 1)] + alpha * d[static_cast<size_t>(j)];
        }
      }

      return d[static_cast<size_t>(kDegree)];
    }

  private:
    int find_span(double u) const
    {
      const int n = static_cast<int>(control_points_.size()) - 1;
      if (u >= knots_[static_cast<size_t>(n + 1)]) {
        return n;
      }

      int low = kDegree;
      int high = n + 1;
      int mid = (low + high) / 2;
      while (u < knots_[static_cast<size_t>(mid)] || u >= knots_[static_cast<size_t>(mid + 1)]) {
        if (u < knots_[static_cast<size_t>(mid)]) {
          high = mid;
        } else {
          low = mid;
        }
        mid = (low + high) / 2;
      }
      return mid;
    }

    std::vector<Eigen::Vector4d> control_points_;
    std::vector<double> knots_;
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

  static void unwrap_pitch_sequence(std::vector<Eigen::Vector4d> & anchors)
  {
    if (anchors.empty()) {
      return;
    }
    for (size_t i = 1; i < anchors.size(); ++i) {
      anchors[i](3) = anchors[i - 1](3) + wrap_to_pi(anchors[i](3) - anchors[i - 1](3));
    }
  }

  MotionLimits current_motion_limits() const
  {
    MotionLimits limits;
    if (payload_attached_) {
      limits.max_cartesian_speed = max_cartesian_speed_payload_;
      limits.max_cartesian_acceleration_accel = max_cartesian_acceleration_accel_payload_;
      limits.max_cartesian_acceleration_decel = max_cartesian_acceleration_decel_payload_;
      limits.max_cartesian_jerk_accel = max_cartesian_jerk_accel_payload_;
      limits.max_cartesian_jerk_decel = max_cartesian_jerk_decel_payload_;
      limits.max_pitch_speed = max_pitch_speed_payload_;
      limits.max_pitch_acceleration_accel = max_pitch_acceleration_accel_payload_;
      limits.max_pitch_acceleration_decel = max_pitch_acceleration_decel_payload_;
      limits.max_pitch_jerk_accel = max_pitch_jerk_accel_payload_;
      limits.max_pitch_jerk_decel = max_pitch_jerk_decel_payload_;
      return limits;
    }

    limits.max_cartesian_speed = max_cartesian_speed_empty_;
    limits.max_cartesian_acceleration_accel = max_cartesian_acceleration_accel_empty_;
    limits.max_cartesian_acceleration_decel = max_cartesian_acceleration_decel_empty_;
    limits.max_cartesian_jerk_accel = max_cartesian_jerk_accel_empty_;
    limits.max_cartesian_jerk_decel = max_cartesian_jerk_decel_empty_;
    limits.max_pitch_speed = max_pitch_speed_empty_;
    limits.max_pitch_acceleration_accel = max_pitch_acceleration_accel_empty_;
    limits.max_pitch_acceleration_decel = max_pitch_acceleration_decel_empty_;
    limits.max_pitch_jerk_accel = max_pitch_jerk_accel_empty_;
    limits.max_pitch_jerk_decel = max_pitch_jerk_decel_empty_;
    return limits;
  }

  static double ratio_limit_or_inf(double limit, double magnitude)
  {
    if (magnitude <= 1e-8) {
      return std::numeric_limits<double>::infinity();
    }
    return limit / magnitude;
  }

  static double positive_limit_or_default(double value)
  {
    if (!std::isfinite(value)) {
      return 1e6;
    }
    return std::max(1e-5, value);
  }

  static std::vector<double> make_open_uniform_knots(size_t control_point_count)
  {
    const int n = static_cast<int>(control_point_count) - 1;
    const int m = n + kDegree + 1;
    std::vector<double> knots(static_cast<size_t>(m + 1), 0.0);
    for (int i = 0; i <= m; ++i) {
      if (i <= kDegree) {
        knots[static_cast<size_t>(i)] = 0.0;
      } else if (i >= m - kDegree) {
        knots[static_cast<size_t>(i)] = 1.0;
      } else {
        const double denom = static_cast<double>(m - 2 * kDegree);
        knots[static_cast<size_t>(i)] = static_cast<double>(i - kDegree) / denom;
      }
    }
    return knots;
  }

  static int find_span_on_knots(int n, double u, const std::vector<double> & knots)
  {
    if (u >= knots[static_cast<size_t>(n + 1)]) {
      return n;
    }

    int low = kDegree;
    int high = n + 1;
    int mid = (low + high) / 2;
    while (u < knots[static_cast<size_t>(mid)] || u >= knots[static_cast<size_t>(mid + 1)]) {
      if (u < knots[static_cast<size_t>(mid)]) {
        high = mid;
      } else {
        low = mid;
      }
      mid = (low + high) / 2;
    }
    return mid;
  }

  static std::array<double, kDegree + 1> basis_functions(
    int span,
    double u,
    const std::vector<double> & knots)
  {
    std::array<double, kDegree + 1> nvals {};
    std::array<double, kDegree + 1> left {};
    std::array<double, kDegree + 1> right {};
    nvals[0] = 1.0;

    for (int j = 1; j <= kDegree; ++j) {
      left[static_cast<size_t>(j)] = u - knots[static_cast<size_t>(span + 1 - j)];
      right[static_cast<size_t>(j)] = knots[static_cast<size_t>(span + j)] - u;
      double saved = 0.0;
      for (int r = 0; r < j; ++r) {
        const double denom = right[static_cast<size_t>(r + 1)] + left[static_cast<size_t>(j - r)];
        const double temp = std::abs(denom) > 1e-12 ? nvals[static_cast<size_t>(r)] / denom : 0.0;
        nvals[static_cast<size_t>(r)] = saved + right[static_cast<size_t>(r + 1)] * temp;
        saved = left[static_cast<size_t>(j - r)] * temp;
      }
      nvals[static_cast<size_t>(j)] = saved;
    }

    return nvals;
  }

  static void fill_basis_row(double u, const std::vector<double> & knots, Eigen::VectorXd & row)
  {
    const int n = static_cast<int>(row.size()) - 1;
    row.setZero();

    if (u <= 0.0) {
      row(0) = 1.0;
      return;
    }
    if (u >= 1.0) {
      row(n) = 1.0;
      return;
    }

    const int span = find_span_on_knots(n, u, knots);
    const auto nvals = basis_functions(span, u, knots);
    for (int j = 0; j <= kDegree; ++j) {
      const int col = span - kDegree + j;
      if (col >= 0 && col <= n) {
        row(col) = nvals[static_cast<size_t>(j)];
      }
    }
  }

  static std::vector<double> build_chord_parameterization(const std::vector<Eigen::Vector4d> & anchors)
  {
    const size_t n = anchors.size();
    std::vector<double> u(n, 0.0);
    if (n <= 1U) {
      return u;
    }

    std::vector<double> seg(n - 1U, 0.0);
    double total = 0.0;
    for (size_t i = 1; i < n; ++i) {
      seg[i - 1U] = (anchors[i] - anchors[i - 1U]).norm();
      total += seg[i - 1U];
    }

    if (total <= 1e-10) {
      for (size_t i = 0; i < n; ++i) {
        u[i] = static_cast<double>(i) / static_cast<double>(n - 1U);
      }
      return u;
    }

    double acc = 0.0;
    for (size_t i = 1; i < n; ++i) {
      acc += seg[i - 1U];
      u[i] = acc / total;
    }

    constexpr double kMinGap = 1e-5;
    for (size_t i = 1; i < n; ++i) {
      if (u[i] <= u[i - 1U] + kMinGap) {
        u[i] = u[i - 1U] + kMinGap;
      }
    }

    const double last = u.back();
    if (last > 1e-12) {
      for (double & ui : u) {
        ui /= last;
      }
    }
    u.front() = 0.0;
    u.back() = 1.0;
    return u;
  }

  bool build_interpolating_control_points(
    const std::vector<Eigen::Vector4d> & anchors,
    std::vector<Eigen::Vector4d> & control_points,
    std::vector<double> & knots) const
  {
    control_points.clear();
    knots.clear();
    if (anchors.size() < 2U) {
      return false;
    }

    if (anchors.size() == 2U) {
      const Eigen::Vector4d d = anchors[1] - anchors[0];
      control_points = {
        anchors[0],
        anchors[0] + d / 3.0,
        anchors[0] + 2.0 * d / 3.0,
        anchors[1]
      };
      knots = make_open_uniform_knots(control_points.size());
      return true;
    }

    const size_t num_data = anchors.size();
    const size_t num_ctrl = num_data + 2U;
    knots = make_open_uniform_knots(num_ctrl);
    const std::vector<double> u_data = build_chord_parameterization(anchors);

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(
      static_cast<Eigen::Index>(num_ctrl), static_cast<Eigen::Index>(num_ctrl));
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(num_ctrl), kDim);

    for (size_t i = 0; i < num_data; ++i) {
      Eigen::VectorXd row(static_cast<Eigen::Index>(num_ctrl));
      fill_basis_row(u_data[i], knots, row);
      A.row(static_cast<Eigen::Index>(i)) = row.transpose();
      for (int j = 0; j < kDim; ++j) {
        B(static_cast<Eigen::Index>(i), j) = anchors[i](j);
      }
    }

    A(static_cast<Eigen::Index>(num_data), 0) = 1.0;
    A(static_cast<Eigen::Index>(num_data), 1) = -2.0;
    A(static_cast<Eigen::Index>(num_data), 2) = 1.0;
    A(static_cast<Eigen::Index>(num_data + 1U), static_cast<Eigen::Index>(num_ctrl - 3U)) = 1.0;
    A(static_cast<Eigen::Index>(num_data + 1U), static_cast<Eigen::Index>(num_ctrl - 2U)) = -2.0;
    A(static_cast<Eigen::Index>(num_data + 1U), static_cast<Eigen::Index>(num_ctrl - 1U)) = 1.0;

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    if (qr.rank() < static_cast<Eigen::Index>(num_ctrl)) {
      RCLCPP_WARN(get_logger(), "B-spline interpolation matrix rank-deficient.");
      return false;
    }

    const Eigen::MatrixXd X = qr.solve(B);
    if (!X.allFinite()) {
      RCLCPP_WARN(get_logger(), "B-spline interpolation solve failed.");
      return false;
    }

    control_points.resize(num_ctrl, Eigen::Vector4d::Zero());
    for (size_t i = 0; i < num_ctrl; ++i) {
      for (int j = 0; j < kDim; ++j) {
        control_points[i](j) = X(static_cast<Eigen::Index>(i), j);
      }
    }
    return true;
  }

  PathDerivatives evaluate_path(const CubicBSplineCurve & curve, double u) const
  {
    PathDerivatives out;
    const double clamped_u = std::clamp(u, 0.0, 1.0);
    out.p = curve.evaluate(clamped_u);

    const double h = derivative_step_;
    if (clamped_u <= h) {
      const Eigen::Vector4d q0 = curve.evaluate(clamped_u);
      const Eigen::Vector4d q1 = curve.evaluate(std::min(1.0, clamped_u + h));
      const Eigen::Vector4d q2 = curve.evaluate(std::min(1.0, clamped_u + 2.0 * h));
      out.dp_du = (q1 - q0) / h;
      out.ddp_du2 = (q2 - 2.0 * q1 + q0) / (h * h);
      return out;
    }

    if (clamped_u >= 1.0 - h) {
      const Eigen::Vector4d q0 = curve.evaluate(clamped_u);
      const Eigen::Vector4d q1 = curve.evaluate(std::max(0.0, clamped_u - h));
      const Eigen::Vector4d q2 = curve.evaluate(std::max(0.0, clamped_u - 2.0 * h));
      out.dp_du = (q0 - q1) / h;
      out.ddp_du2 = (q0 - 2.0 * q1 + q2) / (h * h);
      return out;
    }

    const Eigen::Vector4d q_minus = curve.evaluate(clamped_u - h);
    const Eigen::Vector4d q_plus = curve.evaluate(clamped_u + h);
    out.dp_du = (q_plus - q_minus) / (2.0 * h);
    out.ddp_du2 = (q_plus - 2.0 * out.p + q_minus) / (h * h);
    return out;
  }

  double compute_path_velocity_limit(const Eigen::Vector4d & dpose_du, const MotionLimits & limits) const
  {
    const double linear_tangent_norm = dpose_du.head<3>().norm();
    const double pitch_tangent_abs = std::abs(dpose_du(3));
    const double linear_limit = ratio_limit_or_inf(limits.max_cartesian_speed, linear_tangent_norm);
    const double pitch_limit = ratio_limit_or_inf(limits.max_pitch_speed, pitch_tangent_abs);
    return positive_limit_or_default(std::min(linear_limit, pitch_limit));
  }

  static double max_cartesian_acceleration(const MotionLimits & limits, MotionPhase phase)
  {
    return phase == MotionPhase::kDecelerating ?
      limits.max_cartesian_acceleration_decel : limits.max_cartesian_acceleration_accel;
  }

  static double max_pitch_acceleration(const MotionLimits & limits, MotionPhase phase)
  {
    return phase == MotionPhase::kDecelerating ?
      limits.max_pitch_acceleration_decel : limits.max_pitch_acceleration_accel;
  }

  static double max_cartesian_jerk(const MotionLimits & limits, MotionPhase phase)
  {
    return phase == MotionPhase::kDecelerating ?
      limits.max_cartesian_jerk_decel : limits.max_cartesian_jerk_accel;
  }

  static double max_pitch_jerk(const MotionLimits & limits, MotionPhase phase)
  {
    return phase == MotionPhase::kDecelerating ?
      limits.max_pitch_jerk_decel : limits.max_pitch_jerk_accel;
  }

  double compute_path_acceleration_limit(
    const Eigen::Vector4d & dpose_du,
    const Eigen::Vector4d & ddpose_du2,
    double u_dot,
    MotionPhase phase,
    const MotionLimits & limits) const
  {
    const double linear_tangent_norm = dpose_du.head<3>().norm();
    const double pitch_tangent_abs = std::abs(dpose_du(3));
    const double linear_curvature_norm = ddpose_du2.head<3>().norm() * u_dot * u_dot;
    const double pitch_curvature_abs = std::abs(ddpose_du2(3)) * u_dot * u_dot;
    const double linear_avail = ratio_limit_or_inf(
      std::max(1e-5, max_cartesian_acceleration(limits, phase) - linear_curvature_norm),
      linear_tangent_norm);
    const double pitch_avail = ratio_limit_or_inf(
      std::max(1e-5, max_pitch_acceleration(limits, phase) - pitch_curvature_abs),
      pitch_tangent_abs);
    return positive_limit_or_default(std::min(linear_avail, pitch_avail));
  }

  double compute_path_jerk_limit(
    const Eigen::Vector4d & dpose_du,
    MotionPhase phase,
    const MotionLimits & limits) const
  {
    const double linear_tangent_norm = dpose_du.head<3>().norm();
    const double pitch_tangent_abs = std::abs(dpose_du(3));
    const double linear_limit = ratio_limit_or_inf(
      max_cartesian_jerk(limits, phase), linear_tangent_norm);
    const double pitch_limit = ratio_limit_or_inf(
      max_pitch_jerk(limits, phase), pitch_tangent_abs);
    return positive_limit_or_default(std::min(linear_limit, pitch_limit));
  }

  double estimate_stop_distance(
    double u_dot,
    double u_ddot,
    double a_limit,
    double j_limit,
    double dt) const
  {
    double v = std::max(0.0, u_dot);
    double a = u_ddot;
    double dist = 0.0;
    constexpr int kMaxStopSimSteps = 20000;
    for (int i = 0; i < kMaxStopSimSteps; ++i) {
      if (v <= 1e-4 && std::abs(a) <= 1e-3) {
        break;
      }
      const double delta_a = std::clamp(-a_limit - a, -j_limit * dt, j_limit * dt);
      a += delta_a;
      a = std::clamp(a, -a_limit, a_limit);
      v = std::max(0.0, v + a * dt);
      dist += v * dt;
    }
    return dist;
  }

  bool build_cartesian_samples(
    const std::vector<Eigen::Vector4d> & anchors,
    const MotionLimits & limits,
    std::vector<CartesianSample> & samples_out) const
  {
    constexpr double kCompletionUTolerance = 1e-4;
    constexpr double kCompletionSpeedTolerance = 5e-3;
    std::vector<Eigen::Vector4d> control_points;
    std::vector<double> knots;
    if (!build_interpolating_control_points(anchors, control_points, knots)) {
      return false;
    }

    if (control_points.size() < static_cast<size_t>(kDegree + 1)) {
      return false;
    }

    const CubicBSplineCurve curve(control_points, knots);
    samples_out.clear();
    samples_out.reserve(static_cast<size_t>(max_planning_steps_) + 2U);

    CartesianSample first;
    first.p = anchors.front();
    first.dp.setZero();
    first.ddp.setZero();
    samples_out.push_back(first);

    double u = 0.0;
    double u_dot = 0.0;
    double u_ddot = 0.0;

    for (int step = 0; step < max_planning_steps_; ++step) {
      const PathDerivatives deriv = evaluate_path(curve, u);
      const double v_limit = compute_path_velocity_limit(deriv.dp_du, limits);
      const double a_accel_limit = compute_path_acceleration_limit(
        deriv.dp_du, deriv.ddp_du2, u_dot, MotionPhase::kAccelerating, limits);
      const double a_decel_limit = compute_path_acceleration_limit(
        deriv.dp_du, deriv.ddp_du2, u_dot, MotionPhase::kDecelerating, limits);
      const double j_accel_limit = compute_path_jerk_limit(
        deriv.dp_du, MotionPhase::kAccelerating, limits);
      const double j_decel_limit = compute_path_jerk_limit(
        deriv.dp_du, MotionPhase::kDecelerating, limits);

      const double remaining = 1.0 - u;
      const double stop_distance = estimate_stop_distance(
        u_dot, u_ddot, a_decel_limit, j_decel_limit, plan_dt_);

      double target_u_ddot = 0.0;
      double applied_j_limit = j_accel_limit;
      if (remaining <= stop_distance + 1e-4) {
        target_u_ddot = -a_decel_limit;
        applied_j_limit = j_decel_limit;
      } else if (u_dot < v_limit - 1e-4) {
        target_u_ddot = a_accel_limit;
      } else if (u_dot > v_limit + 1e-4) {
        target_u_ddot = -a_decel_limit;
        applied_j_limit = j_decel_limit;
      }

      const double delta_acc = std::clamp(
        target_u_ddot - u_ddot,
        -applied_j_limit * plan_dt_,
        applied_j_limit * plan_dt_);
      u_ddot += delta_acc;
      u_ddot = std::clamp(u_ddot, -a_decel_limit, a_accel_limit);

      u_dot += u_ddot * plan_dt_;
      if (u_dot < 0.0) {
        u_dot = 0.0;
        u_ddot = std::max(0.0, u_ddot);
      }
      if (u_dot > v_limit) {
        u_dot = v_limit;
        u_ddot = std::min(0.0, u_ddot);
      }

      u += u_dot * plan_dt_;
      if (u >= 1.0) {
        u = 1.0;
        u_dot = 0.0;
        u_ddot = 0.0;
      } else if ((1.0 - u) <= kCompletionUTolerance && u_dot <= kCompletionSpeedTolerance) {
        u = 1.0;
        u_dot = 0.0;
        u_ddot = 0.0;
      }

      const PathDerivatives now = evaluate_path(curve, u);
      CartesianSample s;
      s.p = now.p;
      s.dp = now.dp_du * u_dot;
      s.ddp = now.ddp_du2 * (u_dot * u_dot) + now.dp_du * u_ddot;
      samples_out.push_back(s);

      if (u >= 1.0 - kCompletionUTolerance) {
        break;
      }
    }

    if (u < 1.0 - kCompletionUTolerance) {
      RCLCPP_WARN(
        get_logger(),
        "Failed to finish jerk speed planning in %d steps (u=%.4f).",
        max_planning_steps_, u);
      return false;
    }

    CartesianSample last;
    last.p = anchors.back();
    last.dp.setZero();
    last.ddp.setZero();
    samples_out.push_back(last);
    return true;
  }

  static double compute_position_arc_length(const std::vector<CartesianSample> & samples)
  {
    if (samples.size() < 2U) {
      return 0.0;
    }

    double length = 0.0;
    for (size_t i = 1; i < samples.size(); ++i) {
      length += (samples[i].p.head<3>() - samples[i - 1U].p.head<3>()).norm();
    }
    return length;
  }

  static double compute_pitch_arc_length(const std::vector<CartesianSample> & samples)
  {
    if (samples.size() < 2U) {
      return 0.0;
    }

    double length = 0.0;
    for (size_t i = 1; i < samples.size(); ++i) {
      length += std::abs(wrap_to_pi(samples[i].p(3) - samples[i - 1U].p(3)));
    }
    return length;
  }

  void on_cartesian_path_request(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < static_cast<size_t>(kDim) || msg->data.size() % static_cast<size_t>(kDim) != 0) {
      RCLCPP_WARN(
        get_logger(),
        "Invalid /cartesian_path_request: expected flattened [x,y,z,pitch]*N, N>=1, got %zu values.",
        msg->data.size());
      return;
    }

    if (!has_current_ee_pose_) {
      RCLCPP_WARN(
        get_logger(),
        "No /current_ee_pose_pitch received yet. Cannot use current task frame as trajectory start.");
      return;
    }

    std::vector<Eigen::Vector4d> anchors;
    anchors.reserve(msg->data.size() / static_cast<size_t>(kDim) + 1U);
    anchors.push_back(current_ee_pose_);
    for (size_t i = 0; i < msg->data.size(); i += static_cast<size_t>(kDim)) {
      Eigen::Vector4d p = Eigen::Vector4d::Zero();
      for (int j = 0; j < kDim; ++j) {
        p(j) = msg->data[i + static_cast<size_t>(j)];
      }
      anchors.push_back(p);
    }
    unwrap_pitch_sequence(anchors);
    const MotionLimits limits = current_motion_limits();

    std::vector<CartesianSample> planned;
    if (!build_cartesian_samples(anchors, limits, planned)) {
      RCLCPP_WARN(get_logger(), "Failed to plan Cartesian B-spline jerk trajectory.");
      return;
    }

    cartesian_samples_ = std::move(planned);
    publish_path_marker(cartesian_samples_);
    sample_index_ = 0;
    trajectory_active_ = true;

    const Eigen::Vector4d & start_pose = anchors.front();
    const Eigen::Vector4d & goal_pose = anchors.back();
    const double direct_position_distance = (goal_pose.head<3>() - start_pose.head<3>()).norm();
    const double direct_pitch_delta = std::abs(wrap_to_pi(goal_pose(3) - start_pose(3)));
    const double sampled_position_arc = compute_position_arc_length(cartesian_samples_);
    const double sampled_pitch_arc = compute_pitch_arc_length(cartesian_samples_);

    RCLCPP_INFO(
      get_logger(),
      "Accepted Cartesian path. waypoints=%zu samples=%zu duration=%.3f s "
      "start=[%.4f, %.4f, %.4f, %.4f] goal=[%.4f, %.4f, %.4f, %.4f] "
      "xyz_dist=%.4f m xyz_arc=%.4f m pitch_delta=%.4f rad pitch_arc=%.4f rad "
      "limits={xyz_v=%.3f xyz_a+=%.3f xyz_a-=%.3f xyz_j+=%.3f xyz_j-=%.3f pitch_v=%.3f pitch_a+=%.3f pitch_a-=%.3f pitch_j+=%.3f pitch_j-=%.3f}",
      anchors.size() >= 2U ? anchors.size() - 2U : 0U,
      cartesian_samples_.size(),
      cartesian_samples_.size() * plan_dt_,
      start_pose(0), start_pose(1), start_pose(2), start_pose(3),
      goal_pose(0), goal_pose(1), goal_pose(2), goal_pose(3),
      direct_position_distance, sampled_position_arc, direct_pitch_delta, sampled_pitch_arc,
      limits.max_cartesian_speed,
      limits.max_cartesian_acceleration_accel, limits.max_cartesian_acceleration_decel,
      limits.max_cartesian_jerk_accel, limits.max_cartesian_jerk_decel,
      limits.max_pitch_speed,
      limits.max_pitch_acceleration_accel, limits.max_pitch_acceleration_decel,
      limits.max_pitch_jerk_accel, limits.max_pitch_jerk_decel);
  }

  void on_current_ee_pose(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < static_cast<size_t>(kDim)) {
      return;
    }
    for (int i = 0; i < kDim; ++i) {
      current_ee_pose_(i) = msg->data[static_cast<size_t>(i)];
    }
    has_current_ee_pose_ = true;
  }

  void on_payload_attached(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (payload_attached_ == msg->data) {
      return;
    }

    payload_attached_ = msg->data;
    clear_planned_path(true);

    RCLCPP_INFO(
      get_logger(),
      "Payload state changed to %s. Cleared active Cartesian trajectory.",
      payload_attached_ ? "attached" : "detached");
  }

  void planner_loop()
  {
    if (!trajectory_active_) {
      return;
    }

    if (sample_index_ >= cartesian_samples_.size()) {
      clear_planned_path(true);
      return;
    }

    const CartesianSample & sample = cartesian_samples_[sample_index_];
    publish_cartesian_state(sample);

    ++sample_index_;
    if (sample_index_ >= cartesian_samples_.size()) {
      clear_planned_path(true);
      RCLCPP_INFO(get_logger(), "Cartesian trajectory output finished.");
    }
  }

  void clear_planned_path(bool clear_marker)
  {
    trajectory_active_ = false;
    cartesian_samples_.clear();
    sample_index_ = 0;
    if (clear_marker) {
      clear_path_marker();
    }
  }

  void publish_cartesian_state(const CartesianSample & sample)
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(12);
    for (int i = 0; i < kDim; ++i) {
      msg.data[static_cast<size_t>(i)] = sample.p(i);
      msg.data[static_cast<size_t>(4 + i)] = sample.dp(i);
      msg.data[static_cast<size_t>(8 + i)] = sample.ddp(i);
     
    }
    
    planned_cartesian_pub_->publish(msg);
  }

  void publish_path_marker(const std::vector<CartesianSample> & samples)
  {
    if (samples.empty()) {
      return;
    }

    visualization_msgs::msg::Marker marker;
    marker.header.stamp = now();
    marker.header.frame_id = path_frame_;
    marker.ns = "planned_cartesian_curve";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.01;
    marker.color.r = 0.1F;
    marker.color.g = 0.8F;
    marker.color.b = 0.2F;
    marker.color.a = 1.0F;
    marker.frame_locked = false;
    marker.points.reserve(samples.size());

    for (const auto & sample : samples) {
      geometry_msgs::msg::Point p;
      p.x = sample.p(0);
      p.y = sample.p(1);
      p.z = sample.p(2);
      marker.points.push_back(p);
    }

    path_marker_pub_->publish(marker);
  }

  void clear_path_marker()
  {
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = now();
    marker.header.frame_id = path_frame_;
    marker.ns = "planned_cartesian_curve";
    marker.id = 0;
    marker.action = visualization_msgs::msg::Marker::DELETE;
    path_marker_pub_->publish(marker);
  }

private:
  double planner_hz_ {500.0};
  double plan_dt_ {1.0 / 500.0};
  std::string path_frame_ {"base_link"};
  std::string path_marker_topic_ {"/planned_cartesian_curve"};
  std::string payload_attached_topic_ {"/payload_attached"};
  bool payload_attached_ {false};
  double max_cartesian_speed_empty_ {0.3};
  double max_cartesian_speed_payload_ {0.3};
  double max_cartesian_acceleration_accel_empty_ {0.3};
  double max_cartesian_acceleration_accel_payload_ {0.3};
  double max_cartesian_acceleration_decel_empty_ {0.18};
  double max_cartesian_acceleration_decel_payload_ {0.18};
  double max_cartesian_jerk_accel_empty_ {5.0};
  double max_cartesian_jerk_accel_payload_ {5.0};
  double max_cartesian_jerk_decel_empty_ {2.5};
  double max_cartesian_jerk_decel_payload_ {2.5};
  double max_pitch_speed_empty_ {0.3};
  double max_pitch_speed_payload_ {0.3};
  double max_pitch_acceleration_accel_empty_ {0.3};
  double max_pitch_acceleration_accel_payload_ {0.3};
  double max_pitch_acceleration_decel_empty_ {0.18};
  double max_pitch_acceleration_decel_payload_ {0.18};
  double max_pitch_jerk_accel_empty_ {5.0};
  double max_pitch_jerk_accel_payload_ {5.0};
  double max_pitch_jerk_decel_empty_ {2.5};
  double max_pitch_jerk_decel_payload_ {2.5};

  double derivative_step_ {1e-3};
  int max_planning_steps_ {60000};

  Eigen::Vector4d current_ee_pose_ = Eigen::Vector4d::Zero();
  bool has_current_ee_pose_ {false};

  std::vector<CartesianSample> cartesian_samples_;
  size_t sample_index_ {0};
  bool trajectory_active_ {false};

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr current_ee_pose_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr payload_state_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr cartesian_path_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr planned_cartesian_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_marker_pub_;
  rclcpp::TimerBase::SharedPtr planner_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrajectoryBsplineJerkPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
