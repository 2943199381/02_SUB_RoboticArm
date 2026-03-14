#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/parameter_client.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"

class RobotDescriptionSwitcherNode : public rclcpp::Node
{
public:
  RobotDescriptionSwitcherNode()
  : Node("robot_description_switcher_node")
  {
    urdf_path_empty_ = declare_parameter<std::string>("urdf_path_empty", "");
    urdf_path_payload_ = declare_parameter<std::string>("urdf_path_payload", urdf_path_empty_);
    payload_attached_topic_ = declare_parameter<std::string>("payload_attached_topic", "/payload_attached");
    payload_attached_ = declare_parameter<bool>("payload_attached_initial", false);
    target_node_name_ = declare_parameter<std::string>("target_node_name", "robot_state_publisher");
    robot_description_topic_ = declare_parameter<std::string>("robot_description_topic", "/robot_description");
    sync_period_sec_ = std::max(0.1, declare_parameter<double>("sync_period_sec", 0.5));

    empty_description_ = load_urdf_text(urdf_path_empty_);
    payload_description_ = load_urdf_text(urdf_path_payload_);

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    description_pub_ = create_publisher<std_msgs::msg::String>(robot_description_topic_, qos);

    parameters_client_ = std::make_shared<rclcpp::AsyncParametersClient>(this, normalized_target_node_name());

    payload_sub_ = create_subscription<std_msgs::msg::Bool>(
      payload_attached_topic_, 10,
      std::bind(&RobotDescriptionSwitcherNode::on_payload_attached, this, std::placeholders::_1));

    sync_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(sync_period_sec_)),
      std::bind(&RobotDescriptionSwitcherNode::on_sync_timer, this));

    publish_description();
    request_remote_sync();

    RCLCPP_INFO(
      get_logger(),
      "robot_description_switcher_node started. payload_topic=%s robot_description_topic=%s target_node=%s payload_attached_initial=%s",
      payload_attached_topic_.c_str(), robot_description_topic_.c_str(),
      normalized_target_node_name().c_str(), payload_attached_ ? "true" : "false");
  }

private:
  static std::string uri_encode_path(const std::string & path)
  {
    std::ostringstream out;
    out << std::uppercase << std::hex;
    for (const unsigned char ch : path) {
      if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '~' || ch == '/' || ch == ':') {
        out << static_cast<char>(ch);
      } else {
        out << '%' << std::setw(2) << std::setfill('0') << static_cast<int>(ch);
      }
    }
    return out.str();
  }

  static void replace_all(std::string & input, const std::string & from, const std::string & to)
  {
    if (from.empty()) {
      return;
    }

    size_t pos = 0;
    while ((pos = input.find(from, pos)) != std::string::npos) {
      input.replace(pos, from.size(), to);
      pos += to.size();
    }
  }

  std::string load_urdf_text(const std::string & urdf_path) const
  {
    if (urdf_path.empty()) {
      throw std::runtime_error("URDF path must not be empty");
    }

    std::ifstream input(urdf_path);
    if (!input.is_open()) {
      throw std::runtime_error("Failed to open URDF: " + urdf_path);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    std::string text = buffer.str();

    const auto slash_pos = urdf_path.find_last_of('/');
    const std::string mesh_dir = slash_pos == std::string::npos ? "." : urdf_path.substr(0, slash_pos);
    const std::string mesh_uri_prefix = "file://" + uri_encode_path(mesh_dir) + "/";
    replace_all(text, "package://nmb/urdf/", mesh_uri_prefix);
    return text;
  }

  std::string normalized_target_node_name() const
  {
    if (target_node_name_.empty()) {
      return "/robot_state_publisher";
    }
    return target_node_name_.front() == '/' ? target_node_name_ : "/" + target_node_name_;
  }

  const std::string & current_description() const
  {
    return payload_attached_ ? payload_description_ : empty_description_;
  }

  void publish_description()
  {
    std_msgs::msg::String msg;
    msg.data = current_description();
    description_pub_->publish(msg);
  }

  void request_remote_sync()
  {
    if (!parameters_client_->service_is_ready()) {
      pending_remote_sync_ = true;
      return;
    }

    pending_remote_sync_ = false;
    sync_in_flight_ = true;
    parameters_client_->set_parameters(
      {rclcpp::Parameter("robot_description", current_description())},
      [this](std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> future) {
        on_sync_response(future);
      });
  }

  void on_sync_response(
    std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> future)
  {
    sync_in_flight_ = false;

    try {
      const auto results = future.get();
      if (results.empty()) {
        pending_remote_sync_ = true;
        RCLCPP_WARN(get_logger(), "robot_state_publisher returned no SetParameters result.");
        return;
      }

      const auto & result = results.front();
      if (!result.successful) {
        pending_remote_sync_ = true;
        RCLCPP_WARN(
          get_logger(), "robot_description update rejected: %s",
          result.reason.empty() ? "unknown reason" : result.reason.c_str());
        return;
      }

      RCLCPP_INFO(
        get_logger(), "Synchronized robot_description to %s model.",
        payload_attached_ ? "payload" : "empty");
    } catch (const std::exception & e) {
      pending_remote_sync_ = true;
      RCLCPP_WARN(get_logger(), "Failed to sync robot_description: %s", e.what());
    }
  }

  void on_payload_attached(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (payload_attached_ == msg->data) {
      return;
    }

    payload_attached_ = msg->data;
    publish_description();
    pending_remote_sync_ = true;
    if (!sync_in_flight_) {
      request_remote_sync();
    }
  }

  void on_sync_timer()
  {
    if (!pending_remote_sync_ || sync_in_flight_) {
      return;
    }

    request_remote_sync();
    if (!parameters_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Waiting for parameter service %s/set_parameters",
        normalized_target_node_name().c_str());
    }
  }

  std::string urdf_path_empty_;
  std::string urdf_path_payload_;
  std::string payload_attached_topic_;
  bool payload_attached_ {false};
  std::string target_node_name_;
  std::string robot_description_topic_;
  double sync_period_sec_ {0.5};

  std::string empty_description_;
  std::string payload_description_;
  bool pending_remote_sync_ {true};
  bool sync_in_flight_ {false};

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr description_pub_;
  std::shared_ptr<rclcpp::AsyncParametersClient> parameters_client_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr payload_sub_;
  rclcpp::TimerBase::SharedPtr sync_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RobotDescriptionSwitcherNode>());
  rclcpp::shutdown();
  return 0;
}
