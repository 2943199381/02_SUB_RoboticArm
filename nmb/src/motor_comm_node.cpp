#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "UsbVcpDriver.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"

using namespace std::chrono_literals;

namespace {

constexpr uint16_t kDefaultSendCmdId = 0x0001;
constexpr uint16_t kDefaultJointStateCmdId = 1010;
constexpr size_t kJointStateDof = 4;
constexpr size_t kJointStateFloatCount = 8;

std::string bytes_to_hex_string(const std::vector<uint8_t> & data)
{
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (size_t i = 0; i < data.size(); ++i) {
    if (i > 0U) {
      oss << ' ';
    }
    oss << std::setw(2) << static_cast<int>(data[i]);
  }
  return oss.str();
}

}  // namespace

class MotorCommNode : public rclcpp::Node
{
public:
  MotorCommNode()
  : Node("motor_comm_node")
  {
    rx_topic_ = declare_parameter<std::string>("rx_topic", "/motor_rx_packet");
    torque_topic_ = declare_parameter<std::string>("torque_topic", "/joint_torque_cmd");
    joint_state_topic_ = declare_parameter<std::string>("joint_state_topic", "/joint_states");
    joint_names_ = declare_parameter<std::vector<std::string>>(
      "joint_names", std::vector<std::string>{"j1", "j2", "j3", "j4"});
    if (joint_names_.size() != kJointStateDof) {
      RCLCPP_WARN(
        get_logger(),
        "joint_names must have %zu entries for the fixed USB joint-state mapping. Using default names.",
        kJointStateDof);
      joint_names_ = {"j1", "j2", "j3", "j4"};
    }
    io_cycle_ms_ = declare_parameter<int>("io_cycle_ms", 2);
    serial_device_ = declare_parameter<std::string>("serial_device", "/dev/ttyACM0");
    baud_rate_ = declare_parameter<int>("baud_rate", 115200);
    const int legacy_usb_cmd_id = declare_parameter<int>("usb_cmd_id", kDefaultSendCmdId);
    send_cmd_id_ = static_cast<uint16_t>(declare_parameter<int>("send_cmd_id", legacy_usb_cmd_id));
    joint_state_cmd_id_ = static_cast<uint16_t>(
      declare_parameter<int>("joint_state_cmd_id", kDefaultJointStateCmdId));
    const int poll_timeout_param = static_cast<int>(declare_parameter<int>("poll_timeout_ms", 0));
    poll_timeout_ms_ = std::max(0, poll_timeout_param);
    publish_joint_state_from_usb_ = declare_parameter<bool>("publish_joint_state_from_usb", false);

    driver_ = std::make_unique<usb_host::UsbVcpDriver>(serial_device_, baud_rate_);

    torque_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      torque_topic_, 50,
      std::bind(&MotorCommNode::on_torque_cmd, this, std::placeholders::_1));

    rx_pub_ = create_publisher<std_msgs::msg::UInt8MultiArray>(rx_topic_, 50);
    if (publish_joint_state_from_usb_) {
      joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>(joint_state_topic_, 50);
    }

    send_timer_ = create_wall_timer(
      std::chrono::milliseconds(std::max(io_cycle_ms_, 1)),
      std::bind(&MotorCommNode::send_once, this));

      if (!driver_->startReceiveThread(std::bind(&MotorCommNode::handle_received_frame, this, std::placeholders::_1), &error)) {
      std::cerr << "start receive thread failed: " << error << "\n";}
    RCLCPP_INFO(
      get_logger(),
      "motor_comm_node started. torque_topic=%s rx_topic=%s joint_state_topic=%s publish_joint_state_from_usb=%s send_cmd_id=%u joint_state_cmd_id=%u serial_device=%s baud_rate=%d",
      torque_topic_.c_str(), rx_topic_.c_str(),
      joint_state_topic_.c_str(), publish_joint_state_from_usb_ ? "true" : "false",
      static_cast<unsigned int>(send_cmd_id_),
      static_cast<unsigned int>(joint_state_cmd_id_),
      serial_device_.c_str(), baud_rate_);
  }

private:
  void on_torque_cmd(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    last_tx_packet_.clear();
    last_tx_packet_.reserve(kJointStateDof * sizeof(float));
    for (size_t i = 0; i < kJointStateDof; ++i) {
      const float tau = static_cast<float>(i < msg->data.size() ? msg->data[i] : 0.0);
      const uint8_t * raw = reinterpret_cast<const uint8_t *>(&tau);
      last_tx_packet_.insert(last_tx_packet_.end(), raw, raw + sizeof(float));
    }
    has_pending_tx_ = true;
  }

  void send_once()
  {
    if (!ensure_driver_open()) {
      return;
    }

    if (has_pending_tx_) {
      send_pending_packet();
    }
    // poll_driver_once();
  }

  bool ensure_driver_open()
  {
    if (driver_ && driver_->isOpen()) {
      return true;
    }

    if (!driver_) {
      driver_ = std::make_unique<usb_host::UsbVcpDriver>(serial_device_, baud_rate_);
    }

    std::string error;
    if (!driver_->openPort(&error)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Failed to open motor USB device %s @ %d: %s",
        serial_device_.c_str(), baud_rate_, error.c_str());
      return false;
    }
    if (!driver_->startReceiveThread(std::bind(&MotorCommNode::handle_received_frame, this, std::placeholders::_1), &error)) {
        RCLCPP_ERROR(get_logger(), "Failed to restart receive thread after reconnect: %s", error.c_str());
        driver_->closePort(); // 线程起不来，干脆关掉等下次重试
        return false;
    }
    RCLCPP_INFO(
      get_logger(),
      "Opened motor USB device %s @ %d",
      serial_device_.c_str(), baud_rate_);
    return true;
  }

  void send_pending_packet()
  {
    if (last_tx_packet_.empty()) {
      has_pending_tx_ = false;
      return;
    }

    std::string error;
    if (!driver_->sendFrame(send_cmd_id_, last_tx_packet_, &error)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Failed to send motor command over USB: %s",
        error.c_str());
      driver_->closePort();
      return;
    }

    has_pending_tx_ = false;
  }

  // void poll_driver_once()
  // {
  //   std::string error;
  //   const auto frame_cb = [this](const usb_host::ProtocolFrame & frame) {
  //     handle_received_frame(frame);
  //   };

  //   if (!driver_->pollOnce(frame_cb, poll_timeout_ms_, &error)) {
  //     RCLCPP_ERROR_THROTTLE(
  //       get_logger(), *get_clock(), 2000,
  //       "Failed to poll motor USB device: %s",
  //       error.c_str());
  //     driver_->closePort();
  //   }
  // }

  void handle_received_frame(const usb_host::ProtocolFrame & frame)
  {
    std_msgs::msg::UInt8MultiArray rx;
    rx.data.reserve(frame.payload.size() + 2U);
    rx.data.push_back(static_cast<uint8_t>(frame.cmdId & 0xFF));
    rx.data.push_back(static_cast<uint8_t>((frame.cmdId >> 8) & 0xFF));
    rx.data.insert(rx.data.end(), frame.payload.begin(), frame.payload.end());
    rx_pub_->publish(rx);

    if (publish_joint_state_from_usb_) {
      publish_joint_state_from_frame(frame);
    }

    RCLCPP_INFO(
      get_logger(),
      "motor usb rx cmd=0x%04x len=%zu payload=[%s]",
      frame.cmdId, frame.payload.size(), bytes_to_hex_string(frame.payload).c_str());
  }

  bool decode_joint_state_payload(
    const std::vector<uint8_t> & payload,
    std::vector<double> & position,
    std::vector<double> & velocity) const
  {
    constexpr size_t kExpectedBytes = kJointStateFloatCount * sizeof(float);
    if (payload.size() < kExpectedBytes) {
      return false;
    }

    float values[kJointStateFloatCount] {};
    std::memcpy(values, payload.data(), kExpectedBytes);

    position.resize(kJointStateDof);
    velocity.resize(kJointStateDof);
    for (size_t i = 0; i < kJointStateDof; ++i) {
      position[i] = static_cast<double>(values[i]);
      velocity[i] = static_cast<double>(values[i + kJointStateDof]);
    }
    return true;
  }

  void publish_joint_state_from_frame(const usb_host::ProtocolFrame & frame)
  {
    if (!joint_state_pub_) {
      return;
    }
    if (frame.cmdId != joint_state_cmd_id_) {
      return;
    }

    std::vector<double> position;
    std::vector<double> velocity;
    if (!decode_joint_state_payload(frame.payload, position, velocity)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Failed to decode motor joint state frame. cmd=%u len=%zu expected_bytes=%zu",
        static_cast<unsigned int>(frame.cmdId), frame.payload.size(),
        kJointStateFloatCount * sizeof(float));
      return;
    }

    sensor_msgs::msg::JointState msg;
    msg.header.stamp = now();
    msg.name = joint_names_;
    msg.position = position;
    msg.velocity = velocity;
    joint_state_pub_->publish(msg);
  }

private:
  std::string rx_topic_;
  std::string torque_topic_;
  std::string joint_state_topic_;
  std::vector<std::string> joint_names_;
  int io_cycle_ms_ {2};
  std::string serial_device_ {"/dev/ttyACM0"};
  int baud_rate_ {115200};
  uint16_t send_cmd_id_ {kDefaultSendCmdId};
  uint16_t joint_state_cmd_id_ {kDefaultJointStateCmdId};
  int poll_timeout_ms_ {0};
  bool publish_joint_state_from_usb_ {false};

  std::unique_ptr<usb_host::UsbVcpDriver> driver_;
  std::vector<uint8_t> last_tx_packet_;
  bool has_pending_tx_ {false};

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr torque_sub_;
  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr rx_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::TimerBase::SharedPtr send_timer_;
  std::string error;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MotorCommNode>());
  rclcpp::shutdown();
  return 0;
}
