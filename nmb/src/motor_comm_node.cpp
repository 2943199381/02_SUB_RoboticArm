#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "UsbVcpDriver.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"

using namespace std::chrono_literals;

namespace {

constexpr uint16_t kDefaultUsbCmdId = 0x0001;

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
    tx_topic_ = declare_parameter<std::string>("tx_topic", "/motor_tx_packet");
    rx_topic_ = declare_parameter<std::string>("rx_topic", "/motor_rx_packet");
    io_cycle_ms_ = declare_parameter<int>("io_cycle_ms", 2);
    serial_device_ = declare_parameter<std::string>("serial_device", "/dev/ttyACM0");
    baud_rate_ = declare_parameter<int>("baud_rate", 115200);
    usb_cmd_id_ = static_cast<uint16_t>(declare_parameter<int>("usb_cmd_id", kDefaultUsbCmdId));
    const int poll_timeout_param = static_cast<int>(declare_parameter<int>("poll_timeout_ms", 0));
    poll_timeout_ms_ = std::max(0, poll_timeout_param);

    driver_ = std::make_unique<usb_host::UsbVcpDriver>(serial_device_, baud_rate_);

    tx_sub_ = create_subscription<std_msgs::msg::UInt8MultiArray>(
      tx_topic_, 50,
      std::bind(&MotorCommNode::on_tx_packet, this, std::placeholders::_1));

    rx_pub_ = create_publisher<std_msgs::msg::UInt8MultiArray>(rx_topic_, 50);

    io_timer_ = create_wall_timer(
      std::chrono::milliseconds(std::max(io_cycle_ms_, 1)),
      std::bind(&MotorCommNode::io_once, this));

    RCLCPP_INFO(
      get_logger(),
      "motor_comm_node started. tx_topic=%s rx_topic=%s serial_device=%s baud_rate=%d usb_cmd_id=0x%04x",
      tx_topic_.c_str(), rx_topic_.c_str(), serial_device_.c_str(), baud_rate_, usb_cmd_id_);
  }

private:
  void on_tx_packet(const std_msgs::msg::UInt8MultiArray::SharedPtr msg)
  {
    last_tx_packet_ = msg->data;
    has_pending_tx_ = !last_tx_packet_.empty();
  }

  void io_once()
  {
    if (!ensure_driver_open()) {
      return;
    }

    if (has_pending_tx_) {
      send_pending_packet();
    }

    poll_driver_once();
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
    if (!driver_->sendFrame(usb_cmd_id_, last_tx_packet_, &error)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Failed to send motor command over USB: %s",
        error.c_str());
      driver_->closePort();
      return;
    }

    has_pending_tx_ = false;
  }

  void poll_driver_once()
  {
    std::string error;
    const auto frame_cb = [this](const usb_host::ProtocolFrame & frame) {
      handle_received_frame(frame);
    };

    if (!driver_->pollOnce(frame_cb, poll_timeout_ms_, &error)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Failed to poll motor USB device: %s",
        error.c_str());
      driver_->closePort();
    }
  }

  void handle_received_frame(const usb_host::ProtocolFrame & frame)
  {
    std_msgs::msg::UInt8MultiArray rx;
    rx.data.reserve(frame.payload.size() + 2U);
    rx.data.push_back(static_cast<uint8_t>(frame.cmdId & 0xFF));
    rx.data.push_back(static_cast<uint8_t>((frame.cmdId >> 8) & 0xFF));
    rx.data.insert(rx.data.end(), frame.payload.begin(), frame.payload.end());
    rx_pub_->publish(rx);

    RCLCPP_INFO(
      get_logger(),
      "motor usb rx cmd=0x%04x len=%zu payload=[%s]",
      frame.cmdId, frame.payload.size(), bytes_to_hex_string(frame.payload).c_str());
  }

private:
  std::string tx_topic_;
  std::string rx_topic_;
  int io_cycle_ms_ {2};
  std::string serial_device_ {"/dev/ttyACM0"};
  int baud_rate_ {115200};
  uint16_t usb_cmd_id_ {kDefaultUsbCmdId};
  int poll_timeout_ms_ {0};

  std::unique_ptr<usb_host::UsbVcpDriver> driver_;
  std::vector<uint8_t> last_tx_packet_;
  bool has_pending_tx_ {false};

  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr tx_sub_;
  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr rx_pub_;
  rclcpp::TimerBase::SharedPtr io_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MotorCommNode>());
  rclcpp::shutdown();
  return 0;
}
