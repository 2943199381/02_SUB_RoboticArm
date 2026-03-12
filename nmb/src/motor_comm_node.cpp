#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"

using namespace std::chrono_literals;

class MotorCommNode : public rclcpp::Node
{
public:
  MotorCommNode()
  : Node("motor_comm_node")
  {
    tx_topic_ = declare_parameter<std::string>("tx_topic", "/motor_tx_packet");
    rx_topic_ = declare_parameter<std::string>("rx_topic", "/motor_rx_packet");
    io_cycle_ms_ = declare_parameter<int>("io_cycle_ms", 2);

    tx_sub_ = create_subscription<std_msgs::msg::UInt8MultiArray>(
      tx_topic_, 50,
      std::bind(&MotorCommNode::on_tx_packet, this, std::placeholders::_1));

    rx_pub_ = create_publisher<std_msgs::msg::UInt8MultiArray>(rx_topic_, 50);

    io_timer_ = create_wall_timer(
      std::chrono::milliseconds(std::max(io_cycle_ms_, 1)),
      std::bind(&MotorCommNode::io_once, this));

    RCLCPP_INFO(
      get_logger(),
      "motor_comm_node started. tx_topic=%s, rx_topic=%s",
      tx_topic_.c_str(), rx_topic_.c_str());
  }

private:
  void on_tx_packet(const std_msgs::msg::UInt8MultiArray::SharedPtr msg)
  {
    last_tx_packet_ = msg->data;

    // 这里是和单片机串口/CAN 通信的接入点：
    // 1) 将 msg->data 按协议打包发送给 MCU
    // 2) 在 io_once() 中读取 MCU 返回并发布到 rx_topic_
  }

  void io_once()
  {
    if (last_tx_packet_.empty()) {
      return;
    }

    std_msgs::msg::UInt8MultiArray rx;
    rx.data = build_mock_rx(last_tx_packet_);
    rx_pub_->publish(rx);
  }

  static std::vector<uint8_t> build_mock_rx(const std::vector<uint8_t> & tx)
  {
    std::vector<uint8_t> rx;
    rx.reserve(tx.size() + 2);
    rx.push_back(0xAA);
    for (auto b : tx) {
      rx.push_back(b);
    }
    rx.push_back(0x55);
    return rx;
  }

private:
  std::string tx_topic_;
  std::string rx_topic_;
  int io_cycle_ms_ {2};

  std::vector<uint8_t> last_tx_packet_;

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
