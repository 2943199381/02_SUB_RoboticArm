#pragma once

#include <cstdint>
#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace usb_host {

struct ProtocolFrame {
  uint16_t cmdId = 0;
  std::vector<uint8_t> payload;
};

class UsbVcpDriver {
 public:
  using FrameCallback = std::function<void(const ProtocolFrame&)>;

  UsbVcpDriver(std::string device, int baudRate = 115200);
  ~UsbVcpDriver();

  UsbVcpDriver(const UsbVcpDriver&) = delete;
  UsbVcpDriver& operator=(const UsbVcpDriver&) = delete;

  bool openPort(std::string* error = nullptr);
  void closePort();
  bool isOpen() const;

  bool pollOnce(const FrameCallback& callback, int timeoutMs, std::string* error = nullptr);
  bool run(const FrameCallback& callback, std::string* error = nullptr);
  bool startReceiveThread(const FrameCallback& callback, std::string* error = nullptr);
  void stopReceiveThread();
  bool isReceiveThreadRunning() const;

  bool sendFrame(uint16_t cmdId,
                 const std::vector<uint8_t>& payload,
                 std::string* error = nullptr);

  static uint8_t crc8Maxim(const uint8_t* data, size_t length);
  static uint16_t crc16Modbus(const uint8_t* data, size_t length);

 private:
  static constexpr uint8_t kSof = 0xA5;
  static constexpr size_t kHeaderSize = 5;
  static constexpr size_t kCrcSize = 2;
  static constexpr size_t kMaxPayload = 1024;

  // 协议：
  // [0] SOF(0xA5)
  // [1..2] payload_len (LE)
  // [3..4] cmd_id (LE)
  // [5..] payload
  // [end-2..end-1] CRC16(frame[0..end-3], LE)

  bool configurePort(std::string* error);
  void appendBytes(const uint8_t* data, size_t size);
  bool parseBuffer(const FrameCallback& callback);
  bool writeAll(const uint8_t* data, size_t size, std::string* error);

  std::string device_;
  int baudRate_;
  int fd_;
  std::vector<uint8_t> rxBuffer_;
  std::thread receiveThread_;
  std::atomic<bool> receiveThreadRunning_{false};
};

}  // namespace usb_host
