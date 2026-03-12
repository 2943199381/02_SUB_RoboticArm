#include "UsbVcpDriver.hpp"

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <csignal>
#include <thread>

namespace {

constexpr uint16_t kCmdHostReq = 0x0001;
std::atomic<bool> g_running{true};

void onSignal(int) {
  g_running.store(false);
}

}  // namespace

int main(int argc, char* argv[]) {
  const std::string dev = (argc > 1) ? argv[1] : "/dev/ttyACM0";
  const int baud = (argc > 2) ? std::stoi(argv[2]) : 115200;

  usb_host::UsbVcpDriver driver(dev, baud);
  std::string error;
  if (!driver.openPort(&error)) {
    std::cerr << error << "\n";
    return 1;
  }

  std::cout << "Listening on " << dev << " @ " << baud << "\n";
  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  uint8_t txCounter = 0;
  auto nextSend = std::chrono::steady_clock::now();

  const auto onFrame = [&](const usb_host::ProtocolFrame& frame) {
    std::cout << "Frame: cmd=0x" << std::hex << std::setw(4) << std::setfill('0')
              << frame.cmdId << std::dec
              << ", len=" << frame.payload.size() << ", payload=";

    for (const uint8_t b : frame.payload) {
      std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b) << ' ';
    }
    std::cout << std::dec << "\n";

  };

  if (!driver.startReceiveThread(onFrame, &error)) {
    std::cerr << "start receive thread failed: " << error << "\n";
    return 1;
  }



  
  while (g_running.load()) {
    const auto now = std::chrono::steady_clock::now();

    if (now >= nextSend) {
      std::vector<uint8_t> payload = {txCounter, static_cast<uint8_t>(now.time_since_epoch().count() & 0xFF)};

      if (!driver.sendFrame(kCmdHostReq, payload, &error)) {
        std::cerr << "send failed: " << error << "\n";
        return 1;
      }

      txCounter = static_cast<uint8_t>(txCounter + 1);
      nextSend = now + std::chrono::milliseconds(100);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  driver.stopReceiveThread();
  driver.closePort();
  return 0;
}
