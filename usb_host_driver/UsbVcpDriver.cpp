#include "UsbVcpDriver.hpp"

#include <cerrno>
#include <cstring>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace usb_host {

namespace {

#if defined(__linux__)
speed_t toBaud(int baudRate) {
  switch (baudRate) {
    case 9600:
      return B9600;
    case 19200:
      return B19200;
    case 38400:
      return B38400;
    case 57600:
      return B57600;
    case 115200:
      return B115200;
    case 230400:
      return B230400;
    default:
      return B115200;
  }
}
#endif

#if defined(__linux__)
std::string errnoToString() { return std::strerror(errno); }
#endif

}  // namespace

UsbVcpDriver::UsbVcpDriver(std::string device, int baudRate)
    : device_(std::move(device)), baudRate_(baudRate), fd_(-1) {}

UsbVcpDriver::~UsbVcpDriver() { closePort(); }

bool UsbVcpDriver::openPort(std::string* error) {
#if !defined(__linux__)
  if (error != nullptr) {
    *error = "UsbVcpDriver is only supported on Linux host";
  }
  return false;
#else
  if (fd_ >= 0) {
    return true;
  }

  fd_ = ::open(device_.c_str(), O_RDWR | O_NOCTTY);
  if (fd_ < 0) {
    if (error != nullptr) {
      *error = "open failed: " + device_ + ", error=" + errnoToString();
    }
    return false;
  }

  if (!configurePort(error)) {
    closePort();
    return false;
  }

  rxBuffer_.clear();
  return true;
#endif
}

void UsbVcpDriver::closePort() {
  stopReceiveThread();
#if defined(__linux__)
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
#else
  fd_ = -1;
#endif
}

bool UsbVcpDriver::isOpen() const { return fd_ >= 0; }

bool UsbVcpDriver::pollOnce(const FrameCallback& callback, int timeoutMs, std::string* error) {
#if !defined(__linux__)
  (void)callback;
  (void)timeoutMs;
  if (error != nullptr) {
    *error = "UsbVcpDriver is only supported on Linux host";
  }
  return false;
#else
  if (fd_ < 0) {
    if (error != nullptr) {
      *error = "device is not open";
    }
    return false;
  }

  pollfd pfd{};
  pfd.fd = fd_;
  pfd.events = POLLIN;

  const int pollRet = ::poll(&pfd, 1, timeoutMs);
  if (pollRet < 0) {
    if (error != nullptr) {
      *error = "poll failed, error=" + errnoToString();
    }
    return false;
  }

  if (pollRet == 0) {
    return true;
  }

  if ((pfd.revents & POLLIN) == 0) {
    return true;
  }

  uint8_t temp[256] = {0};
  const ssize_t bytesRead = ::read(fd_, temp, sizeof(temp));
  if (bytesRead < 0) {
    if (errno == EINTR || errno == EAGAIN) {
      return true;
    }
    if (error != nullptr) {
      *error = "read failed, error=" + errnoToString();
    }
    return false;
  }

  if (bytesRead == 0) {
    return true;
  }

  appendBytes(temp, static_cast<size_t>(bytesRead));
  return parseBuffer(callback);
#endif
}

bool UsbVcpDriver::run(const FrameCallback& callback, std::string* error) {
  while (true) {
    if (!pollOnce(callback, 100, error)) {
      return false;
    }
  }
}

bool UsbVcpDriver::startReceiveThread(const FrameCallback& callback, std::string* error) {
  if (receiveThreadRunning_.load()) {
    if (error != nullptr) {
      *error = "receive thread already running";
    }
    return false;
  }

  if (fd_ < 0) {
    if (error != nullptr) {
      *error = "device is not open";
    }
    return false;
  }

  receiveThreadRunning_.store(true);
  receiveThread_ = std::thread([this, callback]() {
    std::string loopError;
    while (receiveThreadRunning_.load()) {
      if (!pollOnce(callback, 100, &loopError)) {
        break;
      }
    }
    receiveThreadRunning_.store(false);
  });
  return true;
}

void UsbVcpDriver::stopReceiveThread() {
  receiveThreadRunning_.store(false);
  if (receiveThread_.joinable()) {
    receiveThread_.join();
  }
}

bool UsbVcpDriver::isReceiveThreadRunning() const { return receiveThreadRunning_.load(); }

bool UsbVcpDriver::sendFrame(uint16_t cmdId,
                             const std::vector<uint8_t>& payload,
                             std::string* error) {
  if (fd_ < 0) {
    if (error != nullptr) {
      *error = "device is not open";
    }
    return false;
  }

  if (payload.size() > kMaxPayload) {
    if (error != nullptr) {
      *error = "payload too large";
    }
    return false;
  }

  std::vector<uint8_t> frame;
  frame.reserve(kHeaderSize + payload.size() + kCrcSize);
  frame.push_back(kSof);

  const uint16_t payloadLen = static_cast<uint16_t>(payload.size());
  frame.push_back(static_cast<uint8_t>(payloadLen & 0xFF));
  frame.push_back(static_cast<uint8_t>((payloadLen >> 8) & 0xFF));
  frame.push_back(static_cast<uint8_t>(cmdId & 0xFF));
  frame.push_back(static_cast<uint8_t>((cmdId >> 8) & 0xFF));
  frame.insert(frame.end(), payload.begin(), payload.end());

  const uint16_t crc = crc16Modbus(frame.data(), frame.size());
  frame.push_back(static_cast<uint8_t>(crc & 0xFF));
  frame.push_back(static_cast<uint8_t>((crc >> 8) & 0xFF));

  return writeAll(frame.data(), frame.size(), error);
}

uint8_t UsbVcpDriver::crc8Maxim(const uint8_t* data, size_t length) {
  uint8_t crc = 0x00;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      if ((crc & 0x01U) != 0U) {
        crc = static_cast<uint8_t>((crc >> 1) ^ 0x8CU);
      } else {
        crc = static_cast<uint8_t>(crc >> 1);
      }
    }
  }
  return crc;
}

uint16_t UsbVcpDriver::crc16Modbus(const uint8_t* data, size_t length) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      if ((crc & 0x0001U) != 0U) {
        crc = static_cast<uint16_t>((crc >> 1) ^ 0xA001U);
      } else {
        crc = static_cast<uint16_t>(crc >> 1);
      }
    }
  }
  return crc;
}

bool UsbVcpDriver::configurePort(std::string* error) {
#if !defined(__linux__)
  (void)error;
  return false;
#else
  termios tty{};
  if (::tcgetattr(fd_, &tty) != 0) {
    if (error != nullptr) {
      *error = "tcgetattr failed, error=" + errnoToString();
    }
    return false;
  }

  const speed_t speed = toBaud(baudRate_);
  ::cfsetispeed(&tty, speed);
  ::cfsetospeed(&tty, speed);

  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
  tty.c_cflag |= CLOCAL | CREAD;
  tty.c_cflag &= ~(PARENB | PARODD);
  tty.c_cflag &= ~CSTOPB;
  tty.c_cflag &= ~CRTSCTS;

  tty.c_iflag &= ~(IXON | IXOFF | IXANY);
  tty.c_iflag &= ~(ICRNL | INLCR | IGNCR);

  tty.c_lflag = 0;
  tty.c_oflag = 0;

  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;

  if (::tcsetattr(fd_, TCSANOW, &tty) != 0) {
    if (error != nullptr) {
      *error = "tcsetattr failed, error=" + errnoToString();
    }
    return false;
  }

  int modemBits = 0;
  if (::ioctl(fd_, TIOCMGET, &modemBits) == 0) {
    modemBits |= (TIOCM_DTR | TIOCM_RTS);
    (void)::ioctl(fd_, TIOCMSET, &modemBits);
  }

  return true;
#endif
}

void UsbVcpDriver::appendBytes(const uint8_t* data, size_t size) {
  rxBuffer_.insert(rxBuffer_.end(), data, data + size);
}

bool UsbVcpDriver::parseBuffer(const FrameCallback& callback) {
  while (rxBuffer_.size() >= (kHeaderSize + kCrcSize)) {
    size_t sofIndex = 0;
    while (sofIndex < rxBuffer_.size()) {
      if (rxBuffer_[sofIndex] == kSof) {
        break;
      }
      ++sofIndex;
    }

    if (sofIndex >= rxBuffer_.size()) {
      rxBuffer_.clear();
      return true;
    }

    if (sofIndex > 0) {
      rxBuffer_.erase(rxBuffer_.begin(), rxBuffer_.begin() + static_cast<long>(sofIndex));
    }

    if (rxBuffer_.size() < (kHeaderSize + kCrcSize)) {
      return true;
    }

    const uint16_t payloadLen = static_cast<uint16_t>(rxBuffer_[1] | (rxBuffer_[2] << 8));
    if (payloadLen > kMaxPayload) {
      rxBuffer_.erase(rxBuffer_.begin());
      continue;
    }

    const size_t fullLen = kHeaderSize + static_cast<size_t>(payloadLen) + kCrcSize;
    if (rxBuffer_.size() < fullLen) {
      return true;
    }

    const uint16_t recvCrc =
        static_cast<uint16_t>(rxBuffer_[fullLen - 2] | (rxBuffer_[fullLen - 1] << 8));
    const uint16_t calcCrc = crc16Modbus(rxBuffer_.data(), fullLen - kCrcSize);
    if (recvCrc == calcCrc) {
      ProtocolFrame frame;
      frame.cmdId = static_cast<uint16_t>(rxBuffer_[3] | (rxBuffer_[4] << 8));
      frame.payload.assign(rxBuffer_.begin() + static_cast<long>(kHeaderSize),
                           rxBuffer_.begin() + static_cast<long>(kHeaderSize + payloadLen));
      callback(frame);
      rxBuffer_.erase(rxBuffer_.begin(), rxBuffer_.begin() + static_cast<long>(fullLen));
      continue;
    }

    rxBuffer_.erase(rxBuffer_.begin());
  }

  return true;
}

bool UsbVcpDriver::writeAll(const uint8_t* data, size_t size, std::string* error) {
#if !defined(__linux__)
  (void)data;
  (void)size;
  if (error != nullptr) {
    *error = "UsbVcpDriver is only supported on Linux host";
  }
  return false;
#else
  size_t written = 0;
  while (written < size) {
    const ssize_t n = ::write(fd_, data + written, size - written);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (error != nullptr) {
        *error = "write failed, error=" + errnoToString();
      }
      return false;
    }
    written += static_cast<size_t>(n);
  }
  return true;
#endif
}

}  // namespace usb_host
