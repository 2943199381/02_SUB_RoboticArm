cmake_minimum_required(VERSION 3.10)

project(usb_host_driver LANGUAGES CXX)

add_executable(usb_recv
  UsbVcpDriver.cpp
  example_main.cpp
)

target_compile_features(usb_recv PRIVATE cxx_std_17)
