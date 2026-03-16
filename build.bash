#!/bin/bash
set -e
colcon build --packages-select nmb --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 
