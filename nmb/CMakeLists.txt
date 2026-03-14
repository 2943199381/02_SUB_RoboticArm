cmake_minimum_required(VERSION 3.8)
project(nmb)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio QUIET)
find_package(ruckig QUIET)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(arm_controller_node src/arm_controller_node.cpp)
target_include_directories(arm_controller_node PRIVATE include)
ament_target_dependencies(
  arm_controller_node
  rclcpp
  std_msgs
  sensor_msgs
  trajectory_msgs
)
target_link_libraries(arm_controller_node Eigen3::Eigen)
if(pinocchio_FOUND)
  target_link_libraries(arm_controller_node pinocchio::pinocchio)
  target_compile_definitions(arm_controller_node PRIVATE NMB_PINOCCHIO_FOUND=1)
endif()

add_executable(arm_automation_state_machine_node src/arm_automation_state_machine_node.cpp)
target_include_directories(arm_automation_state_machine_node PRIVATE include)
ament_target_dependencies(
  arm_automation_state_machine_node
  rclcpp
  std_msgs
  sensor_msgs
)
target_link_libraries(arm_automation_state_machine_node Eigen3::Eigen)

add_executable(motor_comm_node src/motor_comm_node.cpp)
ament_target_dependencies(
  motor_comm_node
  rclcpp
  std_msgs
)

add_executable(trajectory_bspline_jerk_planner_node src/trajectory_bspline_jerk_planner_node.cpp)
target_include_directories(trajectory_bspline_jerk_planner_node PRIVATE include)
ament_target_dependencies(
  trajectory_bspline_jerk_planner_node
  rclcpp
  std_msgs
  geometry_msgs
  visualization_msgs
)
target_link_libraries(trajectory_bspline_jerk_planner_node Eigen3::Eigen)

add_executable(cartesian_ik_mapper_node src/cartesian_ik_mapper_node.cpp)
target_include_directories(cartesian_ik_mapper_node PRIVATE include)
ament_target_dependencies(
  cartesian_ik_mapper_node
  rclcpp
  std_msgs
  sensor_msgs
)
target_link_libraries(cartesian_ik_mapper_node Eigen3::Eigen)
if(pinocchio_FOUND)
  target_link_libraries(cartesian_ik_mapper_node pinocchio::pinocchio)
  target_compile_definitions(cartesian_ik_mapper_node PRIVATE NMB_PINOCCHIO_FOUND=1)
endif()

add_executable(robot_description_switcher_node src/robot_description_switcher_node.cpp)
target_include_directories(robot_description_switcher_node PRIVATE include)
ament_target_dependencies(
  robot_description_switcher_node
  rclcpp
  rcl_interfaces
  std_msgs
)

install(TARGETS
  arm_controller_node
  arm_automation_state_machine_node
  motor_comm_node
  trajectory_bspline_jerk_planner_node
  cartesian_ik_mapper_node
  robot_description_switcher_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/mujoco_joint_sim_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  rviz
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  ../urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
