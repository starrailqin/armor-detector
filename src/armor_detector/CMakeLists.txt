cmake_minimum_required(VERSION 3.8)
project(armor_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(aim_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
set(ONNXRUNTIME_DIR "$ENV{HOME}/onnxruntime-linux-x64-1.24.4")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

if(NOT EXISTS "${ONNXRUNTIME_DIR}")
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_DIR}")
endif()


include_directories(${ONNXRUNTIME_INCLUDE_DIR})

link_directories(${ONNXRUNTIME_LIB_DIR})



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


add_executable(detector_node src/detector_node.cpp)
ament_target_dependencies(detector_node rclcpp sensor_msgs aim_interfaces cv_bridge OpenCV)

target_link_libraries(detector_node ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)

install(TARGETS
	detector_node
	DESTINATION lib/${PROJECT_NAME})
ament_package()

