cmake_minimum_required(VERSION 3.8)
project(robotcomm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
# installed libs
find_package(cppzmq)
add_library(msgpack INTERFACE)
target_include_directories(msgpack INTERFACE "/home/team204/thirdparty/cppack/msgpack/include/")
# wpilib
set(CMAKE_PREFIX_PATH "/home/team204/thirdparty/2025/wpilib/")
find_package(wpilibc CONFIG REQUIRED)
find_package(hal CONFIG REQUIRED)
# revlib driver
include_directories("/home/team204/thirdparty/2025/revlib-driver/include/")
add_library(revlib-driver STATIC IMPORTED)
set_property(TARGET revlib-driver PROPERTY IMPORTED_LOCATION "/home/team204/thirdparty/2025/revlib-driver/lib/linux/arm64/shared/libREVLibDriver.so")
# revlib
include_directories("/home/team204/thirdparty/2025/revlib/include/")
add_library(revlib STATIC IMPORTED)
set_property(TARGET revlib PROPERTY IMPORTED_LOCATION "/home/team204/thirdparty/2025/revlib/lib/linux/arm64/shared/libREVLib.so")
target_link_libraries(revlib INTERFACE wpilibc)
target_link_libraries(revlib INTERFACE revlib-driver)



add_executable(robotcomm src/robotcomm.cpp)
target_include_directories(robotcomm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(robotcomm PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

add_executable(status src/status_node.cpp)
target_include_directories(status PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(status PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(status rclcpp std_msgs)
target_link_libraries(status cppzmq msgpack)


install(TARGETS robotcomm status
  DESTINATION lib/${PROJECT_NAME})

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

ament_package()
