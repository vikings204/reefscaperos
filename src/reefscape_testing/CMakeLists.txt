cmake_minimum_required(VERSION 3.8)
project(reefscape_testing)

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
find_package(Msgpack)
# wpilib
set(CMAKE_PREFIX_PATH "/home/team204/thirdparty/2025/wpilib/")
find_package(wpilibc CONFIG REQUIRED)
find_package(hal CONFIG REQUIRED)
#find_package(wpimath CONFIG REQUIRED)

#include_directories("/home/team204/thirdparty/2025/wpilib/include/hal")
#include_directories("/home/team204/thirdparty/2025/wpilib/include/wpilibc")
#include_directories("/home/team204/thirdparty/2025/wpilib/include/wpilibNewCommands")
#include_directories("/home/team204/thirdparty/2025/wpilib/include/wpimath")
#include_directories("/home/team204/thirdparty/2025/wpilib/include/wpinet")
#include_directories("/home/team204/thirdparty/2025/wpilib/include/wpiutil")
#
#add_library(wpilibc STATIC IMPORTED)
#set_property(TARGET wpilibc PROPERTY IMPORTED_LOCATION "/home/team204/thirdparty/2025/wpilib/lib/libwpilibc.so")
#add_library(wpiHal STATIC IMPORTED)
#set_property(TARGET wpiHal PROPERTY IMPORTED_LOCATION "/home/team204/thirdparty/2025/wpilib/lib/libwpiHal.so")
#add_library(wpimath STATIC IMPORTED)
#set_property(TARGET wpimath PROPERTY IMPORTED_LOCATION "/home/team204/thirdparty/2025/wpilib/lib/libwpimath.so")
#
#target_link_libraries(wpilibc INTERFACE wpimath)

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

# run_motor
add_executable(run_motor src/run_motor.cpp)
ament_target_dependencies(run_motor rclcpp std_msgs)
target_link_libraries(run_motor wpilibc wpimath wpiHal revlib)

# install all executables
install(TARGETS run_motor DESTINATION lib/${PROJECT_NAME})