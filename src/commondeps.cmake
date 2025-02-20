cmake_minimum_required(VERSION 3.8)

# ros dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# regular libs
find_package(cppzmq)
add_library(msgpack INTERFACE)
target_include_directories(msgpack INTERFACE "/home/team204/thirdparty/cppack/msgpack/include/")

# wpilib
set(CMAKE_PREFIX_PATH "/home/team204/thirdparty/2025/wpilib")
find_package(wpilibc CONFIG REQUIRED)
find_package(hal CONFIG REQUIRED)

# vendordeps, some related to each other

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

#message(NOTICE "dependencies have been loaded from commondeps.cmake")