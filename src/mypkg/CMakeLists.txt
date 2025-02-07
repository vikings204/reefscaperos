cmake_minimum_required(VERSION 3.8)
project(mypkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(wpilibc REQUIRED)
find_package(hal REQUIRED)

# revlib driver
include_directories("/home/team204/thirdparty/revlib-driver/include/")
add_library(revlib-driver STATIC IMPORTED)
set_property(TARGET revlib-driver PROPERTY IMPORTED_LOCATION "/home/team204/thirdparty/revlib-driver/lib/linux/arm64/shared/libREVLibDriver.so")
# revlib
include_directories("/home/team204/thirdparty/revlib/include/")
add_library(revlib STATIC IMPORTED)
set_property(TARGET revlib PROPERTY IMPORTED_LOCATION "/home/team204/thirdparty/revlib/lib/linux/arm64/shared/libREVLib.so")
target_link_libraries(revlib INTERFACE wpilibc)
target_link_libraries(revlib INTERFACE revlib-driver)

add_executable(dummynode src/dummynode.cpp src/func.cpp)
target_include_directories(dummynode PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(dummynode PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_link_libraries(dummynode wpilibc)
target_link_libraries(dummynode revlib)

install(TARGETS dummynode
        DESTINATION lib/${PROJECT_NAME})

# autosquare
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
add_executable(autosquare src/autosquare.cpp)
ament_target_dependencies(autosquare rclcpp std_msgs)
install(TARGETS autosquare DESTINATION lib/${PROJECT_NAME})

# motor
add_executable(motor src/motor.cpp)
target_link_libraries(motor wpilibc wpiHal)
target_link_libraries(motor revlib)
install(TARGETS motor DESTINATION lib/${PROJECT_NAME})

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
