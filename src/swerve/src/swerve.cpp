#include <iostream>
#include <rev/SparkMax.h>
#include <chrono>
#include <thread>
#include <hal/HALBase.h>
#include "swerve/swerve_node.h"

int main(int argc, char ** argv)
{
    setenv("HALSIM_EXTENSIONS", "libhalsim_socketcan.so", 0);
    setenv("SOCKETCAN_INTERFACE", "can1", 0);
    HAL_Initialize(500, 0);

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SwerveNode>());
    rclcpp::shutdown();

    HAL_Shutdown();
    return 0;
}
