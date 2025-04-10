//
// Created by team204 on 4/9/25.
//

#ifndef SWERVE_NODE_H
#define SWERVE_NODE_H

#include "rclcpp/rclcpp.hpp"
#include "ctre/phoenix6/Pigeon2.hpp"
#include "swerve/swerve_module.h"
#include "vikings_msgs/srv/blank.hpp"
#include "vikings_msgs/srv/drive_swerve.hpp"

class SwerveNode : public rclcpp::Node {
public:
    SwerveNode();

private:
    ctre::phoenix6::hardware::Pigeon2 gyro;
    SwerveModule modules[4];

    rclcpp::Service<vikings_msgs::srv::Blank>::SharedPtr zeroGyroService;
    rclcpp::Service<vikings_msgs::srv::Blank>::SharedPtr zeroDriveEncodersService;
    rclcpp::Service<vikings_msgs::srv::DriveSwerve>::SharedPtr driveService;

    frc::Rotation2d GetYaw();
    void ZeroGyro();
    void ZeroDriveEncoders();
    void DriveCb(const std::shared_ptr<vikings_msgs::srv::DriveSwerve::Request> request, const std::shared_ptr<vikings_msgs::srv::DriveSwerve::Response> response);
};

#endif //SWERVE_NODE_H
