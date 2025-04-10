//
// Created by team204 on 4/9/25.
//

#include "swerve/swerve_node.h"

using namespace Constants::Swerve;

SwerveNode::SwerveNode() :
Node{"swerve"},
gyro{9},
modules{
    SwerveModule{0, 11, 21, frc::Rotation2d{units::degree_t{0.75 * 360.0}}, 31},
    SwerveModule{1, 12, 22, frc::Rotation2d{units::degree_t{0.888 * 360.0}}, 32},
    SwerveModule{2, 10, 20, frc::Rotation2d{units::degree_t{0.887 * 360.0}}, 30},
    SwerveModule{3, 13, 23, frc::Rotation2d{units::degree_t{0.08 * 360.0}}, 33}
}
{
    gyro.GetConfigurator().Apply(ctre::phoenix6::configs::Pigeon2Configuration{});
    ZeroGyro();

    auto driveCbWrapper = [this](const std::shared_ptr<vikings_msgs::srv::DriveSwerve::Request> request, const std::shared_ptr<vikings_msgs::srv::DriveSwerve::Response> response) {
        DriveCb(request, response);
    };
    driveService = this->create_service<vikings_msgs::srv::DriveSwerve>("swerve/drive", driveCbWrapper);

    auto zeroGyroCb = [this](const std::shared_ptr<vikings_msgs::srv::Blank::Request> request, const std::shared_ptr<vikings_msgs::srv::Blank::Response> response) {
        (void) request;
        (void) response;
        ZeroGyro();
    };
    zeroGyroService = this->create_service<vikings_msgs::srv::Blank>("swerve/zeroGyro", zeroGyroCb);

    auto zeroDriveEncodersCb = [this](const std::shared_ptr<vikings_msgs::srv::Blank::Request> request, const std::shared_ptr<vikings_msgs::srv::Blank::Response> response) {
        (void) request;
        (void) response;
        ZeroDriveEncoders();
    };
    zeroDriveEncodersService = this->create_service<vikings_msgs::srv::Blank>("swerve/zeroDriveEncoders", zeroDriveEncodersCb);
}

void SwerveNode::DriveCb(const std::shared_ptr<vikings_msgs::srv::DriveSwerve::Request> request, const std::shared_ptr<vikings_msgs::srv::DriveSwerve::Response> response) {
    (void) response;

    std::cout << "x: " << request->x << " y: " << request->y << " rot: " << request->rot << " teleop: " << request->teleop << "\n";

    auto newStates = SWERVE_KINEMATICS.ToSwerveModuleStates(request->teleop ?
        frc::ChassisSpeeds::FromFieldRelativeSpeeds(units::meters_per_second_t{request->x}, units::meters_per_second_t{request->y}, units::radians_per_second_t{request->rot}, GetYaw()) :
        frc::ChassisSpeeds{units::meters_per_second_t{request->x}, units::meters_per_second_t{request->y}, units::radians_per_second_t{request->rot}}
    );
    frc::SwerveDriveKinematics<4>::DesaturateWheelSpeeds(&newStates, MAX_SPEED);

    for (SwerveModule& mod : modules) {
        mod.SetDesiredState(newStates[mod.moduleNumber], request->open_loop);
    }
}

frc::Rotation2d SwerveNode::GetYaw() {
    return frc::Rotation2d{units::degree_t{360 - gyro.GetAngle()}};
}
void SwerveNode::ZeroGyro() {
    gyro.SetYaw(units::degree_t{0.0});
}
void SwerveNode::ZeroDriveEncoders() {
    for (SwerveModule& mod : modules) {
        mod.ZeroDriveEncoder();
    }
}