//
// Created by team204 on 3/3/25.
//

#ifndef SWERVE_CONSTANTS_H
#define SWERVE_CONSTANTS_H

namespace Constants {
    namespace Swerve {
        const units::volt_t DRIVE_FF_S {0.667};
        const units::unit_t<units::compound_unit<units::volts, units::inverse<units::meters_per_second>>> DRIVE_FF_V {2.44};
        const units::unit_t<units::compound_unit<units::volts, units::inverse<units::meters_per_second_squared>>> DRIVE_FF_A {0.27};
        const units::meters_per_second_t MAX_SPEED {4.5}; // meters per second
        const float ANGLE_GEAR_RATIO = (150.0 / 7.0);
        const float VOLTAGE_COMPENSATION = 12.0;
        const units::meter_t WHEEL_DIAMETER {4.0_in};
        const float DRIVE_GEAR_RATIO = 8.14;
        const float ANGLE_POSITION_CONVERSION_FACTOR = 1 / ANGLE_GEAR_RATIO;
        const float DRIVE_POSITION_CONVERSION_FACTOR = (WHEEL_DIAMETER.value() * M_PI) / DRIVE_GEAR_RATIO;
        const float DRIVE_VELOCITY_CONVERSION_FACTOR = DRIVE_POSITION_CONVERSION_FACTOR / 60.0;
        const int DRIVE_CURRENT_LIMIT = 40;
        const float DRIVE_PID_P = 1.0;
        const float DRIVE_PID_I = 0.0;
        const float DRIVE_PID_D = 0.01;
        const float DRIVE_PID_FF = 0.0;

        const auto WHEEL_BASE = 23_in;
        const auto TRACK_WIDTH = 23_in;
        const frc::SwerveDriveKinematics SWERVE_KINEMATICS{
                frc::Translation2d{-WHEEL_BASE / 2.0, TRACK_WIDTH / 2.0},
                frc::Translation2d{WHEEL_BASE / 2.0, TRACK_WIDTH / 2.0},
                frc::Translation2d{-WHEEL_BASE / 2.0, -TRACK_WIDTH / 2.0},
                frc::Translation2d{WHEEL_BASE / 2.0, -TRACK_WIDTH / 2.0}
        };

        namespace Mod0 {
            const int DRIVE_MOTOR_ID = 11;
            const int ANGLE_MOTOR_ID = 21;
            const int CANCODER_ID = 31;
            const float ANGLE_OFFSET_ROTS = 0.75;
        }
        namespace Mod1 {
            const int DRIVE_MOTOR_ID = 12;
            const int ANGLE_MOTOR_ID = 22;
            const int CANCODER_ID = 32;
            const float ANGLE_OFFSET_ROTS = 0.888;
        }
        namespace Mod2 {
            const int DRIVE_MOTOR_ID = 10;
            const int ANGLE_MOTOR_ID = 20;
            const int CANCODER_ID = 30;
            const float ANGLE_OFFSET_ROTS = 0.887;
        }
        namespace Mod3 {
            const int DRIVE_MOTOR_ID = 13;
            const int ANGLE_MOTOR_ID = 23;
            const int CANCODER_ID = 33;
            const float ANGLE_OFFSET_ROTS = 0.08;
        }
    }
}

#endif //SWERVE_CONSTANTS_H
