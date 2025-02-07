#include <mypkg/stupid.h>

std::string getMessage() {
    return "STUPID";
}

int square(int n) {
    return n * n;
}

frc::Rotation2d makeRotation(double rad) {
    return {units::radian_t{rad}};
}