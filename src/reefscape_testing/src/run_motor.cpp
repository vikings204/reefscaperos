//
// Created by team204 on 1/24/25.
//

#include <iostream>
#include <frc/geometry/Rotation2d.h>

int main(int argc, char ** argv) {
    std::cout << "run_motor lmao\n";

    const frc::Rotation2d rot{units::degree_t{360.0}};
    std::cout << rot.Radians().value() << "\n";



    return 0;
}