#include <iostream>
#include <rev/CANSparkMax.h>
#include <chrono>
#include <thread>
#include <hal/HALBase.h>

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    HAL_Initialize(500, 0);
    //std::cout << "HAL_GetTeamNumber: " << HAL_GetTeamNumber() << "\n";

    rev::CANSparkMax motor{10, rev::CANSparkLowLevel::MotorType::kBrushless};
    //rev::CANSparkLowLevel::SetEnable(true);
    //rev::CANSparkLowLevel::EnableExternalUSBControl(true);

    motor.Set(0.3);
    //rev::CANSparkLowLevel::SetEnable(true);
    std::cout << "set speed to 0.3\n";

    // std::cout << "starting loop\n";
    // for (int i = 0; i < 100000000; i++) {
    //     motor.Set(0.3);
    //     rev::CANSparkLowLevel::SetEnable(true);
    // }
    // std::cout << "finished loop\n";

    std::cout << "waiting 10 seconds\n";
    std::this_thread::sleep_for(std::literals::chrono_literals::operator ""s(10));
    std::cout << "done\n";

    std::cout << motor.GetBusVoltage() << "\n";
    motor.Set(0);

    HAL_Shutdown();
    return 0;
}
