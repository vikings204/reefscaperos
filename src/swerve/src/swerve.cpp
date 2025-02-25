#include "swerve/swerve.h"
#include <iostream>
#include <rev/SparkMax.h>
#include <chrono>
#include <thread>
#include <hal/HALBase.h>

int main(int argc, char ** argv)
{
  (void) argc;
  (void) argv;

  HAL_Initialize(500, 0);
  rev::spark::SparkMax motor{43, rev::spark::SparkLowLevel::MotorType::kBrushless};

  std::cout << "motor bus voltage: " << motor.GetBusVoltage() << "\n";

  motor.Set(0.3);
  std::cout << "set speed to 0.3\n";
  std::this_thread::sleep_for(std::literals::chrono_literals::operator ""s(10));
  motor.Set(0);
  std::cout << "set speed to 0\n";

  HAL_Shutdown();
  return 0;
}
