#include <iostream>
#include <mypkg/stupid.h>
//#include <rev/CANSparkMax.h>

int main(int argc, char ** argv)
{
  (void) argc;
  (void) argv;

  std::cout << "hello world mypkg package\n";
  std::cout << getMessage() << "\n";

  int num;
  std::cout << "Enter a number: ";
  std::cin >> num;
  std::cout << square(num) << "\n";
  std::cout << makeRotation(3.14159263).Degrees().value() << "\n";
  //std::cout << rev::CANSparkMax::MotorType::kBrushless;


  return 0;
}
