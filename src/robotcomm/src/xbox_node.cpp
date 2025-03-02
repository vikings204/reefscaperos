//
// Created by team204 on 2/3/25.
//

#include "robotcomm/xbox_node.h"


XboxControllerPublisher::XboxControllerPublisher() :
Node{"xbox_controller_publisher"},
ctx{1},
sock{ctx, zmq::socket_type::sub}
{
    driver_publisher = this->create_publisher<vikings_msgs::msg::Xbox>("/robotcomm/driver", 10);
    operator_publisher = this->create_publisher<vikings_msgs::msg::Xbox>("/robotcomm/operator", 10);

    sock.bind("tcp://0.0.0.0:5802");
    sock.set(zmq::sockopt::subscribe, "");

    RCLCPP_INFO(this->get_logger(), "listening for xbox controller messages");

    while (true) {
        zmq::message_t zmsg;
        auto res = sock.recv(zmsg, zmq::recv_flags::none);
        if (res.has_value()) {
            auto state = msgpack::unpack<ControlUpdate>(zmsg.data<uint8_t>(), zmsg.size());

            auto driverMsg = vikings_msgs::msg::Xbox();
            driverMsg.a = state.DRIVER.A;
            driverMsg.b = state.DRIVER.B;
            driverMsg.x = state.DRIVER.X;
            driverMsg.y = state.DRIVER.Y;
            driverMsg.lb = state.DRIVER.LB;
            driverMsg.rb = state.DRIVER.RB;
            driverMsg.back = state.DRIVER.BACK;
            driverMsg.start = state.DRIVER.START;
            driverMsg.lsb = state.DRIVER.LSB;
            driverMsg.rsb = state.DRIVER.RSB;
            driverMsg.lsx = state.DRIVER.LSX;
            driverMsg.lsy = state.DRIVER.LSY;
            driverMsg.rsx = state.DRIVER.RSX;
            driverMsg.rsy = state.DRIVER.RSY;
            driverMsg.pov = state.DRIVER.POV;
            driverMsg.lt = state.DRIVER.LT;
            driverMsg.rt = state.DRIVER.RT;

            auto operatorMsg = vikings_msgs::msg::Xbox();
            operatorMsg.a = state.OPERATOR.A;
            operatorMsg.b = state.OPERATOR.B;
            operatorMsg.x = state.OPERATOR.X;
            operatorMsg.y = state.OPERATOR.Y;
            operatorMsg.lb = state.OPERATOR.LB;
            operatorMsg.rb = state.OPERATOR.RB;
            operatorMsg.back = state.OPERATOR.BACK;
            operatorMsg.start = state.OPERATOR.START;
            operatorMsg.lsb = state.OPERATOR.LSB;
            operatorMsg.rsb = state.OPERATOR.RSB;
            operatorMsg.lsx = state.OPERATOR.LSX;
            operatorMsg.lsy = state.OPERATOR.LSY;
            operatorMsg.rsx = state.OPERATOR.RSX;
            operatorMsg.rsy = state.OPERATOR.RSY;
            operatorMsg.pov = state.OPERATOR.POV;
            operatorMsg.lt = state.OPERATOR.LT;
            operatorMsg.rt = state.OPERATOR.RT;

            driver_publisher->publish(driverMsg);
            operator_publisher->publish(operatorMsg);
        } else {
            RCLCPP_ERROR(this->get_logger(), "received invalid response from zmq 5802");
        }
    }
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<XboxControllerPublisher>());
    rclcpp::shutdown();
    return 0;
}
