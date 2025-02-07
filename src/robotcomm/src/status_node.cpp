//
// Created by team204 on 2/3/25.
//

#include "robotcomm/status_node.h"


RobotStatePublisher::RobotStatePublisher() :
Node{"robot_state_publisher"},
ctx{1},
sock{ctx, zmq::socket_type::sub}
{
    mode_publisher = this->create_publisher<std_msgs::msg::String>("/robotcomm/mode", 10);
    alliance_publisher = this->create_publisher<std_msgs::msg::String>("/robotcomm/alliance", 10);
    dsconnection_publisher = this->create_publisher<std_msgs::msg::Bool>("robotcomm/ds", 10);
    fmsconnection_publisher = this->create_publisher<std_msgs::msg::Bool>("robotcomm/fms", 10);

    sock.bind("tcp://0.0.0.0:5801");
    sock.set(zmq::sockopt::subscribe, "");

    auto defaultModeMsg = std_msgs::msg::String();
    defaultModeMsg.data = "unknown";
    mode_publisher->publish(defaultModeMsg);

    auto defaultAllianceMsg = std_msgs::msg::String();
    defaultAllianceMsg.data = "none";
    alliance_publisher->publish(defaultAllianceMsg);

    auto defaultConnMsg = std_msgs::msg::Bool();
    defaultConnMsg.data = false;
    dsconnection_publisher->publish(defaultConnMsg);
    fmsconnection_publisher->publish(defaultConnMsg);

    RCLCPP_INFO(this->get_logger(), "listening for robot state messages");

    while (true) {
        zmq::message_t zmsg;
        auto res = sock.recv(zmsg, zmq::recv_flags::none);
        if (res.has_value()) {
            auto state = msgpack::unpack<RobotState>(zmsg.data<uint8_t>(), zmsg.size());

            if (state.Mode != cached_mode) {
                auto msg = std_msgs::msg::String();
                switch (state.Mode) {
                    case(0):
                        msg.data = "unknown";
                    break;
                    case(1):
                        msg.data = "disabled";
                    break;
                    case(2):
                        msg.data = "autonomous";
                    break;
                    case(3):
                        msg.data = "teleop";
                    break;
                    case(4):
                        msg.data = "test";
                    break;
                    case(5):
                        msg.data = "estop";
                    break;
                    default:
                        msg.data = "unknown";
                    break;
                }
                std::string logText = "robot mode changed to ";
                logText += msg.data;
                RCLCPP_INFO(this->get_logger(), logText.c_str()); // stupid fawk
                mode_publisher->publish(msg);
                cached_mode = state.Mode;
            }

            if (state.Alliance != cached_alliance) {
                auto msg = std_msgs::msg::String();
                switch (state.Alliance) {
                    case (0):
                        msg.data = "none";
                    break;
                    case (1):
                        msg.data = "blue";
                    break;
                    case (2):
                        msg.data = "red";
                    break;
                    default:
                        msg.data = "none";
                    break;
                }
                std::string logText = "alliance color changed to ";
                logText += msg.data;
                RCLCPP_INFO(this->get_logger(), logText.c_str());
                alliance_publisher->publish(msg);
                cached_alliance = state.Alliance;
            }

            if (state.DSConnected != cached_dsconnected) {
                auto msg = std_msgs::msg::Bool();
                msg.data = state.DSConnected;
                std::string logText = "ds connection state changed to ";
                logText += std::to_string(msg.data);
                RCLCPP_INFO(this->get_logger(), logText.c_str());
                dsconnection_publisher->publish(msg);
                cached_dsconnected = state.DSConnected;
            }

            if (state.FMSConnected != cached_fmsconnected) {
                auto msg = std_msgs::msg::Bool();
                msg.data = state.FMSConnected;
                std::string logText = "fms connection state changed to ";
                logText += std::to_string(msg.data);
                RCLCPP_INFO(this->get_logger(), logText.c_str());
                fmsconnection_publisher->publish(msg);
                cached_fmsconnected = state.FMSConnected;
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "received invalid response from zmq 5801");
        }
    }
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RobotStatePublisher>());
    rclcpp::shutdown();
    return 0;
}
