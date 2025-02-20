//
// Created by team204 on 2/3/25.
//

#ifndef STATUS_NODE_H
#define STATUS_NODE_H

#include <memory>

#include "zmq.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"
#include "msgpack/msgpack.hpp"

class RobotStatePublisher : public rclcpp::Node {
public:
    RobotStatePublisher();

private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mode_publisher;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr alliance_publisher;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr dsconnection_publisher;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr fmsconnection_publisher;
    zmq::context_t ctx;
    zmq::socket_t sock;

    struct RobotState {
        int Mode;
        int Alliance;
        bool DSConnected;
        bool FMSConnected;

        template<class T>
        void pack(T &pack) {
            pack(Mode, Alliance, DSConnected, FMSConnected);
        }
    };

    const int MODE_UNKNOWN = 0;
    const int MODE_DISABLED = 1;
    const int MODE_AUTONOMOUS = 2;
    const int MODE_TELEOP = 3;
    const int MODE_TEST = 4;
    const int MODE_ESTOP = 5;

    const int ALLIANCE_NONE = 0;
    const int ALLIANCE_BLUE = 1;
    const int ALLIANCE_RED = 2;

    int cached_mode = MODE_UNKNOWN;
    int cached_alliance = ALLIANCE_NONE;
    int cached_dsconnected = false;
    int cached_fmsconnected = false;
};

#endif //STATUS_NODE_H