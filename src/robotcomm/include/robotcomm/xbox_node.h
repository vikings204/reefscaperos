//
// Created by team204 on 2/3/25.
//

#ifndef XBOX_NODE_H
#define XBOX_NODE_H

#include <memory>

#include "zmq.hpp"
#include "rclcpp/rclcpp.hpp"
#include "msgpack/msgpack.hpp"
#include "vikings_msgs/msg/xbox.hpp"

class XboxControllerPublisher : public rclcpp::Node {
public:
    XboxControllerPublisher();

private:
    rclcpp::Publisher<vikings_msgs::msg::Xbox>::SharedPtr driver_publisher;
    rclcpp::Publisher<vikings_msgs::msg::Xbox>::SharedPtr operator_publisher;
    zmq::context_t ctx;
    zmq::socket_t sock;

    struct ControllerState {
        bool A;
        bool B;
        bool X;
        bool Y;
        bool LB;
        bool RB;
        bool BACK;
        bool START;
        bool LSB;
        bool RSB;
        double LSX;
        double LSY;
        double RSX;
        double RSY;
        int POV;
        double LT;
        double RT;

        template<class T>
        void pack(T &pack) {
            pack(A, B, X, Y, LB, RB, BACK, START, LSB, RSB, LSX, LSY, RSX, RSY, POV, LT, RT);
        }
    };

    struct ControlUpdate {
        ControllerState DRIVER;
        ControllerState OPERATOR;

        template<class T>
        void pack(T &pack) {
            pack(DRIVER, OPERATOR);
        }
    };
};

#endif //XBOX_NODE_H