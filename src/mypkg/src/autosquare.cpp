#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int64.hpp"

class AutoSquare : public rclcpp::Node
{
public:
    AutoSquare()
    : Node("autosquare")
    {
        subscription_ = this->create_subscription<std_msgs::msg::Int64>(
        "topic", 10, [this] (const std_msgs::msg::Int64 msg) { topic_callback(msg); });
    }

private:
    void topic_callback(const std_msgs::msg::Int64 & msg) const
    {
        const long num = msg.data;
        RCLCPP_INFO(this->get_logger(), "I heard: '%ld'", num*num);
    }
    rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AutoSquare>());
    rclcpp::shutdown();
    return 0;
}