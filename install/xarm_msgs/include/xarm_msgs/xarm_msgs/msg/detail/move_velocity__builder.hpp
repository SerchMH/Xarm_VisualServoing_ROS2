// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from xarm_msgs:msg/MoveVelocity.idl
// generated code does not contain a copyright notice

#ifndef XARM_MSGS__MSG__DETAIL__MOVE_VELOCITY__BUILDER_HPP_
#define XARM_MSGS__MSG__DETAIL__MOVE_VELOCITY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "xarm_msgs/msg/detail/move_velocity__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace xarm_msgs
{

namespace msg
{

namespace builder
{

class Init_MoveVelocity_duration
{
public:
  explicit Init_MoveVelocity_duration(::xarm_msgs::msg::MoveVelocity & msg)
  : msg_(msg)
  {}
  ::xarm_msgs::msg::MoveVelocity duration(::xarm_msgs::msg::MoveVelocity::_duration_type arg)
  {
    msg_.duration = std::move(arg);
    return std::move(msg_);
  }

private:
  ::xarm_msgs::msg::MoveVelocity msg_;
};

class Init_MoveVelocity_is_tool_coord
{
public:
  explicit Init_MoveVelocity_is_tool_coord(::xarm_msgs::msg::MoveVelocity & msg)
  : msg_(msg)
  {}
  Init_MoveVelocity_duration is_tool_coord(::xarm_msgs::msg::MoveVelocity::_is_tool_coord_type arg)
  {
    msg_.is_tool_coord = std::move(arg);
    return Init_MoveVelocity_duration(msg_);
  }

private:
  ::xarm_msgs::msg::MoveVelocity msg_;
};

class Init_MoveVelocity_is_sync
{
public:
  explicit Init_MoveVelocity_is_sync(::xarm_msgs::msg::MoveVelocity & msg)
  : msg_(msg)
  {}
  Init_MoveVelocity_is_tool_coord is_sync(::xarm_msgs::msg::MoveVelocity::_is_sync_type arg)
  {
    msg_.is_sync = std::move(arg);
    return Init_MoveVelocity_is_tool_coord(msg_);
  }

private:
  ::xarm_msgs::msg::MoveVelocity msg_;
};

class Init_MoveVelocity_speeds
{
public:
  Init_MoveVelocity_speeds()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MoveVelocity_is_sync speeds(::xarm_msgs::msg::MoveVelocity::_speeds_type arg)
  {
    msg_.speeds = std::move(arg);
    return Init_MoveVelocity_is_sync(msg_);
  }

private:
  ::xarm_msgs::msg::MoveVelocity msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::xarm_msgs::msg::MoveVelocity>()
{
  return xarm_msgs::msg::builder::Init_MoveVelocity_speeds();
}

}  // namespace xarm_msgs

#endif  // XARM_MSGS__MSG__DETAIL__MOVE_VELOCITY__BUILDER_HPP_
