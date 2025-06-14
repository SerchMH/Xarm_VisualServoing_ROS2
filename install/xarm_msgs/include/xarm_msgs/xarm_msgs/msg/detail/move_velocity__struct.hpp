// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from xarm_msgs:msg/MoveVelocity.idl
// generated code does not contain a copyright notice

#ifndef XARM_MSGS__MSG__DETAIL__MOVE_VELOCITY__STRUCT_HPP_
#define XARM_MSGS__MSG__DETAIL__MOVE_VELOCITY__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__xarm_msgs__msg__MoveVelocity __attribute__((deprecated))
#else
# define DEPRECATED__xarm_msgs__msg__MoveVelocity __declspec(deprecated)
#endif

namespace xarm_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MoveVelocity_
{
  using Type = MoveVelocity_<ContainerAllocator>;

  explicit MoveVelocity_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::DEFAULTS_ONLY == _init)
    {
      this->is_sync = true;
      this->is_tool_coord = false;
      this->duration = -1.0f;
    } else if (rosidl_runtime_cpp::MessageInitialization::ZERO == _init) {
      this->is_sync = false;
      this->is_tool_coord = false;
      this->duration = 0.0f;
    }
  }

  explicit MoveVelocity_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::DEFAULTS_ONLY == _init)
    {
      this->is_sync = true;
      this->is_tool_coord = false;
      this->duration = -1.0f;
    } else if (rosidl_runtime_cpp::MessageInitialization::ZERO == _init) {
      this->is_sync = false;
      this->is_tool_coord = false;
      this->duration = 0.0f;
    }
  }

  // field types and members
  using _speeds_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _speeds_type speeds;
  using _is_sync_type =
    bool;
  _is_sync_type is_sync;
  using _is_tool_coord_type =
    bool;
  _is_tool_coord_type is_tool_coord;
  using _duration_type =
    float;
  _duration_type duration;

  // setters for named parameter idiom
  Type & set__speeds(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->speeds = _arg;
    return *this;
  }
  Type & set__is_sync(
    const bool & _arg)
  {
    this->is_sync = _arg;
    return *this;
  }
  Type & set__is_tool_coord(
    const bool & _arg)
  {
    this->is_tool_coord = _arg;
    return *this;
  }
  Type & set__duration(
    const float & _arg)
  {
    this->duration = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    xarm_msgs::msg::MoveVelocity_<ContainerAllocator> *;
  using ConstRawPtr =
    const xarm_msgs::msg::MoveVelocity_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<xarm_msgs::msg::MoveVelocity_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<xarm_msgs::msg::MoveVelocity_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      xarm_msgs::msg::MoveVelocity_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<xarm_msgs::msg::MoveVelocity_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      xarm_msgs::msg::MoveVelocity_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<xarm_msgs::msg::MoveVelocity_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<xarm_msgs::msg::MoveVelocity_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<xarm_msgs::msg::MoveVelocity_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__xarm_msgs__msg__MoveVelocity
    std::shared_ptr<xarm_msgs::msg::MoveVelocity_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__xarm_msgs__msg__MoveVelocity
    std::shared_ptr<xarm_msgs::msg::MoveVelocity_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MoveVelocity_ & other) const
  {
    if (this->speeds != other.speeds) {
      return false;
    }
    if (this->is_sync != other.is_sync) {
      return false;
    }
    if (this->is_tool_coord != other.is_tool_coord) {
      return false;
    }
    if (this->duration != other.duration) {
      return false;
    }
    return true;
  }
  bool operator!=(const MoveVelocity_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MoveVelocity_

// alias to use template instance with default allocator
using MoveVelocity =
  xarm_msgs::msg::MoveVelocity_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace xarm_msgs

#endif  // XARM_MSGS__MSG__DETAIL__MOVE_VELOCITY__STRUCT_HPP_
