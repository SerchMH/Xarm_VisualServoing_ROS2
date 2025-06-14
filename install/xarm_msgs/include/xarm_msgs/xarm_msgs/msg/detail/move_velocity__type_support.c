// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from xarm_msgs:msg/MoveVelocity.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "xarm_msgs/msg/detail/move_velocity__rosidl_typesupport_introspection_c.h"
#include "xarm_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "xarm_msgs/msg/detail/move_velocity__functions.h"
#include "xarm_msgs/msg/detail/move_velocity__struct.h"


// Include directives for member types
// Member `speeds`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__MoveVelocity_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  xarm_msgs__msg__MoveVelocity__init(message_memory);
}

void xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__MoveVelocity_fini_function(void * message_memory)
{
  xarm_msgs__msg__MoveVelocity__fini(message_memory);
}

size_t xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__size_function__MoveVelocity__speeds(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__get_const_function__MoveVelocity__speeds(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__get_function__MoveVelocity__speeds(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__fetch_function__MoveVelocity__speeds(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__get_const_function__MoveVelocity__speeds(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__assign_function__MoveVelocity__speeds(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__get_function__MoveVelocity__speeds(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__resize_function__MoveVelocity__speeds(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__MoveVelocity_message_member_array[4] = {
  {
    "speeds",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(xarm_msgs__msg__MoveVelocity, speeds),  // bytes offset in struct
    NULL,  // default value
    xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__size_function__MoveVelocity__speeds,  // size() function pointer
    xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__get_const_function__MoveVelocity__speeds,  // get_const(index) function pointer
    xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__get_function__MoveVelocity__speeds,  // get(index) function pointer
    xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__fetch_function__MoveVelocity__speeds,  // fetch(index, &value) function pointer
    xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__assign_function__MoveVelocity__speeds,  // assign(index, value) function pointer
    xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__resize_function__MoveVelocity__speeds  // resize(index) function pointer
  },
  {
    "is_sync",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(xarm_msgs__msg__MoveVelocity, is_sync),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "is_tool_coord",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(xarm_msgs__msg__MoveVelocity, is_tool_coord),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "duration",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(xarm_msgs__msg__MoveVelocity, duration),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__MoveVelocity_message_members = {
  "xarm_msgs__msg",  // message namespace
  "MoveVelocity",  // message name
  4,  // number of fields
  sizeof(xarm_msgs__msg__MoveVelocity),
  xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__MoveVelocity_message_member_array,  // message members
  xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__MoveVelocity_init_function,  // function to initialize message memory (memory has to be allocated)
  xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__MoveVelocity_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__MoveVelocity_message_type_support_handle = {
  0,
  &xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__MoveVelocity_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_xarm_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, xarm_msgs, msg, MoveVelocity)() {
  if (!xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__MoveVelocity_message_type_support_handle.typesupport_identifier) {
    xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__MoveVelocity_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &xarm_msgs__msg__MoveVelocity__rosidl_typesupport_introspection_c__MoveVelocity_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
