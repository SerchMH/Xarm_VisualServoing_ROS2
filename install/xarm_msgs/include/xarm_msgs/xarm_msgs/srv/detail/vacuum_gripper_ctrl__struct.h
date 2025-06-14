// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from xarm_msgs:srv/VacuumGripperCtrl.idl
// generated code does not contain a copyright notice

#ifndef XARM_MSGS__SRV__DETAIL__VACUUM_GRIPPER_CTRL__STRUCT_H_
#define XARM_MSGS__SRV__DETAIL__VACUUM_GRIPPER_CTRL__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/VacuumGripperCtrl in the package xarm_msgs.
typedef struct xarm_msgs__srv__VacuumGripperCtrl_Request
{
  bool on;
  bool wait;
  float timeout;
  float delay_sec;
  /// sync: whether to execute in the motion queue
  bool sync;
  /// hardware_version==1: Plug-in Connection, default
  /// hardware_version==2: Contact Connection
  /// default is 0, equivalent to hardware_version==1 at first
  /// when the hardware_version parameter is specified as 1 or 2, the value will be recorded in the current environment.
  /// If the parameter is not specified in the next call or call the get_vacuum_gripper, the previous value will be used.
  int16_t hardware_version;
} xarm_msgs__srv__VacuumGripperCtrl_Request;

// Struct for a sequence of xarm_msgs__srv__VacuumGripperCtrl_Request.
typedef struct xarm_msgs__srv__VacuumGripperCtrl_Request__Sequence
{
  xarm_msgs__srv__VacuumGripperCtrl_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} xarm_msgs__srv__VacuumGripperCtrl_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/VacuumGripperCtrl in the package xarm_msgs.
typedef struct xarm_msgs__srv__VacuumGripperCtrl_Response
{
  int16_t ret;
  rosidl_runtime_c__String message;
} xarm_msgs__srv__VacuumGripperCtrl_Response;

// Struct for a sequence of xarm_msgs__srv__VacuumGripperCtrl_Response.
typedef struct xarm_msgs__srv__VacuumGripperCtrl_Response__Sequence
{
  xarm_msgs__srv__VacuumGripperCtrl_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} xarm_msgs__srv__VacuumGripperCtrl_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // XARM_MSGS__SRV__DETAIL__VACUUM_GRIPPER_CTRL__STRUCT_H_
