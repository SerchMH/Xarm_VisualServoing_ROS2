// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from xarm_msgs:msg/MoveVelocity.idl
// generated code does not contain a copyright notice
#include "xarm_msgs/msg/detail/move_velocity__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `speeds`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
xarm_msgs__msg__MoveVelocity__init(xarm_msgs__msg__MoveVelocity * msg)
{
  if (!msg) {
    return false;
  }
  // speeds
  if (!rosidl_runtime_c__float__Sequence__init(&msg->speeds, 0)) {
    xarm_msgs__msg__MoveVelocity__fini(msg);
    return false;
  }
  // is_sync
  msg->is_sync = true;
  // is_tool_coord
  msg->is_tool_coord = false;
  // duration
  msg->duration = -1.0f;
  return true;
}

void
xarm_msgs__msg__MoveVelocity__fini(xarm_msgs__msg__MoveVelocity * msg)
{
  if (!msg) {
    return;
  }
  // speeds
  rosidl_runtime_c__float__Sequence__fini(&msg->speeds);
  // is_sync
  // is_tool_coord
  // duration
}

bool
xarm_msgs__msg__MoveVelocity__are_equal(const xarm_msgs__msg__MoveVelocity * lhs, const xarm_msgs__msg__MoveVelocity * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // speeds
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->speeds), &(rhs->speeds)))
  {
    return false;
  }
  // is_sync
  if (lhs->is_sync != rhs->is_sync) {
    return false;
  }
  // is_tool_coord
  if (lhs->is_tool_coord != rhs->is_tool_coord) {
    return false;
  }
  // duration
  if (lhs->duration != rhs->duration) {
    return false;
  }
  return true;
}

bool
xarm_msgs__msg__MoveVelocity__copy(
  const xarm_msgs__msg__MoveVelocity * input,
  xarm_msgs__msg__MoveVelocity * output)
{
  if (!input || !output) {
    return false;
  }
  // speeds
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->speeds), &(output->speeds)))
  {
    return false;
  }
  // is_sync
  output->is_sync = input->is_sync;
  // is_tool_coord
  output->is_tool_coord = input->is_tool_coord;
  // duration
  output->duration = input->duration;
  return true;
}

xarm_msgs__msg__MoveVelocity *
xarm_msgs__msg__MoveVelocity__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  xarm_msgs__msg__MoveVelocity * msg = (xarm_msgs__msg__MoveVelocity *)allocator.allocate(sizeof(xarm_msgs__msg__MoveVelocity), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(xarm_msgs__msg__MoveVelocity));
  bool success = xarm_msgs__msg__MoveVelocity__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
xarm_msgs__msg__MoveVelocity__destroy(xarm_msgs__msg__MoveVelocity * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    xarm_msgs__msg__MoveVelocity__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
xarm_msgs__msg__MoveVelocity__Sequence__init(xarm_msgs__msg__MoveVelocity__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  xarm_msgs__msg__MoveVelocity * data = NULL;

  if (size) {
    data = (xarm_msgs__msg__MoveVelocity *)allocator.zero_allocate(size, sizeof(xarm_msgs__msg__MoveVelocity), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = xarm_msgs__msg__MoveVelocity__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        xarm_msgs__msg__MoveVelocity__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
xarm_msgs__msg__MoveVelocity__Sequence__fini(xarm_msgs__msg__MoveVelocity__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      xarm_msgs__msg__MoveVelocity__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

xarm_msgs__msg__MoveVelocity__Sequence *
xarm_msgs__msg__MoveVelocity__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  xarm_msgs__msg__MoveVelocity__Sequence * array = (xarm_msgs__msg__MoveVelocity__Sequence *)allocator.allocate(sizeof(xarm_msgs__msg__MoveVelocity__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = xarm_msgs__msg__MoveVelocity__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
xarm_msgs__msg__MoveVelocity__Sequence__destroy(xarm_msgs__msg__MoveVelocity__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    xarm_msgs__msg__MoveVelocity__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
xarm_msgs__msg__MoveVelocity__Sequence__are_equal(const xarm_msgs__msg__MoveVelocity__Sequence * lhs, const xarm_msgs__msg__MoveVelocity__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!xarm_msgs__msg__MoveVelocity__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
xarm_msgs__msg__MoveVelocity__Sequence__copy(
  const xarm_msgs__msg__MoveVelocity__Sequence * input,
  xarm_msgs__msg__MoveVelocity__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(xarm_msgs__msg__MoveVelocity);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    xarm_msgs__msg__MoveVelocity * data =
      (xarm_msgs__msg__MoveVelocity *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!xarm_msgs__msg__MoveVelocity__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          xarm_msgs__msg__MoveVelocity__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!xarm_msgs__msg__MoveVelocity__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
