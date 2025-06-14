// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from xarm_msgs:msg/IOState.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "xarm_msgs/msg/detail/io_state__struct.h"
#include "xarm_msgs/msg/detail/io_state__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool xarm_msgs__msg__io_state__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[32];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("xarm_msgs.msg._io_state.IOState", full_classname_dest, 31) == 0);
  }
  xarm_msgs__msg__IOState * ros_message = _ros_message;
  {  // digital_1
    PyObject * field = PyObject_GetAttrString(_pymsg, "digital_1");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->digital_1 = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // digital_2
    PyObject * field = PyObject_GetAttrString(_pymsg, "digital_2");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->digital_2 = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // analog_1
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_1");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->analog_1 = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // analog_2
    PyObject * field = PyObject_GetAttrString(_pymsg, "analog_2");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->analog_2 = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * xarm_msgs__msg__io_state__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of IOState */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("xarm_msgs.msg._io_state");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "IOState");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  xarm_msgs__msg__IOState * ros_message = (xarm_msgs__msg__IOState *)raw_ros_message;
  {  // digital_1
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->digital_1);
    {
      int rc = PyObject_SetAttrString(_pymessage, "digital_1", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // digital_2
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->digital_2);
    {
      int rc = PyObject_SetAttrString(_pymessage, "digital_2", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_1
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->analog_1);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_1", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // analog_2
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->analog_2);
    {
      int rc = PyObject_SetAttrString(_pymessage, "analog_2", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
