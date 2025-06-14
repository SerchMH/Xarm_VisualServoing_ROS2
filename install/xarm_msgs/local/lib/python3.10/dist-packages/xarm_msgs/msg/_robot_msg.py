# generated from rosidl_generator_py/resource/_idl.py.em
# with input from xarm_msgs:msg/RobotMsg.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'angle'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'pose'
# Member 'offset'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RobotMsg(type):
    """Metaclass of message 'RobotMsg'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('xarm_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'xarm_msgs.msg.RobotMsg')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__robot_msg
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__robot_msg
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__robot_msg
            cls._TYPE_SUPPORT = module.type_support_msg__msg__robot_msg
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__robot_msg

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class RobotMsg(metaclass=Metaclass_RobotMsg):
    """Message class 'RobotMsg'."""

    __slots__ = [
        '_header',
        '_state',
        '_mode',
        '_cmdnum',
        '_mt_brake',
        '_mt_able',
        '_err',
        '_warn',
        '_angle',
        '_pose',
        '_offset',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'state': 'int16',
        'mode': 'int16',
        'cmdnum': 'int16',
        'mt_brake': 'int16',
        'mt_able': 'int16',
        'err': 'int16',
        'warn': 'int16',
        'angle': 'sequence<float>',
        'pose': 'float[6]',
        'offset': 'float[6]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 6),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 6),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.state = kwargs.get('state', int())
        self.mode = kwargs.get('mode', int())
        self.cmdnum = kwargs.get('cmdnum', int())
        self.mt_brake = kwargs.get('mt_brake', int())
        self.mt_able = kwargs.get('mt_able', int())
        self.err = kwargs.get('err', int())
        self.warn = kwargs.get('warn', int())
        self.angle = array.array('f', kwargs.get('angle', []))
        if 'pose' not in kwargs:
            self.pose = numpy.zeros(6, dtype=numpy.float32)
        else:
            self.pose = numpy.array(kwargs.get('pose'), dtype=numpy.float32)
            assert self.pose.shape == (6, )
        if 'offset' not in kwargs:
            self.offset = numpy.zeros(6, dtype=numpy.float32)
        else:
            self.offset = numpy.array(kwargs.get('offset'), dtype=numpy.float32)
            assert self.offset.shape == (6, )

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.state != other.state:
            return False
        if self.mode != other.mode:
            return False
        if self.cmdnum != other.cmdnum:
            return False
        if self.mt_brake != other.mt_brake:
            return False
        if self.mt_able != other.mt_able:
            return False
        if self.err != other.err:
            return False
        if self.warn != other.warn:
            return False
        if self.angle != other.angle:
            return False
        if all(self.pose != other.pose):
            return False
        if all(self.offset != other.offset):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def state(self):
        """Message field 'state'."""
        return self._state

    @state.setter
    def state(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'state' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'state' field must be an integer in [-32768, 32767]"
        self._state = value

    @builtins.property
    def mode(self):
        """Message field 'mode'."""
        return self._mode

    @mode.setter
    def mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mode' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'mode' field must be an integer in [-32768, 32767]"
        self._mode = value

    @builtins.property
    def cmdnum(self):
        """Message field 'cmdnum'."""
        return self._cmdnum

    @cmdnum.setter
    def cmdnum(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'cmdnum' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'cmdnum' field must be an integer in [-32768, 32767]"
        self._cmdnum = value

    @builtins.property
    def mt_brake(self):
        """Message field 'mt_brake'."""
        return self._mt_brake

    @mt_brake.setter
    def mt_brake(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mt_brake' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'mt_brake' field must be an integer in [-32768, 32767]"
        self._mt_brake = value

    @builtins.property
    def mt_able(self):
        """Message field 'mt_able'."""
        return self._mt_able

    @mt_able.setter
    def mt_able(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mt_able' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'mt_able' field must be an integer in [-32768, 32767]"
        self._mt_able = value

    @builtins.property
    def err(self):
        """Message field 'err'."""
        return self._err

    @err.setter
    def err(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'err' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'err' field must be an integer in [-32768, 32767]"
        self._err = value

    @builtins.property
    def warn(self):
        """Message field 'warn'."""
        return self._warn

    @warn.setter
    def warn(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'warn' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'warn' field must be an integer in [-32768, 32767]"
        self._warn = value

    @builtins.property
    def angle(self):
        """Message field 'angle'."""
        return self._angle

    @angle.setter
    def angle(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'angle' array.array() must have the type code of 'f'"
            self._angle = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'angle' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._angle = array.array('f', value)

    @builtins.property
    def pose(self):
        """Message field 'pose'."""
        return self._pose

    @pose.setter
    def pose(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'pose' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 6, \
                "The 'pose' numpy.ndarray() must have a size of 6"
            self._pose = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 6 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'pose' field must be a set or sequence with length 6 and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._pose = numpy.array(value, dtype=numpy.float32)

    @builtins.property
    def offset(self):
        """Message field 'offset'."""
        return self._offset

    @offset.setter
    def offset(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'offset' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 6, \
                "The 'offset' numpy.ndarray() must have a size of 6"
            self._offset = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 6 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'offset' field must be a set or sequence with length 6 and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._offset = numpy.array(value, dtype=numpy.float32)
