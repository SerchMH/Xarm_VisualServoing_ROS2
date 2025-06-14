# generated from rosidl_generator_py/resource/_idl.py.em
# with input from xarm_msgs:srv/MoveCircle.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'pose1'
# Member 'pose2'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_MoveCircle_Request(type):
    """Metaclass of message 'MoveCircle_Request'."""

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
                'xarm_msgs.srv.MoveCircle_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__move_circle__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__move_circle__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__move_circle__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__move_circle__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__move_circle__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'SPEED__DEFAULT': 0.0,
            'ACC__DEFAULT': 0.0,
            'MVTIME__DEFAULT': 0.0,
            'WAIT__DEFAULT': False,
            'TIMEOUT__DEFAULT': -1.0,
            'IS_TOOL_COORD__DEFAULT': False,
            'IS_AXIS_ANGLE__DEFAULT': False,
        }

    @property
    def SPEED__DEFAULT(cls):
        """Return default value for message field 'speed'."""
        return 0.0

    @property
    def ACC__DEFAULT(cls):
        """Return default value for message field 'acc'."""
        return 0.0

    @property
    def MVTIME__DEFAULT(cls):
        """Return default value for message field 'mvtime'."""
        return 0.0

    @property
    def WAIT__DEFAULT(cls):
        """Return default value for message field 'wait'."""
        return False

    @property
    def TIMEOUT__DEFAULT(cls):
        """Return default value for message field 'timeout'."""
        return -1.0

    @property
    def IS_TOOL_COORD__DEFAULT(cls):
        """Return default value for message field 'is_tool_coord'."""
        return False

    @property
    def IS_AXIS_ANGLE__DEFAULT(cls):
        """Return default value for message field 'is_axis_angle'."""
        return False


class MoveCircle_Request(metaclass=Metaclass_MoveCircle_Request):
    """Message class 'MoveCircle_Request'."""

    __slots__ = [
        '_pose1',
        '_pose2',
        '_percent',
        '_speed',
        '_acc',
        '_mvtime',
        '_wait',
        '_timeout',
        '_is_tool_coord',
        '_is_axis_angle',
    ]

    _fields_and_field_types = {
        'pose1': 'sequence<float>',
        'pose2': 'sequence<float>',
        'percent': 'float',
        'speed': 'float',
        'acc': 'float',
        'mvtime': 'float',
        'wait': 'boolean',
        'timeout': 'float',
        'is_tool_coord': 'boolean',
        'is_axis_angle': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.pose1 = array.array('f', kwargs.get('pose1', []))
        self.pose2 = array.array('f', kwargs.get('pose2', []))
        self.percent = kwargs.get('percent', float())
        self.speed = kwargs.get(
            'speed', MoveCircle_Request.SPEED__DEFAULT)
        self.acc = kwargs.get(
            'acc', MoveCircle_Request.ACC__DEFAULT)
        self.mvtime = kwargs.get(
            'mvtime', MoveCircle_Request.MVTIME__DEFAULT)
        self.wait = kwargs.get(
            'wait', MoveCircle_Request.WAIT__DEFAULT)
        self.timeout = kwargs.get(
            'timeout', MoveCircle_Request.TIMEOUT__DEFAULT)
        self.is_tool_coord = kwargs.get(
            'is_tool_coord', MoveCircle_Request.IS_TOOL_COORD__DEFAULT)
        self.is_axis_angle = kwargs.get(
            'is_axis_angle', MoveCircle_Request.IS_AXIS_ANGLE__DEFAULT)

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
        if self.pose1 != other.pose1:
            return False
        if self.pose2 != other.pose2:
            return False
        if self.percent != other.percent:
            return False
        if self.speed != other.speed:
            return False
        if self.acc != other.acc:
            return False
        if self.mvtime != other.mvtime:
            return False
        if self.wait != other.wait:
            return False
        if self.timeout != other.timeout:
            return False
        if self.is_tool_coord != other.is_tool_coord:
            return False
        if self.is_axis_angle != other.is_axis_angle:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def pose1(self):
        """Message field 'pose1'."""
        return self._pose1

    @pose1.setter
    def pose1(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'pose1' array.array() must have the type code of 'f'"
            self._pose1 = value
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
                "The 'pose1' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._pose1 = array.array('f', value)

    @builtins.property
    def pose2(self):
        """Message field 'pose2'."""
        return self._pose2

    @pose2.setter
    def pose2(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'pose2' array.array() must have the type code of 'f'"
            self._pose2 = value
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
                "The 'pose2' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._pose2 = array.array('f', value)

    @builtins.property
    def percent(self):
        """Message field 'percent'."""
        return self._percent

    @percent.setter
    def percent(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'percent' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'percent' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._percent = value

    @builtins.property
    def speed(self):
        """Message field 'speed'."""
        return self._speed

    @speed.setter
    def speed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'speed' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'speed' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._speed = value

    @builtins.property
    def acc(self):
        """Message field 'acc'."""
        return self._acc

    @acc.setter
    def acc(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'acc' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'acc' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._acc = value

    @builtins.property
    def mvtime(self):
        """Message field 'mvtime'."""
        return self._mvtime

    @mvtime.setter
    def mvtime(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'mvtime' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'mvtime' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._mvtime = value

    @builtins.property
    def wait(self):
        """Message field 'wait'."""
        return self._wait

    @wait.setter
    def wait(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'wait' field must be of type 'bool'"
        self._wait = value

    @builtins.property
    def timeout(self):
        """Message field 'timeout'."""
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'timeout' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'timeout' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._timeout = value

    @builtins.property
    def is_tool_coord(self):
        """Message field 'is_tool_coord'."""
        return self._is_tool_coord

    @is_tool_coord.setter
    def is_tool_coord(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_tool_coord' field must be of type 'bool'"
        self._is_tool_coord = value

    @builtins.property
    def is_axis_angle(self):
        """Message field 'is_axis_angle'."""
        return self._is_axis_angle

    @is_axis_angle.setter
    def is_axis_angle(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_axis_angle' field must be of type 'bool'"
        self._is_axis_angle = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_MoveCircle_Response(type):
    """Metaclass of message 'MoveCircle_Response'."""

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
                'xarm_msgs.srv.MoveCircle_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__move_circle__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__move_circle__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__move_circle__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__move_circle__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__move_circle__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class MoveCircle_Response(metaclass=Metaclass_MoveCircle_Response):
    """Message class 'MoveCircle_Response'."""

    __slots__ = [
        '_ret',
        '_message',
    ]

    _fields_and_field_types = {
        'ret': 'int16',
        'message': 'string',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.ret = kwargs.get('ret', int())
        self.message = kwargs.get('message', str())

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
        if self.ret != other.ret:
            return False
        if self.message != other.message:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def ret(self):
        """Message field 'ret'."""
        return self._ret

    @ret.setter
    def ret(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'ret' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'ret' field must be an integer in [-32768, 32767]"
        self._ret = value

    @builtins.property
    def message(self):
        """Message field 'message'."""
        return self._message

    @message.setter
    def message(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'message' field must be of type 'str'"
        self._message = value


class Metaclass_MoveCircle(type):
    """Metaclass of service 'MoveCircle'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('xarm_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'xarm_msgs.srv.MoveCircle')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__move_circle

            from xarm_msgs.srv import _move_circle
            if _move_circle.Metaclass_MoveCircle_Request._TYPE_SUPPORT is None:
                _move_circle.Metaclass_MoveCircle_Request.__import_type_support__()
            if _move_circle.Metaclass_MoveCircle_Response._TYPE_SUPPORT is None:
                _move_circle.Metaclass_MoveCircle_Response.__import_type_support__()


class MoveCircle(metaclass=Metaclass_MoveCircle):
    from xarm_msgs.srv._move_circle import MoveCircle_Request as Request
    from xarm_msgs.srv._move_circle import MoveCircle_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
