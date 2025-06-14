# generated from rosidl_generator_py/resource/_idl.py.em
# with input from xarm_msgs:srv/MoveVelocity.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'speeds'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_MoveVelocity_Request(type):
    """Metaclass of message 'MoveVelocity_Request'."""

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
                'xarm_msgs.srv.MoveVelocity_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__move_velocity__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__move_velocity__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__move_velocity__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__move_velocity__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__move_velocity__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'IS_SYNC__DEFAULT': True,
            'IS_TOOL_COORD__DEFAULT': False,
            'DURATION__DEFAULT': -1.0,
        }

    @property
    def IS_SYNC__DEFAULT(cls):
        """Return default value for message field 'is_sync'."""
        return True

    @property
    def IS_TOOL_COORD__DEFAULT(cls):
        """Return default value for message field 'is_tool_coord'."""
        return False

    @property
    def DURATION__DEFAULT(cls):
        """Return default value for message field 'duration'."""
        return -1.0


class MoveVelocity_Request(metaclass=Metaclass_MoveVelocity_Request):
    """Message class 'MoveVelocity_Request'."""

    __slots__ = [
        '_speeds',
        '_is_sync',
        '_is_tool_coord',
        '_duration',
    ]

    _fields_and_field_types = {
        'speeds': 'sequence<float>',
        'is_sync': 'boolean',
        'is_tool_coord': 'boolean',
        'duration': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.speeds = array.array('f', kwargs.get('speeds', []))
        self.is_sync = kwargs.get(
            'is_sync', MoveVelocity_Request.IS_SYNC__DEFAULT)
        self.is_tool_coord = kwargs.get(
            'is_tool_coord', MoveVelocity_Request.IS_TOOL_COORD__DEFAULT)
        self.duration = kwargs.get(
            'duration', MoveVelocity_Request.DURATION__DEFAULT)

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
        if self.speeds != other.speeds:
            return False
        if self.is_sync != other.is_sync:
            return False
        if self.is_tool_coord != other.is_tool_coord:
            return False
        if self.duration != other.duration:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def speeds(self):
        """Message field 'speeds'."""
        return self._speeds

    @speeds.setter
    def speeds(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'speeds' array.array() must have the type code of 'f'"
            self._speeds = value
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
                "The 'speeds' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._speeds = array.array('f', value)

    @builtins.property
    def is_sync(self):
        """Message field 'is_sync'."""
        return self._is_sync

    @is_sync.setter
    def is_sync(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_sync' field must be of type 'bool'"
        self._is_sync = value

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
    def duration(self):
        """Message field 'duration'."""
        return self._duration

    @duration.setter
    def duration(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'duration' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'duration' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._duration = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_MoveVelocity_Response(type):
    """Metaclass of message 'MoveVelocity_Response'."""

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
                'xarm_msgs.srv.MoveVelocity_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__move_velocity__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__move_velocity__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__move_velocity__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__move_velocity__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__move_velocity__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class MoveVelocity_Response(metaclass=Metaclass_MoveVelocity_Response):
    """Message class 'MoveVelocity_Response'."""

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


class Metaclass_MoveVelocity(type):
    """Metaclass of service 'MoveVelocity'."""

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
                'xarm_msgs.srv.MoveVelocity')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__move_velocity

            from xarm_msgs.srv import _move_velocity
            if _move_velocity.Metaclass_MoveVelocity_Request._TYPE_SUPPORT is None:
                _move_velocity.Metaclass_MoveVelocity_Request.__import_type_support__()
            if _move_velocity.Metaclass_MoveVelocity_Response._TYPE_SUPPORT is None:
                _move_velocity.Metaclass_MoveVelocity_Response.__import_type_support__()


class MoveVelocity(metaclass=Metaclass_MoveVelocity):
    from xarm_msgs.srv._move_velocity import MoveVelocity_Request as Request
    from xarm_msgs.srv._move_velocity import MoveVelocity_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
