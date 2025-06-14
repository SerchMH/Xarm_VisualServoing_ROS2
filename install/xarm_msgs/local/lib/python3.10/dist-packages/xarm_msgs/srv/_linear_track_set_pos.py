# generated from rosidl_generator_py/resource/_idl.py.em
# with input from xarm_msgs:srv/LinearTrackSetPos.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_LinearTrackSetPos_Request(type):
    """Metaclass of message 'LinearTrackSetPos_Request'."""

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
                'xarm_msgs.srv.LinearTrackSetPos_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__linear_track_set_pos__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__linear_track_set_pos__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__linear_track_set_pos__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__linear_track_set_pos__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__linear_track_set_pos__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'SPEED__DEFAULT': 0,
            'WAIT__DEFAULT': True,
            'TIMEOUT__DEFAULT': 100.0,
            'AUTO_ENABLE__DEFAULT': True,
        }

    @property
    def SPEED__DEFAULT(cls):
        """Return default value for message field 'speed'."""
        return 0

    @property
    def WAIT__DEFAULT(cls):
        """Return default value for message field 'wait'."""
        return True

    @property
    def TIMEOUT__DEFAULT(cls):
        """Return default value for message field 'timeout'."""
        return 100.0

    @property
    def AUTO_ENABLE__DEFAULT(cls):
        """Return default value for message field 'auto_enable'."""
        return True


class LinearTrackSetPos_Request(metaclass=Metaclass_LinearTrackSetPos_Request):
    """Message class 'LinearTrackSetPos_Request'."""

    __slots__ = [
        '_pos',
        '_speed',
        '_wait',
        '_timeout',
        '_auto_enable',
    ]

    _fields_and_field_types = {
        'pos': 'int16',
        'speed': 'int16',
        'wait': 'boolean',
        'timeout': 'float',
        'auto_enable': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.pos = kwargs.get('pos', int())
        self.speed = kwargs.get(
            'speed', LinearTrackSetPos_Request.SPEED__DEFAULT)
        self.wait = kwargs.get(
            'wait', LinearTrackSetPos_Request.WAIT__DEFAULT)
        self.timeout = kwargs.get(
            'timeout', LinearTrackSetPos_Request.TIMEOUT__DEFAULT)
        self.auto_enable = kwargs.get(
            'auto_enable', LinearTrackSetPos_Request.AUTO_ENABLE__DEFAULT)

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
        if self.pos != other.pos:
            return False
        if self.speed != other.speed:
            return False
        if self.wait != other.wait:
            return False
        if self.timeout != other.timeout:
            return False
        if self.auto_enable != other.auto_enable:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def pos(self):
        """Message field 'pos'."""
        return self._pos

    @pos.setter
    def pos(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'pos' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'pos' field must be an integer in [-32768, 32767]"
        self._pos = value

    @builtins.property
    def speed(self):
        """Message field 'speed'."""
        return self._speed

    @speed.setter
    def speed(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'speed' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'speed' field must be an integer in [-32768, 32767]"
        self._speed = value

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
    def auto_enable(self):
        """Message field 'auto_enable'."""
        return self._auto_enable

    @auto_enable.setter
    def auto_enable(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'auto_enable' field must be of type 'bool'"
        self._auto_enable = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_LinearTrackSetPos_Response(type):
    """Metaclass of message 'LinearTrackSetPos_Response'."""

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
                'xarm_msgs.srv.LinearTrackSetPos_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__linear_track_set_pos__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__linear_track_set_pos__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__linear_track_set_pos__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__linear_track_set_pos__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__linear_track_set_pos__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class LinearTrackSetPos_Response(metaclass=Metaclass_LinearTrackSetPos_Response):
    """Message class 'LinearTrackSetPos_Response'."""

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


class Metaclass_LinearTrackSetPos(type):
    """Metaclass of service 'LinearTrackSetPos'."""

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
                'xarm_msgs.srv.LinearTrackSetPos')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__linear_track_set_pos

            from xarm_msgs.srv import _linear_track_set_pos
            if _linear_track_set_pos.Metaclass_LinearTrackSetPos_Request._TYPE_SUPPORT is None:
                _linear_track_set_pos.Metaclass_LinearTrackSetPos_Request.__import_type_support__()
            if _linear_track_set_pos.Metaclass_LinearTrackSetPos_Response._TYPE_SUPPORT is None:
                _linear_track_set_pos.Metaclass_LinearTrackSetPos_Response.__import_type_support__()


class LinearTrackSetPos(metaclass=Metaclass_LinearTrackSetPos):
    from xarm_msgs.srv._linear_track_set_pos import LinearTrackSetPos_Request as Request
    from xarm_msgs.srv._linear_track_set_pos import LinearTrackSetPos_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
