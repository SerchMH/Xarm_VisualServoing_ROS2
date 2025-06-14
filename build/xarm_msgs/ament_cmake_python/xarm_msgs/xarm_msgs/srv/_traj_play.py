# generated from rosidl_generator_py/resource/_idl.py.em
# with input from xarm_msgs:srv/TrajPlay.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_TrajPlay_Request(type):
    """Metaclass of message 'TrajPlay_Request'."""

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
                'xarm_msgs.srv.TrajPlay_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__traj_play__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__traj_play__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__traj_play__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__traj_play__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__traj_play__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'TIMES__DEFAULT': 1,
            'DOUBLE_SPEED__DEFAULT': 1,
            'WAIT__DEFAULT': False,
        }

    @property
    def TIMES__DEFAULT(cls):
        """Return default value for message field 'times'."""
        return 1

    @property
    def DOUBLE_SPEED__DEFAULT(cls):
        """Return default value for message field 'double_speed'."""
        return 1

    @property
    def WAIT__DEFAULT(cls):
        """Return default value for message field 'wait'."""
        return False


class TrajPlay_Request(metaclass=Metaclass_TrajPlay_Request):
    """Message class 'TrajPlay_Request'."""

    __slots__ = [
        '_filename',
        '_times',
        '_double_speed',
        '_wait',
    ]

    _fields_and_field_types = {
        'filename': 'string',
        'times': 'int16',
        'double_speed': 'int16',
        'wait': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.filename = kwargs.get('filename', str())
        self.times = kwargs.get(
            'times', TrajPlay_Request.TIMES__DEFAULT)
        self.double_speed = kwargs.get(
            'double_speed', TrajPlay_Request.DOUBLE_SPEED__DEFAULT)
        self.wait = kwargs.get(
            'wait', TrajPlay_Request.WAIT__DEFAULT)

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
        if self.filename != other.filename:
            return False
        if self.times != other.times:
            return False
        if self.double_speed != other.double_speed:
            return False
        if self.wait != other.wait:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def filename(self):
        """Message field 'filename'."""
        return self._filename

    @filename.setter
    def filename(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'filename' field must be of type 'str'"
        self._filename = value

    @builtins.property
    def times(self):
        """Message field 'times'."""
        return self._times

    @times.setter
    def times(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'times' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'times' field must be an integer in [-32768, 32767]"
        self._times = value

    @builtins.property
    def double_speed(self):
        """Message field 'double_speed'."""
        return self._double_speed

    @double_speed.setter
    def double_speed(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'double_speed' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'double_speed' field must be an integer in [-32768, 32767]"
        self._double_speed = value

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


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_TrajPlay_Response(type):
    """Metaclass of message 'TrajPlay_Response'."""

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
                'xarm_msgs.srv.TrajPlay_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__traj_play__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__traj_play__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__traj_play__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__traj_play__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__traj_play__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class TrajPlay_Response(metaclass=Metaclass_TrajPlay_Response):
    """Message class 'TrajPlay_Response'."""

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


class Metaclass_TrajPlay(type):
    """Metaclass of service 'TrajPlay'."""

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
                'xarm_msgs.srv.TrajPlay')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__traj_play

            from xarm_msgs.srv import _traj_play
            if _traj_play.Metaclass_TrajPlay_Request._TYPE_SUPPORT is None:
                _traj_play.Metaclass_TrajPlay_Request.__import_type_support__()
            if _traj_play.Metaclass_TrajPlay_Response._TYPE_SUPPORT is None:
                _traj_play.Metaclass_TrajPlay_Response.__import_type_support__()


class TrajPlay(metaclass=Metaclass_TrajPlay):
    from xarm_msgs.srv._traj_play import TrajPlay_Request as Request
    from xarm_msgs.srv._traj_play import TrajPlay_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
