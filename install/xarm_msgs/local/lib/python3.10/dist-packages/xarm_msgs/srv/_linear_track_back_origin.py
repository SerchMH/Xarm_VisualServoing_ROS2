# generated from rosidl_generator_py/resource/_idl.py.em
# with input from xarm_msgs:srv/LinearTrackBackOrigin.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_LinearTrackBackOrigin_Request(type):
    """Metaclass of message 'LinearTrackBackOrigin_Request'."""

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
                'xarm_msgs.srv.LinearTrackBackOrigin_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__linear_track_back_origin__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__linear_track_back_origin__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__linear_track_back_origin__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__linear_track_back_origin__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__linear_track_back_origin__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'WAIT__DEFAULT': True,
            'AUTO_ENABLE__DEFAULT': True,
        }

    @property
    def WAIT__DEFAULT(cls):
        """Return default value for message field 'wait'."""
        return True

    @property
    def AUTO_ENABLE__DEFAULT(cls):
        """Return default value for message field 'auto_enable'."""
        return True


class LinearTrackBackOrigin_Request(metaclass=Metaclass_LinearTrackBackOrigin_Request):
    """Message class 'LinearTrackBackOrigin_Request'."""

    __slots__ = [
        '_wait',
        '_auto_enable',
    ]

    _fields_and_field_types = {
        'wait': 'boolean',
        'auto_enable': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.wait = kwargs.get(
            'wait', LinearTrackBackOrigin_Request.WAIT__DEFAULT)
        self.auto_enable = kwargs.get(
            'auto_enable', LinearTrackBackOrigin_Request.AUTO_ENABLE__DEFAULT)

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
        if self.wait != other.wait:
            return False
        if self.auto_enable != other.auto_enable:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

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


class Metaclass_LinearTrackBackOrigin_Response(type):
    """Metaclass of message 'LinearTrackBackOrigin_Response'."""

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
                'xarm_msgs.srv.LinearTrackBackOrigin_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__linear_track_back_origin__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__linear_track_back_origin__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__linear_track_back_origin__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__linear_track_back_origin__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__linear_track_back_origin__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class LinearTrackBackOrigin_Response(metaclass=Metaclass_LinearTrackBackOrigin_Response):
    """Message class 'LinearTrackBackOrigin_Response'."""

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


class Metaclass_LinearTrackBackOrigin(type):
    """Metaclass of service 'LinearTrackBackOrigin'."""

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
                'xarm_msgs.srv.LinearTrackBackOrigin')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__linear_track_back_origin

            from xarm_msgs.srv import _linear_track_back_origin
            if _linear_track_back_origin.Metaclass_LinearTrackBackOrigin_Request._TYPE_SUPPORT is None:
                _linear_track_back_origin.Metaclass_LinearTrackBackOrigin_Request.__import_type_support__()
            if _linear_track_back_origin.Metaclass_LinearTrackBackOrigin_Response._TYPE_SUPPORT is None:
                _linear_track_back_origin.Metaclass_LinearTrackBackOrigin_Response.__import_type_support__()


class LinearTrackBackOrigin(metaclass=Metaclass_LinearTrackBackOrigin):
    from xarm_msgs.srv._linear_track_back_origin import LinearTrackBackOrigin_Request as Request
    from xarm_msgs.srv._linear_track_back_origin import LinearTrackBackOrigin_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
