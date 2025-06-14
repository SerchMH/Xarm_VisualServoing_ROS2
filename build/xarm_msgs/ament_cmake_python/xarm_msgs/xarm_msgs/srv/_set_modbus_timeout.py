# generated from rosidl_generator_py/resource/_idl.py.em
# with input from xarm_msgs:srv/SetModbusTimeout.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SetModbusTimeout_Request(type):
    """Metaclass of message 'SetModbusTimeout_Request'."""

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
                'xarm_msgs.srv.SetModbusTimeout_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_modbus_timeout__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_modbus_timeout__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_modbus_timeout__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_modbus_timeout__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_modbus_timeout__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'IS_TRANSPARENT_TRANSMISSION__DEFAULT': False,
        }

    @property
    def IS_TRANSPARENT_TRANSMISSION__DEFAULT(cls):
        """Return default value for message field 'is_transparent_transmission'."""
        return False


class SetModbusTimeout_Request(metaclass=Metaclass_SetModbusTimeout_Request):
    """Message class 'SetModbusTimeout_Request'."""

    __slots__ = [
        '_timeout',
        '_is_transparent_transmission',
    ]

    _fields_and_field_types = {
        'timeout': 'int32',
        'is_transparent_transmission': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.timeout = kwargs.get('timeout', int())
        self.is_transparent_transmission = kwargs.get(
            'is_transparent_transmission', SetModbusTimeout_Request.IS_TRANSPARENT_TRANSMISSION__DEFAULT)

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
        if self.timeout != other.timeout:
            return False
        if self.is_transparent_transmission != other.is_transparent_transmission:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def timeout(self):
        """Message field 'timeout'."""
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'timeout' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'timeout' field must be an integer in [-2147483648, 2147483647]"
        self._timeout = value

    @builtins.property
    def is_transparent_transmission(self):
        """Message field 'is_transparent_transmission'."""
        return self._is_transparent_transmission

    @is_transparent_transmission.setter
    def is_transparent_transmission(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'is_transparent_transmission' field must be of type 'bool'"
        self._is_transparent_transmission = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_SetModbusTimeout_Response(type):
    """Metaclass of message 'SetModbusTimeout_Response'."""

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
                'xarm_msgs.srv.SetModbusTimeout_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__set_modbus_timeout__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__set_modbus_timeout__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__set_modbus_timeout__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__set_modbus_timeout__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__set_modbus_timeout__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SetModbusTimeout_Response(metaclass=Metaclass_SetModbusTimeout_Response):
    """Message class 'SetModbusTimeout_Response'."""

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


class Metaclass_SetModbusTimeout(type):
    """Metaclass of service 'SetModbusTimeout'."""

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
                'xarm_msgs.srv.SetModbusTimeout')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__set_modbus_timeout

            from xarm_msgs.srv import _set_modbus_timeout
            if _set_modbus_timeout.Metaclass_SetModbusTimeout_Request._TYPE_SUPPORT is None:
                _set_modbus_timeout.Metaclass_SetModbusTimeout_Request.__import_type_support__()
            if _set_modbus_timeout.Metaclass_SetModbusTimeout_Response._TYPE_SUPPORT is None:
                _set_modbus_timeout.Metaclass_SetModbusTimeout_Response.__import_type_support__()


class SetModbusTimeout(metaclass=Metaclass_SetModbusTimeout):
    from xarm_msgs.srv._set_modbus_timeout import SetModbusTimeout_Request as Request
    from xarm_msgs.srv._set_modbus_timeout import SetModbusTimeout_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
