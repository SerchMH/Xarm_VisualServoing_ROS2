# generated from rosidl_generator_py/resource/_idl.py.em
# with input from xarm_msgs:srv/GetSetModbusData.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'modbus_data'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_GetSetModbusData_Request(type):
    """Metaclass of message 'GetSetModbusData_Request'."""

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
                'xarm_msgs.srv.GetSetModbusData_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__get_set_modbus_data__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__get_set_modbus_data__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__get_set_modbus_data__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__get_set_modbus_data__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__get_set_modbus_data__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'MODBUS_LENGTH__DEFAULT': 0,
            'HOST_ID__DEFAULT': 9,
            'IS_TRANSPARENT_TRANSMISSION__DEFAULT': False,
            'USE_503_PORT__DEFAULT': False,
        }

    @property
    def MODBUS_LENGTH__DEFAULT(cls):
        """Return default value for message field 'modbus_length'."""
        return 0

    @property
    def HOST_ID__DEFAULT(cls):
        """Return default value for message field 'host_id'."""
        return 9

    @property
    def IS_TRANSPARENT_TRANSMISSION__DEFAULT(cls):
        """Return default value for message field 'is_transparent_transmission'."""
        return False

    @property
    def USE_503_PORT__DEFAULT(cls):
        """Return default value for message field 'use_503_port'."""
        return False


class GetSetModbusData_Request(metaclass=Metaclass_GetSetModbusData_Request):
    """Message class 'GetSetModbusData_Request'."""

    __slots__ = [
        '_modbus_data',
        '_modbus_length',
        '_ret_length',
        '_host_id',
        '_is_transparent_transmission',
        '_use_503_port',
    ]

    _fields_and_field_types = {
        'modbus_data': 'sequence<uint8>',
        'modbus_length': 'int16',
        'ret_length': 'int16',
        'host_id': 'uint8',
        'is_transparent_transmission': 'boolean',
        'use_503_port': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('uint8')),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.modbus_data = array.array('B', kwargs.get('modbus_data', []))
        self.modbus_length = kwargs.get(
            'modbus_length', GetSetModbusData_Request.MODBUS_LENGTH__DEFAULT)
        self.ret_length = kwargs.get('ret_length', int())
        self.host_id = kwargs.get(
            'host_id', GetSetModbusData_Request.HOST_ID__DEFAULT)
        self.is_transparent_transmission = kwargs.get(
            'is_transparent_transmission', GetSetModbusData_Request.IS_TRANSPARENT_TRANSMISSION__DEFAULT)
        self.use_503_port = kwargs.get(
            'use_503_port', GetSetModbusData_Request.USE_503_PORT__DEFAULT)

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
        if self.modbus_data != other.modbus_data:
            return False
        if self.modbus_length != other.modbus_length:
            return False
        if self.ret_length != other.ret_length:
            return False
        if self.host_id != other.host_id:
            return False
        if self.is_transparent_transmission != other.is_transparent_transmission:
            return False
        if self.use_503_port != other.use_503_port:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def modbus_data(self):
        """Message field 'modbus_data'."""
        return self._modbus_data

    @modbus_data.setter
    def modbus_data(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'B', \
                "The 'modbus_data' array.array() must have the type code of 'B'"
            self._modbus_data = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'modbus_data' field must be a set or sequence and each value of type 'int' and each unsigned integer in [0, 255]"
        self._modbus_data = array.array('B', value)

    @builtins.property
    def modbus_length(self):
        """Message field 'modbus_length'."""
        return self._modbus_length

    @modbus_length.setter
    def modbus_length(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'modbus_length' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'modbus_length' field must be an integer in [-32768, 32767]"
        self._modbus_length = value

    @builtins.property
    def ret_length(self):
        """Message field 'ret_length'."""
        return self._ret_length

    @ret_length.setter
    def ret_length(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'ret_length' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'ret_length' field must be an integer in [-32768, 32767]"
        self._ret_length = value

    @builtins.property
    def host_id(self):
        """Message field 'host_id'."""
        return self._host_id

    @host_id.setter
    def host_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'host_id' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'host_id' field must be an unsigned integer in [0, 255]"
        self._host_id = value

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

    @builtins.property
    def use_503_port(self):
        """Message field 'use_503_port'."""
        return self._use_503_port

    @use_503_port.setter
    def use_503_port(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'use_503_port' field must be of type 'bool'"
        self._use_503_port = value


# Import statements for member types

# Member 'ret_data'
# already imported above
# import array

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_GetSetModbusData_Response(type):
    """Metaclass of message 'GetSetModbusData_Response'."""

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
                'xarm_msgs.srv.GetSetModbusData_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__get_set_modbus_data__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__get_set_modbus_data__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__get_set_modbus_data__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__get_set_modbus_data__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__get_set_modbus_data__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class GetSetModbusData_Response(metaclass=Metaclass_GetSetModbusData_Response):
    """Message class 'GetSetModbusData_Response'."""

    __slots__ = [
        '_ret',
        '_message',
        '_ret_data',
    ]

    _fields_and_field_types = {
        'ret': 'int16',
        'message': 'string',
        'ret_data': 'sequence<uint8>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('uint8')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.ret = kwargs.get('ret', int())
        self.message = kwargs.get('message', str())
        self.ret_data = array.array('B', kwargs.get('ret_data', []))

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
        if self.ret_data != other.ret_data:
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

    @builtins.property
    def ret_data(self):
        """Message field 'ret_data'."""
        return self._ret_data

    @ret_data.setter
    def ret_data(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'B', \
                "The 'ret_data' array.array() must have the type code of 'B'"
            self._ret_data = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'ret_data' field must be a set or sequence and each value of type 'int' and each unsigned integer in [0, 255]"
        self._ret_data = array.array('B', value)


class Metaclass_GetSetModbusData(type):
    """Metaclass of service 'GetSetModbusData'."""

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
                'xarm_msgs.srv.GetSetModbusData')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__get_set_modbus_data

            from xarm_msgs.srv import _get_set_modbus_data
            if _get_set_modbus_data.Metaclass_GetSetModbusData_Request._TYPE_SUPPORT is None:
                _get_set_modbus_data.Metaclass_GetSetModbusData_Request.__import_type_support__()
            if _get_set_modbus_data.Metaclass_GetSetModbusData_Response._TYPE_SUPPORT is None:
                _get_set_modbus_data.Metaclass_GetSetModbusData_Response.__import_type_support__()


class GetSetModbusData(metaclass=Metaclass_GetSetModbusData):
    from xarm_msgs.srv._get_set_modbus_data import GetSetModbusData_Request as Request
    from xarm_msgs.srv._get_set_modbus_data import GetSetModbusData_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
