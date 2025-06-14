# generated from rosidl_generator_py/resource/_idl.py.em
# with input from xarm_msgs:srv/FtCaliLoad.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'datas'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_FtCaliLoad_Request(type):
    """Metaclass of message 'FtCaliLoad_Request'."""

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
                'xarm_msgs.srv.FtCaliLoad_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__ft_cali_load__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__ft_cali_load__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__ft_cali_load__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__ft_cali_load__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__ft_cali_load__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'ASSOCIATION_SETTING_TCP_LOAD__DEFAULT': False,
            'M__DEFAULT': 0.325,
            'X__DEFAULT': -17.0,
            'Y__DEFAULT': 9.0,
            'Z__DEFAULT': 11.8,
        }

    @property
    def ASSOCIATION_SETTING_TCP_LOAD__DEFAULT(cls):
        """Return default value for message field 'association_setting_tcp_load'."""
        return False

    @property
    def M__DEFAULT(cls):
        """Return default value for message field 'm'."""
        return 0.325

    @property
    def X__DEFAULT(cls):
        """Return default value for message field 'x'."""
        return -17.0

    @property
    def Y__DEFAULT(cls):
        """Return default value for message field 'y'."""
        return 9.0

    @property
    def Z__DEFAULT(cls):
        """Return default value for message field 'z'."""
        return 11.8


class FtCaliLoad_Request(metaclass=Metaclass_FtCaliLoad_Request):
    """Message class 'FtCaliLoad_Request'."""

    __slots__ = [
        '_datas',
        '_association_setting_tcp_load',
        '_m',
        '_x',
        '_y',
        '_z',
    ]

    _fields_and_field_types = {
        'datas': 'sequence<float>',
        'association_setting_tcp_load': 'boolean',
        'm': 'float',
        'x': 'float',
        'y': 'float',
        'z': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.datas = array.array('f', kwargs.get('datas', []))
        self.association_setting_tcp_load = kwargs.get(
            'association_setting_tcp_load', FtCaliLoad_Request.ASSOCIATION_SETTING_TCP_LOAD__DEFAULT)
        self.m = kwargs.get(
            'm', FtCaliLoad_Request.M__DEFAULT)
        self.x = kwargs.get(
            'x', FtCaliLoad_Request.X__DEFAULT)
        self.y = kwargs.get(
            'y', FtCaliLoad_Request.Y__DEFAULT)
        self.z = kwargs.get(
            'z', FtCaliLoad_Request.Z__DEFAULT)

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
        if self.datas != other.datas:
            return False
        if self.association_setting_tcp_load != other.association_setting_tcp_load:
            return False
        if self.m != other.m:
            return False
        if self.x != other.x:
            return False
        if self.y != other.y:
            return False
        if self.z != other.z:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def datas(self):
        """Message field 'datas'."""
        return self._datas

    @datas.setter
    def datas(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'datas' array.array() must have the type code of 'f'"
            self._datas = value
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
                "The 'datas' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._datas = array.array('f', value)

    @builtins.property
    def association_setting_tcp_load(self):
        """Message field 'association_setting_tcp_load'."""
        return self._association_setting_tcp_load

    @association_setting_tcp_load.setter
    def association_setting_tcp_load(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'association_setting_tcp_load' field must be of type 'bool'"
        self._association_setting_tcp_load = value

    @builtins.property
    def m(self):
        """Message field 'm'."""
        return self._m

    @m.setter
    def m(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'm' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'm' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._m = value

    @builtins.property
    def x(self):
        """Message field 'x'."""
        return self._x

    @x.setter
    def x(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'x' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'x' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._x = value

    @builtins.property
    def y(self):
        """Message field 'y'."""
        return self._y

    @y.setter
    def y(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'y' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'y' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._y = value

    @builtins.property
    def z(self):
        """Message field 'z'."""
        return self._z

    @z.setter
    def z(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'z' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'z' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._z = value


# Import statements for member types

# already imported above
# import builtins

# already imported above
# import rosidl_parser.definition


class Metaclass_FtCaliLoad_Response(type):
    """Metaclass of message 'FtCaliLoad_Response'."""

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
                'xarm_msgs.srv.FtCaliLoad_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__ft_cali_load__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__ft_cali_load__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__ft_cali_load__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__ft_cali_load__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__ft_cali_load__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class FtCaliLoad_Response(metaclass=Metaclass_FtCaliLoad_Response):
    """Message class 'FtCaliLoad_Response'."""

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


class Metaclass_FtCaliLoad(type):
    """Metaclass of service 'FtCaliLoad'."""

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
                'xarm_msgs.srv.FtCaliLoad')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__ft_cali_load

            from xarm_msgs.srv import _ft_cali_load
            if _ft_cali_load.Metaclass_FtCaliLoad_Request._TYPE_SUPPORT is None:
                _ft_cali_load.Metaclass_FtCaliLoad_Request.__import_type_support__()
            if _ft_cali_load.Metaclass_FtCaliLoad_Response._TYPE_SUPPORT is None:
                _ft_cali_load.Metaclass_FtCaliLoad_Response.__import_type_support__()


class FtCaliLoad(metaclass=Metaclass_FtCaliLoad):
    from xarm_msgs.srv._ft_cali_load import FtCaliLoad_Request as Request
    from xarm_msgs.srv._ft_cali_load import FtCaliLoad_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
