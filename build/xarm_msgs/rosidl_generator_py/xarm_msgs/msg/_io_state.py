# generated from rosidl_generator_py/resource/_idl.py.em
# with input from xarm_msgs:msg/IOState.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_IOState(type):
    """Metaclass of message 'IOState'."""

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
                'xarm_msgs.msg.IOState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__io_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__io_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__io_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__io_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__io_state

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class IOState(metaclass=Metaclass_IOState):
    """Message class 'IOState'."""

    __slots__ = [
        '_digital_1',
        '_digital_2',
        '_analog_1',
        '_analog_2',
    ]

    _fields_and_field_types = {
        'digital_1': 'int32',
        'digital_2': 'int32',
        'analog_1': 'float',
        'analog_2': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.digital_1 = kwargs.get('digital_1', int())
        self.digital_2 = kwargs.get('digital_2', int())
        self.analog_1 = kwargs.get('analog_1', float())
        self.analog_2 = kwargs.get('analog_2', float())

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
        if self.digital_1 != other.digital_1:
            return False
        if self.digital_2 != other.digital_2:
            return False
        if self.analog_1 != other.analog_1:
            return False
        if self.analog_2 != other.analog_2:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def digital_1(self):
        """Message field 'digital_1'."""
        return self._digital_1

    @digital_1.setter
    def digital_1(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'digital_1' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'digital_1' field must be an integer in [-2147483648, 2147483647]"
        self._digital_1 = value

    @builtins.property
    def digital_2(self):
        """Message field 'digital_2'."""
        return self._digital_2

    @digital_2.setter
    def digital_2(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'digital_2' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'digital_2' field must be an integer in [-2147483648, 2147483647]"
        self._digital_2 = value

    @builtins.property
    def analog_1(self):
        """Message field 'analog_1'."""
        return self._analog_1

    @analog_1.setter
    def analog_1(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'analog_1' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'analog_1' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._analog_1 = value

    @builtins.property
    def analog_2(self):
        """Message field 'analog_2'."""
        return self._analog_2

    @analog_2.setter
    def analog_2(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'analog_2' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'analog_2' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._analog_2 = value
