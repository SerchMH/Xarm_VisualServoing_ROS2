# generated from rosidl_generator_py/resource/_idl.py.em
# with input from xarm_msgs:msg/CIOState.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'input_digitals'
# Member 'output_digitals'
# Member 'input_analogs'
# Member 'output_analogs'
# Member 'input_conf'
# Member 'output_conf'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_CIOState(type):
    """Metaclass of message 'CIOState'."""

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
                'xarm_msgs.msg.CIOState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__cio_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__cio_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__cio_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__cio_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__cio_state

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


class CIOState(metaclass=Metaclass_CIOState):
    """Message class 'CIOState'."""

    __slots__ = [
        '_header',
        '_state',
        '_code',
        '_input_digitals',
        '_output_digitals',
        '_input_analogs',
        '_output_analogs',
        '_input_conf',
        '_output_conf',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'state': 'int16',
        'code': 'int16',
        'input_digitals': 'uint16[2]',
        'output_digitals': 'uint16[2]',
        'input_analogs': 'float[2]',
        'output_analogs': 'float[2]',
        'input_conf': 'uint8[16]',
        'output_conf': 'uint8[16]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.BasicType('int16'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint16'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint16'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 16),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 16),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.state = kwargs.get('state', int())
        self.code = kwargs.get('code', int())
        if 'input_digitals' not in kwargs:
            self.input_digitals = numpy.zeros(2, dtype=numpy.uint16)
        else:
            self.input_digitals = numpy.array(kwargs.get('input_digitals'), dtype=numpy.uint16)
            assert self.input_digitals.shape == (2, )
        if 'output_digitals' not in kwargs:
            self.output_digitals = numpy.zeros(2, dtype=numpy.uint16)
        else:
            self.output_digitals = numpy.array(kwargs.get('output_digitals'), dtype=numpy.uint16)
            assert self.output_digitals.shape == (2, )
        if 'input_analogs' not in kwargs:
            self.input_analogs = numpy.zeros(2, dtype=numpy.float32)
        else:
            self.input_analogs = numpy.array(kwargs.get('input_analogs'), dtype=numpy.float32)
            assert self.input_analogs.shape == (2, )
        if 'output_analogs' not in kwargs:
            self.output_analogs = numpy.zeros(2, dtype=numpy.float32)
        else:
            self.output_analogs = numpy.array(kwargs.get('output_analogs'), dtype=numpy.float32)
            assert self.output_analogs.shape == (2, )
        if 'input_conf' not in kwargs:
            self.input_conf = numpy.zeros(16, dtype=numpy.uint8)
        else:
            self.input_conf = numpy.array(kwargs.get('input_conf'), dtype=numpy.uint8)
            assert self.input_conf.shape == (16, )
        if 'output_conf' not in kwargs:
            self.output_conf = numpy.zeros(16, dtype=numpy.uint8)
        else:
            self.output_conf = numpy.array(kwargs.get('output_conf'), dtype=numpy.uint8)
            assert self.output_conf.shape == (16, )

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
        if self.code != other.code:
            return False
        if all(self.input_digitals != other.input_digitals):
            return False
        if all(self.output_digitals != other.output_digitals):
            return False
        if all(self.input_analogs != other.input_analogs):
            return False
        if all(self.output_analogs != other.output_analogs):
            return False
        if all(self.input_conf != other.input_conf):
            return False
        if all(self.output_conf != other.output_conf):
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
    def code(self):
        """Message field 'code'."""
        return self._code

    @code.setter
    def code(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'code' field must be of type 'int'"
            assert value >= -32768 and value < 32768, \
                "The 'code' field must be an integer in [-32768, 32767]"
        self._code = value

    @builtins.property
    def input_digitals(self):
        """Message field 'input_digitals'."""
        return self._input_digitals

    @input_digitals.setter
    def input_digitals(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint16, \
                "The 'input_digitals' numpy.ndarray() must have the dtype of 'numpy.uint16'"
            assert value.size == 2, \
                "The 'input_digitals' numpy.ndarray() must have a size of 2"
            self._input_digitals = value
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
                 len(value) == 2 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 65536 for val in value)), \
                "The 'input_digitals' field must be a set or sequence with length 2 and each value of type 'int' and each unsigned integer in [0, 65535]"
        self._input_digitals = numpy.array(value, dtype=numpy.uint16)

    @builtins.property
    def output_digitals(self):
        """Message field 'output_digitals'."""
        return self._output_digitals

    @output_digitals.setter
    def output_digitals(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint16, \
                "The 'output_digitals' numpy.ndarray() must have the dtype of 'numpy.uint16'"
            assert value.size == 2, \
                "The 'output_digitals' numpy.ndarray() must have a size of 2"
            self._output_digitals = value
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
                 len(value) == 2 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 65536 for val in value)), \
                "The 'output_digitals' field must be a set or sequence with length 2 and each value of type 'int' and each unsigned integer in [0, 65535]"
        self._output_digitals = numpy.array(value, dtype=numpy.uint16)

    @builtins.property
    def input_analogs(self):
        """Message field 'input_analogs'."""
        return self._input_analogs

    @input_analogs.setter
    def input_analogs(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'input_analogs' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 2, \
                "The 'input_analogs' numpy.ndarray() must have a size of 2"
            self._input_analogs = value
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
                 len(value) == 2 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'input_analogs' field must be a set or sequence with length 2 and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._input_analogs = numpy.array(value, dtype=numpy.float32)

    @builtins.property
    def output_analogs(self):
        """Message field 'output_analogs'."""
        return self._output_analogs

    @output_analogs.setter
    def output_analogs(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'output_analogs' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 2, \
                "The 'output_analogs' numpy.ndarray() must have a size of 2"
            self._output_analogs = value
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
                 len(value) == 2 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'output_analogs' field must be a set or sequence with length 2 and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._output_analogs = numpy.array(value, dtype=numpy.float32)

    @builtins.property
    def input_conf(self):
        """Message field 'input_conf'."""
        return self._input_conf

    @input_conf.setter
    def input_conf(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'input_conf' numpy.ndarray() must have the dtype of 'numpy.uint8'"
            assert value.size == 16, \
                "The 'input_conf' numpy.ndarray() must have a size of 16"
            self._input_conf = value
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
                 len(value) == 16 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'input_conf' field must be a set or sequence with length 16 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._input_conf = numpy.array(value, dtype=numpy.uint8)

    @builtins.property
    def output_conf(self):
        """Message field 'output_conf'."""
        return self._output_conf

    @output_conf.setter
    def output_conf(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'output_conf' numpy.ndarray() must have the dtype of 'numpy.uint8'"
            assert value.size == 16, \
                "The 'output_conf' numpy.ndarray() must have a size of 16"
            self._output_conf = value
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
                 len(value) == 16 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'output_conf' field must be a set or sequence with length 16 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._output_conf = numpy.array(value, dtype=numpy.uint8)
