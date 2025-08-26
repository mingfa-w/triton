import os
from typing import List, Sequence, Optional, Union

from triton._C.libtriton import ir
from triton.language import semantic as real_semantic
from triton.language.core import (
    _constexpr_to_value,
    _tensor_member_fn,
    _unwrap_iterable,
    builtin,
    constexpr,
    dtype as real_dtype,
    float32,
    tensor,
    check_bit_width,
    _unwrap_if_constexpr,
    range,
)
from typing import Optional
# from triton.language.core import _unwrap_if_constexpr, _unwrap_shape
from . import semantic
from .tensor_descriptor import tensor_descriptor, tensor_descriptor_base

# from ._utils import validate_block_shape

# class dtype(real_dtype):

#     def to_ir(self, builder: ir.builder) -> ir.type:
#         if self.name in ("uint8", "uint16", "uint32", "uint64"):
#             raise ValueError(f"type {self} not supported in this architecture for now.")

#         if self.name.startswith("fp8"):
#             if self.name not in builder.options.supported_fp8_dtypes:
#                 raise ValueError(f'type {self} not supported in this architecture. '
#                                  f'The supported fp8 dtypes are {builder.options.supported_fp8_dtypes}')
#             if self.name in builder.options.deprecated_fp8_dtypes:
#                 warn(f"{self.name} is deprecated in this architecture and will be removed in a future triton release")

#         if self.name == 'void':
#             return builder.get_void_ty()
#         elif self.name == 'int1':
#             return builder.get_int1_ty()
#         elif self.name in ('int8', 'uint8'):
#             return builder.get_int8_ty()
#         elif self.name in ('int16', 'uint16'):
#             return builder.get_int16_ty()
#         elif self.name in ('int32', 'uint32'):
#             return builder.get_int32_ty()
#         elif self.name in ('int64', 'uint64'):
#             return builder.get_int64_ty()
#         elif self.name == 'fp8e5':
#             return builder.get_fp8e5_ty()
#         elif self.name == 'fp8e5b16':
#             return builder.get_fp8e5b16_ty()
#         elif self.name == 'fp8e4nv':
#             return builder.get_fp8e4nv_ty()
#         elif self.name == 'fp8e4b8':
#             return builder.get_fp8e4b8_ty()
#         elif self.name == 'fp8e4b15':
#             return builder.get_fp8e4b15_ty()
#         elif self.name == 'fp16':
#             return builder.get_half_ty()
#         elif self.name == 'bf16':
#             return builder.get_bf16_ty()
#         elif self.name == 'fp32':
#             return builder.get_float_ty()
#         elif self.name == 'fp64':
#             return builder.get_double_ty()
#         raise ValueError(f'fail to convert {self} to ir type')

# class pointer_type(dtype):

#     def __init__(self, element_ty: dtype, address_space: int = 1, const: bool = False):
#         element_ty = _unwrap_if_constexpr(element_ty)
#         if not isinstance(element_ty, dtype):
#             raise TypeError(f'element_ty has type `{type(element_ty).__name__}`; expected `dtype`.')
#         self.element_ty = element_ty
#         self.address_space = address_space
#         self.const = const
#         self.name = f'pointer<{element_ty}>' if not const else f'const_pointer<{element_ty}>'

#     def to_ir(self, builder: ir.builder):
#         return builder.get_ptr_ty(self.element_ty.to_ir(builder), self.address_space)

#     def __str__(self):
#         return self.name

#     def __repr__(self):
#         return self.__str__()

#     def is_ptr(self):
#         return True

#     def is_const(self):
#         return self.const

#     def __eq__(self, other: pointer_type) -> bool:
#         if not isinstance(other, pointer_type):
#             return False
#         return self.element_ty == other.element_ty and self.address_space == other.address_space and self.const == other.const

#     def __ne__(self, other: pointer_type) -> bool:
#         return not self.__eq__(other)

#     @property
#     def scalar(self):
#         return self

# class block_type(dtype):

#     def __init__(self, element_ty: dtype, shape: List):
#         self.element_ty = element_ty

#         # Note that block_type's shape is a list of int
#         # while tensor's shape is a list of constexpr.

#         # shape can be empty ([]) when an input is a 0D tensor.
#         self.shape = _unwrap_shape(shape)
#         if not self.shape:
#             raise TypeError('0d block_type is forbidden')

#         self.numel = validate_block_shape(self.shape)
#         self.name = f'<{self.shape}, {self.element_ty}>'

#     def to_ir(self, builder: ir.builder) -> ir.block_type:
#         return builder.get_block_ty(self.element_ty.to_ir(builder), self.shape)

#     def __str__(self):
#         return self.name

#     def __repr__(self):
#         return self.__str__()

#     def is_block(self):
#         return True

#     def get_block_shapes(self) -> List[int]:
#         return self.shape

#     def __eq__(self, other: block_type) -> bool:
#         if not isinstance(other, block_type):
#             return False
#         return self.element_ty == other.element_ty and self.shape == other.shape

#     def __ne__(self, other: block_type) -> bool:
#         return not self.__eq__(other)

#     @property
#     def scalar(self):
#         return self.element_ty

# class function_type(dtype):

#     def __init__(self, ret_types: List[dtype], param_types: List[dtype]) -> None:
#         self.ret_types = ret_types
#         self.param_types = param_types

#     def __str__(self):
#         return f'fn ({self.param_types}) -> {self.ret_types}'

#     def to_ir(self, builder: ir.builder):
#         ir_param_types = [ty.to_ir(builder) for ty in self.param_types]
#         ret_types = [ret_type.to_ir(builder) for ret_type in self.ret_types]
#         return builder.get_function_ty(ir_param_types, ret_types)

@_tensor_member_fn
@builtin
def cast(input, dtype: real_dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, overflow_mode: Optional[str] = None, _builder=None):
    """
    Casts a tensor to the given :code:`dtype`.

    :param dtype: The target data type.
    :type dtype: tl.dtype
    :param fp_downcast_rounding: The rounding mode for downcasting
        floating-point values. This parameter is only used when self is a
        floating-point tensor and dtype is a floating-point type with a
        smaller bitwidth. Supported values are :code:`"rtne"` (round to
        nearest, ties to even) and :code:`"rtz"` (round towards zero).
    :type fp_downcast_rounding: str, optional
    :param bitcast: If true, the tensor is bitcasted to the given
        :code:`dtype`, instead of being numerically casted.
    :type bitcast: bool, optional
    :param overflow_mode: When overflow_mode is not set or is "trunc",
        truncation (cut-off) will be used to handle overflow. When
        overflow_mode is "sautrate", the maximum value of the data type
        will be used to handle overflow.
    :type overflow_mode: string, optional
    """
    overflow_modes = ["trunc", "saturate"]
    input = semantic.to_tensor(input, _builder)
    if isinstance(bitcast, constexpr):
        bitcast = bitcast.value
    if bitcast:
        return semantic.bitcast(input, dtype, _builder)
    ret = semantic.cast(input, dtype, _builder, fp_downcast_rounding)
    if overflow_mode is not None:
        if overflow_mode in overflow_modes:
            semantic.compile_hint(ret, "overflow_mode", overflow_mode, _builder)
        else:
            raise ValueError(f"Unknown overflow_mode:{overflow_mode} is found.")
    return ret


@_tensor_member_fn
@builtin
def trans(input: tensor, *dims, _builder=None):
    """
    Permutes the dimensions of a tensor.

    If the parameter :code:`dims` is not specified, the function defaults to a (1,0) permutation,
    effectively transposing a 2D tensor.

    :param input: The input tensor.
    :param dims: The desired ordering of dimensions.  For example,
        :code:`(2, 1, 0)` reverses the order dims in a a 3D tensor.

    :code:`dims` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        trans(x, (2, 1, 0))
        trans(x, 2, 1, 0)

    :py:func:`permute` is equivalent to this function, except it doesn't
    have the special case when no permutation is specified.
    """
    if not dims:
        dims = (1, 0)
    dims = _unwrap_iterable(dims)
    return real_semantic.permute(input, dims, _builder)


@builtin
def dot(
    input,
    other,
    acc=None,
    input_precision=None,
    allow_tf32=None,
    max_num_imprecise_acc=None,
    out_dtype=float32,
    _builder=None,
):
    assert (
        input_precision is None or allow_tf32 is None
    ), "Only one of input_precision and allow_tf32 can be specified"
    assert (
        not allow_tf32
    ), "allow_tf32 is deprecated, please use input_precision='hf32' on Ascend instead."
    if input_precision is None:
        supports_tf32 = (
            _builder and "tf32" in _builder.options.allowed_dot_input_precisions
        )
        default_precision = (
            "tf32" if (supports_tf32 and (allow_tf32 or allow_tf32 is None)) else "ieee"
        )
        input_precision = os.getenv("TRITON_F32_DEFAULT", default_precision)
    else:
        assert input_precision not in [
            "tf32",
            "tf32x3",
        ], "input_precision == tf32 or tf32x3 is invalid, please use input_precision='hf32' on Ascend instead."
    input_precision = _constexpr_to_value(input_precision)
    out_dtype = _constexpr_to_value(out_dtype)
    max_num_imprecise_acc = _constexpr_to_value(max_num_imprecise_acc)
    return semantic.dot(
        input, other, acc, input_precision, max_num_imprecise_acc, out_dtype, _builder
    )


@_tensor_member_fn
@builtin
def gather(src, index, axis, _builder=None):
    """Gather from a tensor along a given dimension.
    :param src: the source tensor
    :type src: Tensor
    :param index: the index tensor
    :type index: Tensor
    :param axis: the dimension to gather along
    :type axis: int
    """
    axis = _constexpr_to_value(axis)
    return semantic.gather(src, index, axis, _builder)


@_tensor_member_fn
@builtin
def insert_slice(ful, sub, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
    """
    Insert a tensor to another tensor as specified by the operation’s offsets, sizes and strides arguments.

    :param ful: The tensor to receive tensor.
    :type ful: Tensor
    :param sub: The tensor to be inserted.
    :type sub: Tensor
    :param offsets:
    :type offsets: tuple of ints
    :param sizes:
    :type sizes: tuple of ints
    :param strides:
    :type strides: tuple of ints
    """
    assert len(ful.shape) > 0
    assert len(ful.shape) == len(sub.shape)
    new_offsets = [
        real_semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in offsets
    ]
    out = semantic.insert_slice(ful, sub, new_offsets, sizes, strides, _builder)
    return out


@_tensor_member_fn
@builtin
def extract_slice(ful, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
    """
    Extract a tensor from another tensor as specified by the operation’s offsets, sizes and strides arguments.

    :param ful: The tensor to split.
    :type ful: Tensor
    :param offsets:
    :type offsets: tuple of ints
    :param sizes:
    :type sizes: tuple of ints
    :param strides:
    :type strides: tuple of ints
    """
    assert len(ful.shape) > 0
    new_offsets = [
        real_semantic.to_tensor(o, _builder) if isinstance(o, constexpr) else o
        for o in offsets
    ]
    sub = semantic.extract_slice(ful, new_offsets, sizes, strides, _builder)
    return sub

@_tensor_member_fn
@builtin
def get_element(src, indice, _builder=None, _generator=None):
    """
    get_element op reads a ranked tensor and returns one element as specified by the given indices.
    The result of the op is a value with the same type as the elements of the tensor.
    The arity of indices must match the rank of the accessed value.

    :param src: The tensor to be accessed.
    :type src: Tensor
    :param indice:
    :type indice: tuple of ints
    """
    assert len(src.shape) > 0
    new_indice = [
        real_semantic.to_tensor(i, _builder) if isinstance(i, constexpr) else i
        for i in indice
    ]
    return semantic.get_element(src, new_indice, _builder)

@builtin
def __lshift__(self, other, _builder=None):
    if self.type.scalar.is_floating():
        raise TypeError(f"unexpected type {self.type.scalar}")
    check_bit_width(self, other)
    other = _unwrap_if_constexpr(other)
    return semantic.shl(self, other, _builder)


@builtin
def __rshift__(self, other, _builder=None):
    if self.type.scalar.is_floating():
        raise TypeError(f"unexpected type {self.type.scalar}")
    other = _unwrap_if_constexpr(other)
    check_bit_width(self, other)
    if self.dtype.is_int_signed():
        return semantic.ashr(self, other, _builder)
    else:
        return semantic.lshr(self, other, _builder)


class parallel(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.

    This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
    :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param bind_sub_block: Tells the compiler if multiple vector cores participate in the loop.
        This is used in the mixed cube-vector kernel on 910B. The number of vector cores is determined by the number of
        iteration in this loop. Currently on 910B, max 2 vector cores could be used.
    """
    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, bind_sub_block: bool = False):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)
        self.bind_sub_block = bind_sub_block


@builtin
def compile_hint(ptr, hint_name, hint_val=None, _builder=None):
    hint_name = _constexpr_to_value(hint_name)
    assert isinstance(hint_name, str), f"hint name: {hint_name} is not string"
    hint_val = _unwrap_if_constexpr(hint_val) if hint_val else hint_val
    semantic.compile_hint(ptr, hint_name, hint_val, _builder)


@builtin
def sort(ptr, dim=-1, descending=False, _builder=None):
    """
    Triton sort 前端接口

    参数：
        ptr: tl.tensor，输入张量
        dim: int 或 tl.constexpr[int]，排序维度
        descending: bool 或 tl.constexpr[bool]，是否降序
        _builder: ir.builder，底层 IR 构建器
    返回：
        values: tl.tensor，排序后的值（类型与输入一致）
    """

    try:
        dim = int(dim.value) if hasattr(dim, "value") else int(dim)
    except Exception as e:
        raise TypeError(f"dim must be an integer (or tl.constexpr int), got {dim!r}. Error: {str(e)}") from e

    if hasattr(descending, "value"):
        descending = bool(descending.value)
    else:
        descending = bool(descending)

    return semantic.sort(ptr, dim, descending, _builder)

    
@builtin
def multibuffer(src: tensor, size, _builder=None):
    """
    Set multi_buffer for an existing tensor
    :src: tensor set to bufferize multiple time
    :size: number of copies
    """
    buffer_size = _constexpr_to_value(size)
    assert isinstance(buffer_size, int) and buffer_size == 2, f"only support bufferize equals 2"
    semantic.compile_hint(src, "multi_buffer", buffer_size, _builder)


@builtin
def sync_block_all(mode, event_id, _builder=None):
    mode = _constexpr_to_value(mode)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(mode, str), f"mode: {mode} is not string"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    assert mode == "all_cube" or mode == "all_vector" or mode == "all", f"ERROR: mode = {mode}, only supports all_cube/all_vector/all"
    semantic.custom_op(_builder, "sync_block_all", mode=mode, event_id=event_id)


@builtin
def sync_block_set(sender, receiver, event_id, _builder=None):
    sender = _constexpr_to_value(sender)
    receiver = _constexpr_to_value(receiver)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(sender, str) and (sender == "cube" or sender == "vector"), f"ERROR: sender = {sender}, only supports cube/vector"
    assert isinstance(receiver, str) and (receiver == "cube" or receiver == "vector"), f"ERROR: receiver = {receiver}, only supports cube/vector"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f'Unexpected pair: {sender} -> {receiver}, only supports cube -> vector or vector -> cube')
    semantic.custom_op(_builder, "sync_block_set", sender=sender, event_id=event_id)


@builtin
def sync_block_wait(sender, receiver, event_id, _builder=None):
    sender = _constexpr_to_value(sender)
    receiver = _constexpr_to_value(receiver)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(sender, str) and (sender == "cube" or sender == "vector"), f"ERROR: sender = {sender}, only supports cube/vector"
    assert isinstance(receiver, str) and (receiver == "cube" or receiver == "vector"), f"ERROR: receiver = {receiver}, only supports cube/vector"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f'Unexpected pair: {sender} -> {receiver}, only supports cube -> vector or vector -> cube')
    semantic.custom_op(_builder, "sync_block_wait", sender=sender, event_id=event_id)


@builtin
def load_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[Union[constexpr, tensor]],
                           _builder=None) -> tensor:
    """Load a block of data from a tensor descriptor."""
    return desc.load(offsets, _builder=_builder)


@builtin
def store_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[Union[constexpr, tensor]], value: tensor,
                            _builder=None) -> tensor:
    """Store a block of data to a tensor descriptor."""
    return desc.store(offsets, value, _builder=_builder)


@builtin
def make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    _builder=None,
) -> tensor_descriptor:
    """Make a tensor descriptor object

    :param base: the base pointer of the tensor, must be 16-byte aligned
    :param shape: A list of non-negative integers representing the tensor shape
    :param strides: A list of tensor strides. Leading dimensions must be multiples
        of 16-byte strides and the last dimension must be contiguous.
    :param block_shape: The shape of block to be loaded/stored from global memory

    Notes
    *****
    On NVIDIA GPUs with TMA support, this will result in a TMA descriptor object
    and loads and stores from the descriptor will be backed by the TMA hardware.

    Currently only 2-5 dimensional tensors are supported.

    Example
    *******
    .. code-block:: python

        @triton.jit
        def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
            desc = tl.make_tensor_descriptor(
                in_out_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[M_BLOCK, N_BLOCK],
            )

            moffset = tl.program_id(0) * M_BLOCK
            noffset = tl.program_id(1) * N_BLOCK

            value = desc.load([moffset, noffset])
            desc.store([moffset, noffset], tl.abs(value))

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        M, N = 256, 256
        x = torch.randn(M, N, device="cuda")
        M_BLOCK, N_BLOCK = 32, 32
        grid = (M // M_BLOCK, N // N_BLOCK)
        inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)

    """
    return semantic.make_tensor_descriptor(base, shape, strides, block_shape, _builder)


def dtype_to_ir(self, builder: ir.builder) -> ir.type:
    if self.name.startswith("fp8"):
        raise ValueError(f'unexpected type fp8.')

    if self.name == 'void':
        return builder.get_void_ty()
    elif self.name == 'int1':
        return builder.get_int1_ty()
    elif self.name in ('int8', 'uint8'):
        return builder.get_int8_ty()
    elif self.name in ('int16', 'uint16'):
        return builder.get_int16_ty()
    elif self.name in ('int32', 'uint32'):
        return builder.get_int32_ty()
    elif self.name in ('int64', 'uint64'):
        return builder.get_int64_ty()
    elif self.name == 'fp8e5':
        return builder.get_fp8e5_ty()
    elif self.name == 'fp8e5b16':
        return builder.get_fp8e5b16_ty()
    elif self.name == 'fp8e4nv':
        return builder.get_fp8e4nv_ty()
    elif self.name == 'fp8e4b8':
        return builder.get_fp8e4b8_ty()
    elif self.name == 'fp8e4b15':
        return builder.get_fp8e4b15_ty()
    elif self.name == 'fp16':
        return builder.get_half_ty()
    elif self.name == 'bf16':
        return builder.get_bf16_ty()
    elif self.name == 'fp32':
        return builder.get_float_ty()
    elif self.name == 'fp64':
        return builder.get_double_ty()
    raise ValueError(f'fail to convert {self} to ir type')
    
