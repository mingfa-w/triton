# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import List, Optional, Union, Tuple
import numbers
import triton.language as tl
from triton._C.libtriton import ir
from triton.language.semantic import (
    wrap_tensor, 
    _str_to_rounding_mode, 
    not_equal, 
    _str_to_dot_input_precision,
    binary_op_type_checking_impl, 
    integer_promote_impl, 
    broadcast_impl_shape, 
    _str_to_sem, 
    _str_to_scope, 
    bitcast,
    bitwise_op_type_checking_impl,
    shl, ashr, lshr, fdiv, sub, mul, 
    to_tensor, 
    _str_to_load_cache_modifier, 
    _str_to_eviction_policy,
    _str_to_padding_option, 
    _canonicalize_boundary_check,
    permute,
    reshape,
)
import triton.language.math as math
import triton.language.core as core
from triton.language._utils import TRITON_MAX_TENSOR_NUMEL

from .tensor_descriptor import (
    _unwrap_if_constexpr,
    _unwrap_shape,
    block_type,
    tensor_descriptor
)

try:
    import acl
    is_compile_on_910_95 = acl.get_soc_name().startswith("Ascend910_95")
except Exception as e:
    is_compile_on_910_95 = False

def arange(start: int, end: int, builder: ir.builder) -> tl.tensor:
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("arange's arguments must be of type tl.constexpr")
    is_start_int64 = bool(start >> 32)
    is_end_int64 = bool(end >> 32)
    if is_start_int64 or is_end_int64:
        raise ValueError("arange must fit in int32")
    if end <= start:
        raise ValueError("arange's end argument must be greater than the start argument")
    range = end - start
    if range > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"end - start must be less than or equal to TRITON_MAX_TENSOR_NUMEL = {TRITON_MAX_TENSOR_NUMEL}")
    shape = [range]
    ret_ty = tl.block_type(tl.int32, shape)
    return tl.tensor(builder.create_make_range(start, end), ret_ty)

def cast(input: tl.tensor, dst_ty: tl.dtype, builder: ir.builder,
         fp_downcast_rounding: Optional[str] = None) -> tl.tensor:
    src_ty = input.type
    if isinstance(dst_ty, tl.constexpr):
        dst_ty = dst_ty.value
    if isinstance(fp_downcast_rounding, tl.constexpr):
        fp_downcast_rounding = fp_downcast_rounding.value
    if src_ty.is_block():
        dst_ty = tl.block_type(dst_ty.scalar, input.type.get_block_shapes())
    if src_ty == dst_ty:
        return input
    
    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar
    if src_sca_ty == dst_sca_ty:
        return input

    # For fp downcasting default rounding mode should be RTNE, for all other conversions it should
    # not be set
    fp_downcast_rounding = _str_to_rounding_mode(fp_downcast_rounding)
    use_custom_rounding = False
    if dst_sca_ty.is_floating() and src_sca_ty.is_floating(
    ) and dst_sca_ty.primitive_bitwidth < src_sca_ty.primitive_bitwidth:
        if fp_downcast_rounding is None: fp_downcast_rounding = ir.ROUNDING_MODE.RTNE
        elif fp_downcast_rounding != ir.ROUNDING_MODE.RTNE: use_custom_rounding = True
    else:
        if fp_downcast_rounding is not None:
            raise ValueError("fp_downcast_rounding should be set only for truncating fp conversions. "
                             "Source scalar type is " + str(src_sca_ty) + " and destination type is " + str(dst_sca_ty))
    if not is_compile_on_910_95:
       if (src_sca_ty.is_fp8() or dst_sca_ty.is_fp8()) or (src_sca_ty.is_fp64() or dst_sca_ty.is_fp64()):
            raise ValueError("[fp8, fp64] is unsupported on Ascend for now."
                           "Source scalar type is " + str(src_sca_ty) + " and destination type is " + str(dst_sca_ty))
    if (src_sca_ty.is_fp8e4b15() or dst_sca_ty.is_fp8e4b15()):
        assert builder.codegen_fns.get(
            "convert_custom_types") is not None, "target doesn't provide conversion for this type."
        return builder.codegen_fns["convert_custom_types"](input, dst_ty, fp_downcast_rounding, _builder=builder)
    # Casting with customized floating types involved: fp8 <=> bf16, fp16, fp32, fp64
    # and non-default rounding modes for downcasting
    if (src_sca_ty.is_fp8() and dst_sca_ty.is_floating()) or \
       (src_sca_ty.is_floating() and dst_sca_ty.is_fp8()) or \
       use_custom_rounding:
        return tl.tensor(builder.create_fp_to_fp(input.handle, dst_ty.to_ir(builder), fp_downcast_rounding), dst_ty)

    # bf16 <=> (not fp32)
    if (src_sca_ty.is_fp16() and not dst_sca_ty.is_fp32()) or \
       (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()):
        return cast(cast(input, tl.float32, builder), dst_sca_ty, builder)

    # Standard floating types' casting: truncation
    #   fp64 => fp32, fp16, bf16
    #   fp32 => fp16, bf16
    truncate_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth > dst_sca_ty.primitive_bitwidth
    if truncate_fp:
        return tl.tensor(builder.create_fp_trunc(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Standard floating types' casting: extension
    #   fp32 => fp64
    #   fp16 => fp32, fp64
    #   bf16 => fp32, fp64
    ext_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth < dst_sca_ty.primitive_bitwidth
    if ext_fp:
        return tl.tensor(builder.create_fp_ext(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting between integer types
    if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
       (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
        sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        else:
            return tl.tensor(builder.create_int_cast(input.handle, dst_ty.to_ir(builder), sign_extend), dst_ty)

    # Casting standard floating types to integer types
    if src_sca_ty.is_standard_floating() and dst_sca_ty.is_int():
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        elif dst_sca_ty.is_int_signed():
            return tl.tensor(builder.create_fp_to_si(input.handle, dst_ty.to_ir(builder)), dst_ty)
        else:
            return tl.tensor(builder.create_fp_to_ui(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting integer types to standard floating types
    if src_sca_ty.is_int() and dst_sca_ty.is_standard_floating():
        if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
            return tl.tensor(builder.create_ui_to_fp(input.handle, dst_ty.to_ir(builder)), dst_ty)
        else:
            return tl.tensor(builder.create_si_to_fp(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting pointer types to integer types
    if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
        bitwidth = dst_sca_ty.int_bitwidth
        if bitwidth == 64:
            return tl.tensor(builder.create_ptr_to_int(input.handle, dst_ty.to_ir(builder)), dst_ty)
        if bitwidth == 1:
            return not_equal(cast(input, tl.int64, builder), tl.tensor(builder.get_int64(0), tl.int64), builder)

    # Casting integer types to pointer types
    if src_sca_ty.is_int() and dst_sca_ty.is_ptr():
        return tl.tensor(builder.create_int_to_ptr(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting pointer types to pointer types
    if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
        return tl.tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)), dst_ty)

    assert False, f'cannot cast {input} to {dst_ty}'

def dot(lhs: tl.tensor, rhs: tl.tensor, acc: tl.tensor, input_precision: Optional[str], max_num_imprecise_acc: int,
        out_dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    assert lhs.type.is_block() and rhs.type.is_block()

    if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
        # All combinations of supported fp8 x fp8 are permitted
        pass
    else:
        assert lhs.dtype in (tl.int1, tl.int8, tl.uint8, tl.float16, tl.bfloat16,
                             tl.float32), f"Unsupported lhs dtype {lhs.dtype}"
        assert rhs.dtype in (tl.int1, tl.int8, tl.uint8, tl.float16, tl.bfloat16,
                             tl.float32), f"Unsupported rhs dtype {rhs.dtype}"
        assert lhs.dtype == rhs.dtype, f"Both operands must be same dtype. Got {lhs.dtype} and {rhs.dtype}"

    if lhs.dtype.is_fp8e4b15() or rhs.dtype.is_fp8e4b15():
        lhs = cast(lhs, tl.float16, builder)
        rhs = cast(rhs, tl.float16, builder)

    if input_precision is None:
        input_precision = builder.options.default_dot_input_precision

    input_precision = _str_to_dot_input_precision(input_precision, builder)

    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)
    assert lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3, f"Both inputs must be either 2D or 3D; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
    assert lhs.shape[-1].value == rhs.shape[
        -2].value, f"First input shape ({lhs.shape}) and second input shape {rhs.shape} are not compatible for matmul (second index of first shape ({lhs.shape[-1].value}) must be equal to first index of second shape ({rhs.shape[-2].value})"
    assert builder.codegen_fns.get("min_dot_size") is not None, "target doesn't provide lower shape bounds for dot."
    min_dot_size = builder.codegen_fns["min_dot_size"](lhs.type, rhs.type)
    assert lhs.shape[-2].value >= min_dot_size[0] and lhs.shape[-1].value >= min_dot_size[2] \
        and rhs.shape[-1].value >= min_dot_size[1], \
            f"Input shapes should have M >= {min_dot_size[0]}, N >= {min_dot_size[1]} and K >= {min_dot_size[2]}"
    if lhs.type.scalar.is_int():
        assert lhs.type.scalar == tl.int8, "only int8 supported!"
        _0 = builder.get_int32(0)
        ret_scalar_ty = tl.int32
    elif out_dtype.is_bf16():
        raise ValueError(
            "out_dtype=bfloat16 is unsupported. Please use out_dtype=float32/float16 and cast with `.to(tl.bfloat16)`")
    elif lhs.type.scalar.is_fp32() or lhs.type.scalar.is_bf16():
        _0 = builder.get_fp32(0)
        ret_scalar_ty = tl.float32
    else:
        _0 = builder.get_fp16(0) if out_dtype.is_fp16() else builder.get_fp32(0)
        ret_scalar_ty = out_dtype

    M = lhs.type.shape[-2]
    N = rhs.type.shape[-1]
    K = lhs.type.shape[-1]
    B = lhs.type.shape[0] if lhs_rank == 3 else None
    ret_ty = tl.block_type(ret_scalar_ty, [B, M, N] if B else [M, N])
    if acc is None:
        acc_handle = builder.create_splat(_0, [B, M, N] if B else [M, N])
    else:
        acc_handle = acc.handle
        assert acc.type == ret_ty

    if (input_precision == getattr(ir.INPUT_PRECISION, "HF32")):
        if (not lhs.dtype.is_fp32() or not rhs.dtype.is_fp32() or not ret_scalar_ty.is_fp32()):
            raise ValueError("input_precision = 'hf32' must be used with f32 * f32 = f32 on Ascend")

    if max_num_imprecise_acc is not None:
        print("max_num_imprecise_acc in tl.dot is not supported on Ascend yet. Thus it is ignored.")
    max_num_imprecise_acc = 0
    return tl.tensor(builder.create_dot(lhs.handle, rhs.handle, acc_handle, input_precision, max_num_imprecise_acc),
                     ret_ty)

# Use Union instead of |. Becase python 3.9 does not support |.
# It will reports error: TypeError: unsupported operand type(s) for |: 'type' and 'ABCMeta'
def floordiv(input: Union[tl.tensor, numbers.Number], other: Union[tl.tensor, numbers.Number], builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if input_scalar_ty.is_int() and other_scalar_ty.is_int():
        ret_ty = integer_promote_impl(input_scalar_ty, other_scalar_ty)
        input = cast(input, ret_ty, builder)
        other = cast(other, ret_ty, builder)
        if ret_ty.is_int_signed():
            return tl.tensor(builder.create_sdiv(input.handle, other.handle), input.type)
        else:
            return tl.tensor(builder.create_udiv(input.handle, other.handle), input.type)
    raise TypeError(f"unexpected type {input_scalar_ty}")


def mod(input: Union[tl.tensor, numbers.Number], other: Union[tl.tensor, numbers.Number], builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    # float
    if scalar_ty.is_floating():
        floor = math.floor(fdiv(input, other, False, builder), _builder=builder)
        ret = sub(input, mul(floor, other, True, builder), True, builder)
        return ret
    # % int
    elif scalar_ty.is_int():
        if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
            raise TypeError("Cannot mod " + scalar_ty.__repr__() + " by " + other_scalar_ty.__repr__() + " "
                            "because they have different signedness;"
                            "this is unlikely to result in a useful answer. Cast them to the same signedness.")
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_srem(input.handle, other.handle), input.type)
        else:
            return tl.tensor(builder.create_urem(input.handle, other.handle), input.type)
    raise TypeError(f"unexpected type {scalar_ty}")


def minus(input: tl.tensor, builder: ir.builder) -> tl.tensor:
    input_sca_ty = input.type.scalar
    if hasattr(input, 'was_bool_to_int8'):
        if input.type.scalar.is_int8():
            raise TypeError(f"unexpected type bool")
    if input_sca_ty.is_ptr():
        raise ValueError("wrong type argument to unary minus (" + input_sca_ty.__repr__() + ")")
    _0 = tl.tensor(builder.get_null_value(input_sca_ty.to_ir(builder)), input_sca_ty)
    return sub(_0, input, True, builder)


def and_(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    if input.type.scalar.is_floating():
        raise TypeError(f"unexpected type {input.type.scalar}")
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_and(input.handle, other.handle), input.type)


def or_(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    if input.type.scalar.is_floating():
        raise TypeError(f"unexpected type {input.type.scalar}")
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_or(input.handle, other.handle), input.type)


def xor_(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    if input.type.scalar.is_floating():
        raise TypeError(f"unexpected type {input.type.scalar}")
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_xor(input.handle, other.handle), input.type)


def gather(src: tl.tensor, index: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    assert index.dtype.is_int(), "index must be an integer tensor"
    if not (src.dtype.is_floating() or src.dtype.is_int8()):
        raise ValueError(f"Expected dtype fp16/fp32/bf16/f8E5M2/f8E4M3FN/int8, but got {src.dtype}")

    rank = len(src.type.shape)
    assert len(index.type.shape) == rank, "source and index tensors must have the same rank"

    assert -rank <= axis < rank, f"gather axis {axis} must be < source rank ({rank})"
    if axis < 0:
        axis += rank

    for d in range(rank):
        if d == axis:
            continue
        assert index.type.shape[d] == src.type.shape[d], f"index dim {axis} must match the corresponding source dim"

    gather = builder.create_gather(src.handle, index.handle, axis)
    return wrap_tensor(gather, src.type.scalar, index.type.shape)

def insert_slice(ful: tl.tensor, sub: tl.tensor, offsets: List[tl.tensor], sizes: List[int], strides: List[int], builder: ir.builder) -> tl.tensor:
    assert(len(ful.shape) == len(offsets))
    assert(len(ful.shape) == len(sizes))
    assert(len(ful.shape) == len(strides))
    assert(all([s>=1 for s in sizes]))
    assert(all([s>=0 for s in strides]))
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, ful.shape)
    out = builder.create_insert_slice(ful.handle, sub.handle, new_offsets, sizes, strides)
    return tl.tensor(out, ret_type)


def invert(input: tl.tensor, builder: tl.tensor) -> tl.tensor:
    if hasattr(input, 'was_bool_to_int8'):
        assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
        input = cast(input, tl.int1, builder)
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_floating():
        raise TypeError(f"unexpected type {input_sca_ty}")
    if input_sca_ty.is_ptr():
        raise ValueError("wrong type argument to unary invert (" + input_sca_ty.__repr__() + ")")
    _1 = tl.tensor(builder.get_all_ones_value(input_sca_ty.to_ir(builder)), input_sca_ty)
    return xor_(input, _1, builder)


def logical_and(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    dst_sca_ty = tl.dtype("int1")
    dst_bits = dst_sca_ty.primitive_bitwidth
    if hasattr(input, 'was_bool_to_int8'):
        assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
        input = cast(input, tl.int1, builder)
    if not input.type.is_int1():
        src_sca_ty = input.type.scalar
        src_bits = src_sca_ty.primitive_bitwidth
        if src_bits == dst_bits or src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
            input = bitcast(input, tl.dtype("int1"), builder)
        else:
            input = not_equal(input, 0, builder)
    if hasattr(other, 'was_bool_to_int8'):
        assert other.type.scalar.is_int8(), "Other input wat bool to int8. However, other input.type is not int8."
        other = cast(other, tl.int1, builder)
    if not other.type.is_int1():
        src_sca_ty = other.type.scalar
        src_bits = src_sca_ty.primitive_bitwidth
        if src_bits == dst_bits or src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
            other = bitcast(other, tl.dtype("int1"), builder)
        else:
            other = not_equal(other, 0, builder)
    return and_(input, other, builder)


def logical_or(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    dst_sca_ty = tl.dtype("int1")
    dst_bits = dst_sca_ty.primitive_bitwidth
    if hasattr(input, 'was_bool_to_int8'):
        assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
        input = cast(input, tl.int1, builder)
    if not input.type.is_int1():
        src_sca_ty = input.type.scalar
        src_bits = src_sca_ty.primitive_bitwidth
        if src_bits == dst_bits or src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
            input = bitcast(input, tl.dtype("int1"), builder)
        else:
            input = not_equal(input, 0, builder)
    if hasattr(other, 'was_bool_to_int8'):
        assert other.type.scalar.is_int8(), "Other wat bool to int8. However, other.type is not int8."
        other = cast(other, tl.int1, builder)
    if not other.type.is_int1():
        src_sca_ty = other.type.scalar
        src_bits = src_sca_ty.primitive_bitwidth
        if src_bits == dst_bits or src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
            other = bitcast(other, tl.dtype("int1"), builder)
        else:
            other = not_equal(other, 0, builder)
    return or_(input, other, builder)


def not_(input: tl.tensor, builder: ir.builder):
    if hasattr(input, 'was_bool_to_int8'):
        assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
        input = cast(input, tl.int1, builder)
    if input.type.scalar.is_floating():
        raise TypeError(f"unexpected type {input.type.scalar}")
    return invert(input, builder)


def _load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, care_padding, builder):
    # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
    if not ptr.type.scalar.is_ptr():
        raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")

    # Check `mask`, `other`, `boundary_check`, and `padding` arguments
    if mask is None and other is not None:
        raise ValueError("`other` cannot be provided without `mask`")
    if padding or boundary_check:
        raise ValueError("`padding_option` or `boundary_check` argument is not supported for loading a tensor of"
                         "pointers or loading a scalar. Because the compiler does not know the boundary; please "
                         "use block pointers (defined by `make_block_ptr`) instead")

    if mask is not None and other is None and care_padding == True:
        other = to_tensor(0, builder)
    # For a pointer of scalar, check the type of `mask` and `other`
    if not ptr.type.is_block():
        if mask and mask.type.is_block():
            raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
        if other and other.type.is_block():
            raise ValueError("Other argument cannot be block type if pointer argument is not a block")

    # Make `mask` and `other` into the same shape as `ptr`
    if ptr.type.is_block():
        if mask is not None:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if other is not None:
            other = broadcast_impl_shape(other, ptr.type.get_block_shapes(), builder)

    # Get `pointer_type<elt_ty>` and `elt_ty`
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty

    # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
    is_bool = elt_ty == tl.int1
    if is_bool:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = cast(ptr, ptr_ty, builder)

    # Cast `other` into `elt_ty` type
    if other is not None:
        other = cast(other, elt_ty, builder)

    # Create loaded result type `dst_ty`
    if ptr.type.is_block():
        shape = ptr.type.get_block_shapes()
        dst_ty = tl.block_type(elt_ty, shape)
    else:
        # Load by de-referencing the pointer of scalar
        dst_ty = elt_ty

    # Build IR
    if mask is None:
        ret = tl.tensor(builder.create_load(ptr.handle, cache, eviction, is_volatile), dst_ty)
    else:
        ret = tl.tensor(
            builder.create_masked_load(ptr.handle, mask.handle, other.handle if other else None, cache, eviction,
                                       is_volatile), dst_ty)
    # Do not cast back to int1 when is_bool=true. We directly use the int8 tensor given by tl.load
    if is_bool:
        ret.was_bool_to_int8 = True

    return ret


def _load_block_pointer(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder):
    # Load by a block pointer: pointer_type<block_type<>>
    # Block pointer can not have mask and other arguments
    if mask is not None or other is not None:
        raise ValueError("mask and other arguments cannot be specified for loading block pointers")

    elt_ty = ptr.type.element_ty.element_ty
    assert elt_ty != tl.int1, "`tl.int1` should be rewrited in `tl.make_block_ptr`"
    if elt_ty.is_int() and padding == ir.PADDING_OPTION.PAD_NAN:
        raise ValueError("Padding option `nan` is not supported for integer block pointers")

    # `dst_ty` is de-referenced type of the pointer type
    dst_ty = ptr.type.element_ty

    # Check `boundary_check` argument
    boundary_check = _canonicalize_boundary_check(boundary_check, dst_ty.get_block_shapes())

    if boundary_check and padding is None:
        padding = ir.PADDING_OPTION.PAD_ZERO

    # Build IR
    return tl.tensor(
        builder.create_tensor_pointer_load(ptr.handle, boundary_check, padding, cache, eviction, is_volatile), dst_ty)


def load(ptr: tl.tensor, mask: Optional[tl.tensor], other: Optional[tl.tensor], boundary_check: Tuple,
         padding_option: str, cache_modifier: str, eviction_policy: str, is_volatile: bool, care_padding: bool,
         builder: ir.builder) -> tl.tensor:
    # Cache, eviction and padding options
    cache = _str_to_load_cache_modifier(cache_modifier)
    eviction = _str_to_eviction_policy(eviction_policy)
    padding = _str_to_padding_option(padding_option)

    if ptr.type.is_ptr() and ptr.type.element_ty.is_block():
        # Load by a block pointer: `pointer_type<block_type<>>`
        return _load_block_pointer(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder)
    else:
        # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
        return _load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, care_padding, builder)


def minimum(x: tl.tensor, y: tl.tensor, propagate_nan: tl.PropagateNan, builder: ir.builder):
    x, y = binary_op_type_checking_impl(x, y, builder)
    dtype = x.dtype
    if dtype.is_bool():
        raise TypeError(f"Unexpected dtype {dtype}")
    if dtype.is_floating():
        if propagate_nan == tl.PropagateNan.ALL:
            return tl.tensor(builder.create_minimumf(x.handle, y.handle), x.type)
        elif propagate_nan == tl.PropagateNan.NONE:
            return tl.tensor(builder.create_minnumf(x.handle, y.handle), x.type)
        else:
            raise ValueError(f"Unexpected propagate_nan {propagate_nan}")
    elif dtype.is_int_signed():
        return tl.tensor(builder.create_minsi(x.handle, y.handle), x.type)
    elif dtype.is_int_unsigned():
        return tl.tensor(builder.create_minui(x.handle, y.handle), x.type)
    else:
        raise TypeError(f"Unexpected dtype {dtype}")

def maximum(x: tl.tensor, y: tl.tensor, propagate_nan: tl.PropagateNan, builder: ir.builder):
    x, y = binary_op_type_checking_impl(x, y, builder)
    dtype = x.dtype
    if dtype.is_bool():
        raise TypeError(f"Unexpected dtype {dtype}")
    if dtype.is_floating():
        if propagate_nan == tl.PropagateNan.ALL:
            return tl.tensor(builder.create_maximumf(x.handle, y.handle), x.type)
        elif propagate_nan == tl.PropagateNan.NONE:
            return tl.tensor(builder.create_maxnumf(x.handle, y.handle), x.type)
        else:
            raise ValueError(f"Unexpected propagate_nan {propagate_nan}")
    elif dtype.is_int_signed():
        return tl.tensor(builder.create_maxsi(x.handle, y.handle), x.type)
    elif dtype.is_int_unsigned():
        return tl.tensor(builder.create_maxui(x.handle, y.handle), x.type)
    else:
        raise TypeError(f"Unexpected dtype {dtype}")

def extract_slice(ful: tl.tensor, offsets: List[tl.tensor], sizes: List[int], strides: List[int], builder: ir.builder) -> tl.tensor:
    assert(len(ful.shape) == len(offsets))
    assert(len(ful.shape) == len(sizes))
    assert(len(ful.shape) == len(strides))
    assert(all([s>=1 for s in sizes]))
    assert(all([s>=0 for s in strides]))
    new_offsets = [o.handle for o in offsets]
    ret_type = tl.block_type(ful.type.scalar, sizes)
    out = builder.create_extract_slice(ful.handle, new_offsets, sizes, strides)
    return tl.tensor(out, ret_type)

def get_element(src: tl.tensor, indice: List[tl.tensor], builder: ir.builder):
    if len(src.shape) != len(indice):
        raise ValueError("Indice's rank must be equal to src tensor's rank")

    new_indice = [i.handle for i in indice]
    result = builder.create_extract_scalar(src.handle, new_indice)
    return wrap_tensor(result, src.type.scalar, None)

def atom_red_typechecking_impl(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, op: str,
                               builder: ir.builder) -> Tuple[tl.tensor, tl.tensor, tl.tensor]:
    if not ptr.type.scalar.is_ptr():
        raise ValueError("Pointer argument of store instruction is " + ptr.type.__repr__())
    if ptr.type.is_const() or ptr.type.element_ty.is_const():
        raise ValueError("Cannot store to a constant pointer")
    if ptr.type.is_block():
        if mask is not None:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if val is not None:
            val = broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
    val = cast(val, ptr.type.scalar.element_ty, builder)
    if not mask:
        mask_ir = builder.get_int1(True)
        mask_ty = tl.int1
        if ptr.type.is_block():
            mask_ir = builder.create_splat(mask_ir, ptr.type.get_block_shapes())
            mask_ty = tl.block_type(tl.int1, ptr.type.get_block_shapes())
        mask = tl.tensor(mask_ir, mask_ty)
    return ptr, val, mask


def atomic_cas(ptr: tl.tensor, cmp: tl.tensor, val: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    element_ty = ptr.type.scalar.element_ty
    if element_ty in [tl.int1, tl.int8, tl.float64, tl.bfloat16]:
        raise ValueError(f"atomic_cas does not support {str(element_ty)}. "
                         "All support dtypes are int16, int32, int64, float16, float32.")
    return tl.tensor(builder.create_atomic_cas(ptr.handle, cmp.handle, val.handle, sem, scope), val.type)


def atomic_max(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'max', builder)
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    sca_ty = val.type.scalar
    # direct call to atomic_max for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(
                builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, ptr.handle, val.handle, mask.handle, sem, scope), val.type)
        else:
            return tl.tensor(
                builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

    # Design for NPU
    return tl.tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

def atomic_min(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'min', builder)
    sem = _str_to_sem(sem)
    scope = _str_to_scope(scope)
    sca_ty = val.type.scalar
    # direct call to atomic_min for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(
                builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, ptr.handle, val.handle, mask.handle, sem, scope), val.type)
        else:
            return tl.tensor(
                builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

    # Design for NPU
    return tl.tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, ptr.handle, val.handle, mask.handle, sem, scope), val.type)


def compile_hint(ptr: tl.tensor, hint_name: str, hint_val, builder: ir.builder):
    if not hint_val:
        hint_val = builder.get_unit_attr()
    elif isinstance(hint_val, bool):
        hint_val = builder.get_bool_attr(hint_val)
    elif isinstance(hint_val, int):
        hint_val = builder.get_int32_attr(hint_val)
    elif isinstance(hint_val, core.constexpr):
        hint_val = builder.get_str_attr(hint_val.value)
    elif isinstance(hint_val, list):
        # only support i64 array attr for now
        hint_val = builder.get_i64_array_attr(hint_val)
    else:
        raise ValueError(f"Unsupported hint value type: {type(hint_val)}")
    builder.create_annotation(ptr.handle, hint_name, hint_val)


def custom_op(builder: ir.builder, op_name: str, **kwargs):
    if op_name == "sync_block_all":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["mode"], kwargs["event_id"])

    elif op_name == "sync_block_set":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["sender"], kwargs["event_id"])
    
    elif op_name == "sync_block_wait":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["sender"], kwargs["event_id"])
    
    raise ValueError(f"Unsupported custom op: {op_name}")


def sort(ptr: tl.tensor, dim: int, descending, builder: ir.builder):
    """
    Triton sort 操作

    参数：
        ptr: tl.tensor，输入张量
        dim: int，排序维度，必须是尾轴（最后一维）
        descending: bool 或 constexpr，是否降序
        builder: ir.builder，底层 IR 构建器
    返回：
        values: tl.tensor，排序后的值（类型与输入一致）
    """

    allowed_types = {tl.int8, tl.int16, tl.bfloat16, tl.float16, tl.float32, tl.int32, tl.int64, tl.float8e4nv, tl.float8e5}
    base_ty = ptr.type.scalar if hasattr(ptr.type, "scalar") else ptr.type
    if base_ty not in allowed_types:
        raise TypeError(
            f"tt.sort only supports int8, int16, bfloat16, float16, float32, int32, int64, float8e4nv, float8e5"
            f"but got {ptr.type}"
        )

    shape = getattr(ptr, "shape", None)
    if shape is None or shape == ():
        shape = getattr(getattr(ptr, "type", None), "shape", None)

    rank = None
    if shape is not None:
        try:
            rank = len(shape)
        except Exception:
            rank = len(list(shape))

    if rank is not None:
        if rank < 1:
            raise ValueError("tt.sort requires tensor rank >= 1")
        last_dim = rank - 1
        norm_dim = dim if dim >= 0 else dim + rank
        if norm_dim != last_dim:
            raise ValueError(
                f"tt.sort only supports sorting along the last dimension "
                f"(dim={last_dim} or -1) for shape {tuple(shape)}, but got dim={dim}"
            )
        dim = last_dim
    else:
        if dim != -1:
            raise ValueError(
                "tt.sort only supports the last dimension; when rank is unknown "
                "you must pass dim=-1"
            )

    if hasattr(descending, "value"):
        descending = bool(descending.value)
    else:
        descending = bool(descending)

    sorted_vals = builder.create_sort(ptr.handle, dim, descending)

    values = tl.tensor(sorted_vals, type=ptr.type)

    return values


def _str_to_fp_type(float_format: Optional[str]):
    if float_format == 'e4m3':
        return ir.F8F6F4TY.E4M3
    if float_format == 'e5m2':
        return ir.F8F6F4TY.E5M2
    if float_format == 'e2m3':
        return ir.F8F6F4TY.E2M3
    if float_format == 'e3m2':
        return ir.F8F6F4TY.E3M2
    if float_format == 'e2m1':
        return ir.F8F6F4TY.E2M1
    if float_format == 'bf16':
        return ir.F8F6F4TY.BF16
    if float_format == 'fp16':
        return ir.F8F6F4TY.FP16
    raise ValueError(f"Invalid float format: {float_format}.")


def _bitcast_to_fp_type(val: tl.tensor, float_format: str, builder: ir.builder):
    triton_ty = {"e5m2": tl.float8e5, "e4m3": tl.float8e4nv, "bf16": tl.bfloat16, "fp16": tl.float16}.get(float_format)
    if triton_ty is None:
        assert float_format == "e2m1", f"Internal Error: Unexpected float format: {float_format}"
        assert val.dtype == tl.uint8, f"e2m1 format must be packed as uint8. Got {val.dtype}"
        return val
    if val.dtype == triton_ty:
        return val
    else:
        unsigned_ty = {"e5m2": tl.uint8, "e4m3": tl.uint8, "bf16": tl.uint16, "fp16": tl.uint16}[float_format]
        assert val.dtype == unsigned_ty, f"Unexpected dtype for {float_format}. Got {val.dtype}"
        return bitcast(val, triton_ty, builder)


def dot_scaled(lhs: tl.tensor, lhs_scale: tl.tensor, lhs_format: str, rhs: tl.tensor, rhs_scale: Optional[tl.tensor],
               rhs_format: str, acc: Union[tl.tensor, None], out_dtype: tl.dtype, lhs_k_pack, rhs_k_pack, 
               builder: ir.builder) -> tl.tensor:
    assert lhs.type.is_block() and rhs.type.is_block()
    assert lhs.dtype == tl.bfloat16 or lhs.dtype == tl.float16, f"lhs matrix dtype must be bf16 or fp16"
    assert rhs.dtype == tl.bfloat16 or rhs.dtype == tl.float16, f"rhs matrix dtype must be bf16 or fp16"
    assert lhs.dtype == rhs.dtype, f"lhs rhs matrix must get same dtype"
    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)
    assert lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3, f"Both inputs must be either 2D or 3D; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
    lhs_format: str = lhs_format.value
    rhs_format: str = rhs_format.value
    lhs_format_enum = _str_to_fp_type(lhs_format)
    rhs_format_enum = _str_to_fp_type(rhs_format)
    allowed_formats = {"bf16", "fp16"} # unsupported fp8/4 dtype: "e2m1", "e4m3", "e5m2" 
    assert lhs_format in allowed_formats, f"NYI: lhs_format {lhs_format}"
    assert rhs_format in allowed_formats, f"NYI: rhs_format {rhs_format}"
    rhs_scale_is_none = rhs_scale is None or (isinstance(rhs_scale, tl.constexpr) and rhs_scale.value is None)
    lhs_scale_is_none = lhs_scale is None or (isinstance(lhs_scale, tl.constexpr) and lhs_scale.value is None)
    assert isinstance(lhs_scale, tl.tensor) and lhs_scale.dtype == tl.int8, f"lhs_scale must be int8 tensor"
    if not rhs_scale_is_none:
        assert isinstance(rhs_scale, tl.tensor) and rhs_scale.dtype == tl.int8, f"rhs_scale must be int8 tensor"
    lhs = _bitcast_to_fp_type(lhs, lhs_format, builder)
    rhs = _bitcast_to_fp_type(rhs, rhs_format, builder)

    if lhs_k_pack == False:
        dims = (1, 0)
        dims = core._unwrap_iterable(dims)
        tmp_lhs = permute(lhs, dims, builder)
        lhs = reshape(tmp_lhs, (lhs.shape[0], lhs.shape[1]), True, builder)

    if rhs_k_pack == False:
        dims = (1, 0)
        dims = core._unwrap_iterable(dims)
        tmp_rhs = permute(rhs, dims, builder)
        rhs = reshape(tmp_rhs, (rhs.shape[0], rhs.shape[1]), True, builder)
        
    assert lhs.type.shape[-1] == rhs.type.shape[-2], (
        f"lhs last dimension (columns) {lhs.shape[-1]} "
        f"must equal rhs penultimate dimension (rows) {rhs.shape[-2]}"
    )
    M = lhs.type.shape[-2]
    K, N = rhs.type.shape[-2:]
    PACKED_A = 2 if lhs_format == "e2m1" else 1
    PACKED_B = 2 if lhs_format == "e2m1" else 1
    assert K * PACKED_B == PACKED_A * lhs.type.shape[
        -1], f"Reduction dimension should pack the same number of elements; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
    B = lhs.type.shape[0] if lhs_rank == 3 else None

    ret_ty = tl.block_type(out_dtype, [B, M, N] if B else [M, N])
    _0 = builder.get_fp32(0)
    if acc is None:
        acc_handle = builder.create_splat(_0, [B, M, N] if B else [M, N])
    else:
        acc_handle = acc.handle
        assert acc.type == ret_ty
    rhs_scale_handle = None if rhs_scale_is_none else rhs_scale.handle
    lhs_scale_handle = None if lhs_scale_is_none else lhs_scale.handle
    return tl.tensor(
        builder.create_dot_scaled(lhs.handle, lhs_scale.handle, lhs_format_enum, rhs.handle, rhs_scale_handle,
                                  rhs_format_enum, acc_handle), ret_ty)




def scalar_constant(value, dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    if dtype is None:
        raise ValueError("dtype must be specified when value is not a tensor")
    if value == 0:
        value = builder.get_null_value(dtype.to_ir(builder))
    else:
        get_value_fn = getattr(builder, f"get_{dtype.name}")
        value = get_value_fn(value)
    return tl.tensor(value, dtype)


def make_scalar(value, dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    if isinstance(value, tl.tensor):
        assert value.numel.value == 1, "only accepts size-1 tensor"
        return cast(value, dtype, builder)
    return scalar_constant(value, dtype, builder)


def make_tensor_descriptor(
    base: tl.tensor,
    shape: List[tl.tensor],
    strides: List[tl.tensor],
    block_shape: List[tl.constexpr],
    builder: ir.builder
) -> tensor_descriptor:
    ndim = len(shape)
    if not (1 <= ndim <= 5):
        raise ValueError(f"Expected 1 <= ndim <= 5 but got {ndim} dimensions")
    if len(strides) != ndim:
        raise ValueError(f"Expected {ndim} strides but got {len(strides)}")
    if len(block_shape) != ndim:
        raise ValueError(f"Expected block_shape to have {ndim} dimensions but got {len(strides)}")
    assert isinstance(base.dtype, tl.pointer_type)
    primitive_bitwidth = base.dtype.element_ty.primitive_bitwidth
    if primitive_bitwidth == 1:
        raise ValueError("int1 type is not supported for make_tensor_descriptor yet")
    elem_size = primitive_bitwidth // 8
    contig_dim_size = _unwrap_if_constexpr(block_shape[-1])
    if contig_dim_size * elem_size < 16:
        raise ValueError(
            f"Descriptor block shape must have at least 16 bytes in the last dimension, but got {contig_dim_size} * {elem_size} = {contig_dim_size * elem_size} bytes"
        )

    strides[-1] = _unwrap_if_constexpr(strides[-1])
    if strides[-1] != 1:
        raise ValueError(f"Tensor descriptor last dim must be 1 but got {strides[-1]}")
    
    shape = [make_scalar(x, tl.int32, builder) for x in shape]
    strides = [make_scalar(x, tl.int64, builder) for x in strides]

    block_shape = _unwrap_shape(block_shape)

    assert isinstance(base.type, tl.pointer_type)
    desc_block_type = block_type(base.type.element_ty, block_shape)
    base_handle = base.handle
    is_signed_int = base.type.element_ty.is_int_signed()

    handle = builder.create_make_tensor_descriptor(base_handle, [s.handle for s in shape],
                                                    [s.handle for s in strides], block_shape, is_signed_int)
    return tensor_descriptor(handle, shape, strides, desc_block_type)


def index_select_simd(
    src: tl.tensor,
    dim: int,
    index: tl.tensor,
    src_shape: List[Union[int, tl.tensor]],
    src_offset: List[Union[int, tl.tensor]],
    read_shape: List[Union[int, tl.tensor]],
    builder: ir.builder
) -> tl.tensor:
    """
    Index select operation (SIMD version) that loads data from multiple indices along a dimension.
    
    Args:
        src: Source tensor pointer (in GM)
        dim: Dimension along which to select indices
        index: 1D tensor of indices to select (in UB)
        src_shape: Complete shape of source tensor. Each element can be int or tensor.
        src_offset: Starting offset for reading. Each element can be int or tensor.
        read_shape: Size to read (tile shape). Each element can be int or tensor.
        builder: IR builder

    Returns:
        Result tensor in UB

    Constraints:
        - read_shape[dim] must be -1
        - src_offset[dim] can be -1 (ignored)
        - All list parameters must have the same length (ndim)
    """
    # Validate inputs
    ndim = len(src_shape)
    assert len(src_offset) == ndim, \
        f"src_offset length {len(src_offset)} must match src_shape length {ndim}"
    assert len(read_shape) == ndim, \
        f"read_shape length {len(read_shape)} must match src_shape length {ndim}"
    assert 0 <= dim < ndim, \
        f"dim={dim} must be in range [0, {ndim})"
    assert len(index.shape) == 1, \
        f"index must be 1D tensor, got {len(index.shape)}D"
    assert dim < ndim - 1, \
        f"index_select_simd cannot support trailing dimension as dim={dim}, ndim={ndim}"
    
    newsrc_shape = [o.handle for o in src_shape]
    newsrc_offset = [o.handle for o in src_offset]
    # Create output type
    return_shape = [
        index.shape[0] if i == dim else read_shape[i] 
        for i in range(ndim)
    ]
    element_ty = src.type.element_ty
    output_ty = tl.block_type(element_ty, return_shape)
    out = builder.create_index_select_simd(src.handle, index.handle, dim, newsrc_shape, newsrc_offset, read_shape, return_shape)
    return tl.tensor(out, output_ty)

def embedding_gather(src: tl.tensor, idx: tl.tensor, bound: int, blksiz: int, offsets: Tuple, numels: Tuple, builder: ir.builder) -> tl.tensor:
    """
    Embedding
    :src_ptr:
    :idx:
    """
    assert idx.dtype.is_int(), "index must be an integer tensor"
    if not src.dtype.element_ty.is_floating():
        raise ValueError(f"Expected dtype fp16/fp32/bf16, but got {src.dtype.element_ty}")
    
    def _convert_elem_to_ir_value(builder, elem, require_i64):
        if isinstance(elem, int):
            elem = tl.constexpr(elem)
        if isinstance(elem, tl.constexpr):
            if require_i64:
                assert -2**63 <= elem.value < 2**63, f"Block pointers only support 64 bit `shape/strides`, " \
                    f"got a value {elem.value} which is out of the range"
                return builder.get_int64(elem.value)
            else:
                assert -2**31 <= elem.value < 2**31, f"Block pointers only support 32 bit `offsets/block_shape`, " \
                    f"got a value {elem.value} which is out of the range"
                return builder.get_int32(elem.value)
        elif isinstance(elem, tl.tensor):
            if require_i64:
                return builder.create_int_cast(elem.handle, builder.get_int64_ty(), elem.dtype.is_int_signed())
            else:
                return elem.handle
        else:
            assert False, f"Unsupported element type in shape/strides/offsets: {type(elem)}"

    require_i64 = idx.dtype.is_int64()
    # require_i64 = True
    offsets = [_convert_elem_to_ir_value(builder, elem, require_i64) for elem in offsets]
    numels = [_convert_elem_to_ir_value(builder, elem, require_i64) for elem in numels]
    ret = builder.create_embedding_gather(src.handle, idx.handle, bound, blksiz, offsets, numels)
    ret_shape = [_unwrap_if_constexpr(s) for s in idx.shape]
    ret_shape.append(blksiz)
    return wrap_tensor(ret, src.dtype.element_ty, ret_shape)