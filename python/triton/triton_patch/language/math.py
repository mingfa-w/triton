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

from functools import wraps
from typing import List
import numbers
from triton.language import core
from triton.language.math import _add_math_1arg_docstr, _add_math_2arg_docstr, _add_math_3arg_docstr
from triton.language import semantic

T = core.TypeVar('T')


def _check_dtype(dtypes: List[str]) -> T:
    """
    We're following libdevice's convention to check accepted data types for math functions.
    It is not a good practice to support all data types as accelerators/GPUs don't support
    many float16 and bfloat16 math operations.
    We should let the users know that they are using and invoke explicit cast to convert
    the data type to the supported one.
    """

    def wrapper(fn):

        @wraps(fn)
        def check(*args, **kwargs):
            # concatenate args and kwargs
            all_args = list(args) + list(kwargs.values())
            for arg in [a for a in all_args if isinstance(a, core.tensor)]:
                arg_type = arg.type.scalar.name
                if hasattr(arg, 'was_bool_to_int8') and arg.was_bool_to_int8:
                    # In Triton, int1 maps to the boolean type
                    arg_type = 'int1'
                if arg_type not in dtypes:
                    raise ValueError(f"Expected dtype {dtypes} but got {arg_type}")
            return fn(*args, **kwargs)

        return check

    return wrapper


@core.builtin
@_check_dtype(dtypes=["int32", "uint32", "int64"])
@_add_math_2arg_docstr("most significant N bits of the 2N-bit product")
def umulhi(x, y, _builder=None):
    x = semantic.to_tensor(x, _builder)
    y = semantic.to_tensor(y, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    return core.tensor(_builder.create_umulhi(x.handle, y.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("exponential")
@core._tensor_member_fn
def exp(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_exp(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("exponential (base 2)")
@core._tensor_member_fn
def exp2(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_exp2(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("natural logarithm")
@core._tensor_member_fn
def log(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_log(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("logarithm (base 2)")
@core._tensor_member_fn
def log2(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_log2(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("cosine")
@core._tensor_member_fn
def cos(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_cos(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("sine")
@core._tensor_member_fn
def sin(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_sin(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("fast square root")
@core._tensor_member_fn
def sqrt(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_sqrt(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("precise square root (rounding to nearest wrt the IEEE standard)")
@core._tensor_member_fn
def sqrt_rn(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_precise_sqrt(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("inverse square root")
@core._tensor_member_fn
def rsqrt(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_rsqrt(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_2arg_docstr("precise division (rounding to nearest wrt the IEEE standard)")
def div_rn(x, y, _builder=None):
    x = semantic.to_tensor(x, _builder)
    y = semantic.to_tensor(y, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    return core.tensor(_builder.create_precise_divf(x.handle, y.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("error function")
@core._tensor_member_fn
def erf(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_erf(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("error function")
@core._tensor_member_fn
def tanh(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_tanh(x.handle), x.type)

@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_1arg_docstr("floor")
@core._tensor_member_fn
def floor(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_floor(x.handle), x.type)


@core.builtin
@_add_math_1arg_docstr("ceil")
@core._tensor_member_fn
def ceil(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    if x.type.scalar.is_int():
        return x
    elif x.type.scalar.is_floating():
        return core.tensor(_builder.create_ceil(x.handle), x.type)
    raise ValueError("ceil does not support boolean type")


@core.builtin
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5"])
@_add_math_3arg_docstr("fused multiply-add")
def fma(x, y, z, _builder=None):
    x = semantic.to_tensor(x, _builder)
    y = semantic.to_tensor(y, _builder)
    z = semantic.to_tensor(z, _builder)
    x, y = core.binary_op_type_legalization(x, y, _builder)
    z, x = core.binary_op_type_legalization(z, x, _builder)
    z, y = core.binary_op_type_legalization(z, y, _builder)
    return core.tensor(_builder.create_fma(x.handle, y.handle, z.handle), x.type)


@core.builtin
@_add_math_2arg_docstr("cdiv")
@core._tensor_member_fn
def cdiv(x, div, _builder=None):
    if isinstance(x, core.constexpr):
        x = x.value
    if isinstance(div, core.constexpr):
        div = div.value
    from math import ceil as py_ceil
    if isinstance(x, numbers.Number) and isinstance(div, numbers.Number):
        if isinstance(x, bool) or isinstance(div, bool):
            raise ValueError("cdiv does not support boolean type")
        elif isinstance(x, int) and isinstance(div, int):
            res = x // div
            rem = x % div
            return res + (1 if rem != 0 else 0)
        else:
            return py_ceil(x / div)

    x = semantic.to_tensor(x, _builder)
    div = semantic.to_tensor(div, _builder)
    x_scalar_type = x.type.scalar
    div_scalar_type = div.type.scalar
    if x_scalar_type.is_bool() or div_scalar_type.is_bool():
        raise ValueError("cdiv does not support boolean type")
    elif x_scalar_type.is_int() and div_scalar_type.is_int():
        # integer cdiv: (x + div - 1) // div as before
        return semantic.floordiv(
            semantic.add(x, semantic.sub(div, 1, True, _builder), True, _builder),
            div,
            _builder
        )
    else:
        div_res = semantic.truediv(x, div, _builder)
        cdiv_res = core.tensor(_builder.create_ceil(div_res.handle), div_res.type)
        return semantic.cast(cdiv_res, x_scalar_type, _builder)