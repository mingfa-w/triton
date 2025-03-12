import bytedance.triton.language as tl
import bytedance.triton as triton


@triton.jit
def custom_add(a_ptr):
    tl.store(a_ptr, 1.0)
