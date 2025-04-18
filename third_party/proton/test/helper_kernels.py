import bytedtriton.language as tl
import bytedtriton as triton


@triton.jit
def custom_add(a_ptr):
    tl.store(a_ptr, 1.0)
