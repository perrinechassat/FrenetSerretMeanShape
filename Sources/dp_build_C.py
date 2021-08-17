from cffi import FFI
import os

ffibuilder = FFI()

with open(os.path.join(os.path.dirname(__file__), "DP_C.h")) as f:
    ffibuilder.cdef(f.read())

ffibuilder.set_source("_DP_C",
    '#include "DP_C.h"',
    sources=["Sources/DP_C.c"],
    include_dirs=["Sources/"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
