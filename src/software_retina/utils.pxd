# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

cpdef cnp.int32_t[:, ::1] pad_grayscaled(cnp.ndarray[cnp.uint8_t, ndim=2] img, int padding)

cpdef cnp.int32_t[:, :, ::1] pad_coloured(cnp.ndarray[cnp.uint8_t, ndim=3] img, int padding)

cdef double multiply_and_sum2d(cnp.int32_t[:, ::1] image_extract, cnp.int32_t[:, ::1] coeff_mem_view) nogil

cdef cnp.float64_t[::1] multiply_and_sum3d(cnp.int32_t[:, :, ::1] image_extract, cnp.int32_t[:, ::1] coeff_mem_view, cnp.float64_t[::1] sum3d_return) nogil
