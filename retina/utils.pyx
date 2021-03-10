# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

import numpy as np
cimport numpy as cnp 

cpdef cnp.int32_t[:,::1] pad_grayscaled (cnp.ndarray[cnp.uint8_t, ndim=2] img, int padding): #changed here 64 to 32
   
    cdef cnp.int32_t[:, ::1] image_mem_view = img.astype(dtype=np.int32) #changed here 64 to 32
    cdef int firstDimension = img.shape[0]
    cdef int firstAccumulator = 0
    cdef int secondDimension = img.shape[1]
    cdef int secondAccumulator = 0
    cdef int padding_twice = 2*padding
    cdef cnp.int32_t[:,::1] out #changed here 64 to 32
    cdef (int, int) size = (0, 0)
    cdef Py_ssize_t i
    cdef int imgDimension = img.ndim

    for i in range(imgDimension):
        if i == 1: 
            firstAccumulator += firstDimension + padding_twice
        else:
            secondAccumulator += secondDimension + padding_twice
    
    size = (firstAccumulator, secondAccumulator)
    out = np.zeros(size, dtype = np.int32)

    out[padding:-padding, padding:-padding] = image_mem_view #changed here 64 to 32
    
    return out

cpdef cnp.int32_t[:,:,::1] pad_colored (cnp.ndarray[cnp.uint8_t, ndim=3] img, int padding): #changed here 64 to 32
   
    cdef cnp.int32_t[:,:,::1] image_mem_view = img.astype(dtype=np.int32) #changed here 64 to 32
    cdef int firstDimension = img.shape[0]
    cdef int firstAccumulator = 0
    cdef int secondDimension = img.shape[1]
    cdef int secondAccumulator = 0
    cdef int thirdDimension = img.shape[2]
    cdef int padding_twice = 2*padding
    cdef cnp.int32_t[:,:,::1] out #changed here 64 to 32
    cdef (int, int, int) size = (0, 0, 0)
    cdef Py_ssize_t i
    cdef int imgDimension = img.ndim

    for i in range(imgDimension):
        if i == 1: 
            firstAccumulator += firstDimension + padding_twice
        if i == 2:
            secondAccumulator += secondDimension + padding_twice        
    
    size = (firstAccumulator, secondAccumulator, thirdDimension)
    out = np.zeros(size, dtype = np.int32)

    out[padding:-padding, padding:-padding, :] = image_mem_view

    # image_mem_view =  out
    
    return out

cdef double multiply_sum2d(cnp.int32_t[:, ::1] extract_image, cnp.int32_t[:,::1] coeff_mem_view) nogil:
    cdef size_t i, I, j, J
    # cdef float element_from_mem = coeff_mem_view[i, j]/100000000
    cdef double total = 0
    cdef signed long long x
    cdef signed long long y
    # cdef int current = 0
    I = extract_image.shape[0]
    J = extract_image.shape[1]
    
    for i in range(I):
        for j in range(J):
            # print(element_from_mem)
            # with gil:
                x = extract_image[i, j]
                y = coeff_mem_view[i, j]
                total += x*y
                # total = total/10
            # total += current

    return total/100000000 #8 0


cdef cnp.float64_t [::1] multiply_sum3d(cnp.int32_t[:, :, ::1] extract_image, cnp.int32_t[:,::1] coeff_mem_view, cnp.float64_t [::1] sum3d_return) nogil:
    cdef size_t i, I, j, J
    cdef double total = 0
    # cdef signed long long x
    # cdef signed long long y
    I = extract_image.shape[0]
    J = extract_image.shape[1]
    K = extract_image.shape[2]
    cdef signed long long x
    cdef signed long long y
    cdef signed long long column_0 = 0
    cdef signed long long column_1 = 0
    cdef signed long long column_2 = 0
    cdef double total_column_0 = 0
    cdef double total_column_1 = 0
    cdef double total_column_2 = 0

    # cdef cnp.float64_t[::1] intensity_row = None
    for i in range(I):
        for j in range(J):
            for k in range(K):
                if k == 0:
                    x = extract_image[i, j, k]
                    y = coeff_mem_view[i, j]
                    total_column_0 += x*y
                if k == 1:
                    x = extract_image[i, j, k]
                    y = coeff_mem_view[i, j]
                    total_column_1 += x*y
                if k == 2:
                    x = extract_image[i, j, k]
                    y = coeff_mem_view[i, j]
                    total_column_2 += x*y
    
    sum3d_return[0] = total_column_0/100000000
    sum3d_return[1] = total_column_1/100000000
    sum3d_return[2] = total_column_2/100000000

    return sum3d_return