from cython.parallel import parallel, prange
import numpy as np
cimport cython

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef normal_index(unsigned short[:,:] result):
    cdef unsigned short i,j
    for i in range(1080):
        for j in range(1920):
            result[i,j] = 5

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef linear_index(unsigned short[:] result_flat):
    cdef unsigned int x
    for x in range(1080*1920):
        result_flat[x] = 5

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef normal_index_multiply(unsigned char[:,:] img, unsigned short[:,:] result):
    cdef unsigned short i,j
    for i in range(1080):
        for j in range(1920):
            result[i,j] = img[i,j]*5

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef linear_index_multiply(unsigned char[:] img_flat, unsigned short[:] result_flat):
    cdef unsigned int x
    for x in range(1080*1920):
        result_flat[x] = img_flat[x]*5

cpdef numpy_multiply(img, result):
    result = img*5
    
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef multicore_multiply(unsigned int size, unsigned char[::1] img_flat, unsigned char[::1] coeffs, unsigned short[::1] result_flat):
    cdef unsigned int x
    with nogil:
        for x in range(size):
            result_flat[x] = img_flat[x]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef multAcc(unsigned char[::1] img_flat, unsigned char[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] result_flat):
    cdef unsigned int x
    with nogil:
        for x in range(img_flat.shape[0]):
            result_flat[idx[x]] += img_flat[x]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef multAccIf(unsigned char[::1] img_flat, unsigned char[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] result_flat):
    cdef unsigned int x
    with nogil:
        for x in range(img_flat.shape[0]):
            if coeffs[x] > 0:
                result_flat[idx[x]] += img_flat[x]*coeffs[x]
            
            
@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef multAccIf32(unsigned int[::1] img_flat, unsigned int[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] result_flat):
    cdef unsigned int x
    with nogil:
        for x in range(img_flat.shape[0]):
            if coeffs[x] > 0:
                result_flat[idx[x]] += img_flat[x]*coeffs[x]
            
@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef multAccRGB(unsigned char[::1] R, unsigned char[::1] G, unsigned char[::1] B, unsigned char[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] result_R, unsigned int[::1] result_G, unsigned int[::1] result_B):
    cdef unsigned int x
    with nogil:
        for x in range(R.shape[0]):
            result_R[idx[x]] += R[x]*coeffs[x]
            result_G[idx[x]] += G[x]*coeffs[x]
            result_B[idx[x]] += B[x]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef multAccRGB_cache(unsigned char[::1] R, unsigned char[::1] G, unsigned char[::1] B, unsigned char[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] result_R, unsigned int[::1] result_G, unsigned int[::1] result_B):
    cdef unsigned int x, index
    cdef unsigned char coeff
    with nogil:
        for x in range(R.shape[0]):
            index = idx[x]
            coeff = coeffs[x]
            result_R[index] += R[x]*coeff
            result_G[index] += G[x]*coeff
            result_B[index] += B[x]*coeff           

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef multAccRGB_cache_opt(unsigned char[::1] R, unsigned char[::1] G, unsigned char[::1] B, unsigned char[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] result_R, unsigned int[::1] result_G, unsigned int[::1] result_B):
    cdef unsigned int x, index
    cdef unsigned char coeff
    with nogil:
        for x in range(R.shape[0]):
            coeff = coeffs[x]
            if coeff > 0:
                index = idx[x]
                result_R[index] += R[x]*coeff
                result_G[index] += G[x]*coeff
                result_B[index] += B[x]*coeff

@cython.wraparound(False)
@cython.boundscheck(False)             
cpdef multiThread(int nThreads, unsigned char[::1] R, unsigned char[::1] G, unsigned char[::1] B, unsigned char[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] result_R, unsigned int[::1] result_G, unsigned int[::1] result_B):
    cdef unsigned int x, index
    cdef unsigned char coeff
    with nogil,parallel(num_threads=nThreads):
        for x in prange(R.shape[0]):
            coeff = coeffs[x]
            if coeff > 0:
                index = idx[x]
                result_R[index] += R[x]*coeff
                result_G[index] += G[x]*coeff
                result_B[index] += B[x]*coeff
                
@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef backProject(unsigned int[::1] result_flat, unsigned char[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] back_projected):
    cdef unsigned int x
    with nogil:
        for x in range(back_projected.shape[0]):
            if coeffs[x] > 0:
                 back_projected[x] += result_flat[idx[x]]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef backProjectRGB(unsigned int[::1] result_R, unsigned int[::1] result_G, unsigned int[::1] result_B, unsigned char[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] BP_R, unsigned int[::1] BP_G, unsigned int[::1] BP_B):
    cdef unsigned int x, index
    cdef unsigned char coeff
    with nogil:
        for x in range(BP_R.shape[0]):
            coeff = coeffs[x]
            if coeff > 0:
                index = idx[x]
                BP_R[x] += result_R[index]*coeffs[x]
                BP_G[x] += result_G[index]*coeffs[x]
                BP_B[x] += result_B[index]*coeffs[x]