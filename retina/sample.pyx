# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

import numpy as np
import sys
import itertools
from cython.parallel import parallel
from cython.parallel import prange
from os.path import dirname
from os.path import join

cimport cython
cimport numpy as cnp

# Original code provided by Piotr Ozimek

py = sys.version_info.major

if py == 2: import cPickle as pickle
elif py == 3: 
    import torch
    import pickle

datadir = join(dirname(dirname(__file__)), "cythonised_retina")

def loadPickle(path):
        with open(path, 'rb') as handle:
            if py == 3: 
                return pickle.load(handle, encoding='latin1')
            return pickle.load(handle)

cdef class Retina:
    cdef int N, width
    cdef cnp.float64_t[:, ::1] loc
    cdef cnp.int32_t[:, :, ::1] coeff
    cdef cnp.float64_t[::1] _V_gray
    cdef cnp.float64_t[:, ::1] _V_colored

    def __init__(self):

        self.N = 0
        self.width = 0
        self.loc = np.load('../data/50k/50k_rf_loc.pkl', allow_pickle=True)
        self.coeff =  np.load('../data/50k/50k_rf_coeff.pkl', allow_pickle=True)
        self._V_gray = np.zeros((1), dtype=np.float64)
        self._V_colored = np.zeros((1, 1), dtype=np.float64)


    def load_loc_from_path(self, input):
        if isinstance(input, str):
            self.loc = np.load(path, allow_pickle=True)
        else:
            raise ValueError("Only accepting string path of a pickled file")
        self.N = len(self.loc)
        self.width = 2*int(np.abs(self.loc[:,:2]).max() + np.asarray(self.loc[:,6]).max()/2.0)

    def load_coeff(self, input):
        if isinstance(input, str):
            self.coeff = loadPickle(input)
        elif isinstance(input, np.ndarray):
            self.coeff = input
        else:
            print("Only accepting pickled/ normal arrays")
            
    cpdef cnp.ndarray[cnp.float64_t, ndim=1] sample_grayscale (self, cnp.ndarray[cnp.uint8_t, ndim=2] image, (int, int) fixation):
        """Sample an image"""
        cdef cnp.float64_t [:,::1] loc_memory_view = self.loc
        cdef cnp.int32_t [:,:,::1] coeff_memory_view = self.coeff
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        cdef int p
        cdef cnp.int32_t[:, ::1] pic
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y
        cdef cnp.ndarray[cnp.float64_t, ndim=1] V

        cdef Py_ssize_t i
        cdef float w
        cdef int y1
        cdef int x1
        cdef int y2
        cdef int x2

        p = self.width

        pic = pad_grayscaled(image, p)
    
        X = np.asarray(loc_memory_view[:,0]) + fixation_x + p
        Y = np.asarray(loc_memory_view[:,1]) + fixation_y + p
        V = np.empty((self.N), dtype=np.float64)
        
        with nogil, parallel():
            for i in prange(self.N):
                w = loc_memory_view[i, 6]
                y1 = int(Y[i] - w/2+0.5)
                y2 = int(Y[i] + w/2+0.5)
                x1 = int(X[i] - w/2+0.5)
                x2 = int(X[i] + w/2+0.5)
                V[i] = multiply_sum2d(pic[y1:y2,x1:x2], coeff_memory_view[i,:,:])

        self._V_gray = V
       
        return V

    cpdef cnp.ndarray[cnp.float64_t, ndim=2] sample_colored (self, cnp.ndarray[cnp.uint8_t, ndim=3] image, (int, int) fixation):
        """Sample an image"""
        cdef cnp.float64_t [:,::1] loc_memory_view = self.loc
        cdef cnp.int32_t [:,:,::1] coeff_memory_view = self.coeff
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        cdef cnp.int32_t[:,:, ::1] pic
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y
        cdef Py_ssize_t i
        cdef float w
        cdef int y1
        cdef int x1
        cdef int y2
        cdef int x2
        cdef cnp.float64_t [:,::1] V = np.empty((self.N, 3), dtype=np.float64)
        cdef cnp.float64_t [::1] sum3d_return = np.empty((3), dtype=np.float64)

        # self.validate()
        # self._fixation = fixation

        pic = pad_colored(image, self.width)

        X = np.asarray(loc_memory_view[:,0]) + fixation_x + self.width
        Y = np.asarray(loc_memory_view[:,1]) + fixation_y + self.width

        # V = np.empty((self.N, 3), dtype=np.float64)
        
        with nogil, parallel():
            for i in prange(self.N):
                w = loc_memory_view[i, 6]
                y1 = int(Y[i] - w/2+0.5)
                y2 = int(Y[i] + w/2+0.5)
                x1 = int(X[i] - w/2+0.5)
                x2 = int(X[i] + w/2+0.5)
                V[i] = multiply_sum3d(pic[y1:y2,x1:x2,:], coeff_memory_view[i,:,:], sum3d_return)

        self._V_colored = V

        return np.asarray(V)
    
cpdef cnp.int32_t[:,::1] pad_grayscaled (cnp.ndarray[cnp.uint8_t, ndim=2] img, int padding):
    cdef cnp.int32_t[:, ::1] image_mem_view = img.astype(dtype=np.int32)
    cdef int first_dimension = img.shape[0]
    cdef int first_accumulator = 0
    cdef int second_dimension = img.shape[1]
    cdef int second_accumulator = 0
    cdef int padding_twice = 2*padding
    cdef cnp.int32_t[:,::1] out
    cdef Py_ssize_t i
    cdef int img_dimension = img.ndim

    for i in range(img_dimension):
        if i == 1: 
            first_accumulator += first_dimension + padding_twice
        else:
            second_accumulator += second_dimension + padding_twice
    
    out = np.zeros((first_accumulator, second_accumulator), dtype = np.int32)

    out[padding:-padding, padding:-padding] = image_mem_view 
    
    return out

cpdef cnp.int32_t[:,:,::1] pad_colored (cnp.ndarray[cnp.uint8_t, ndim=3] img, int padding):
    cdef cnp.int32_t[:,:,::1] image_mem_view = img.astype(dtype=np.int32) 
    cdef int first_dimension = img.shape[0]
    cdef int first_accumulator = 0
    cdef int second_dimension = img.shape[1]
    cdef int second_accumulator = 0
    cdef int third_dimension = img.shape[2]
    cdef int padding_twice = 2*padding
    cdef cnp.int32_t[:,:,::1] out
    cdef Py_ssize_t i
    cdef int img_dimension = img.ndim

    for i in range(img_dimension):
        if i == 1: 
            first_accumulator += first_dimension + padding_twice
        if i == 2:
            second_accumulator += second_dimension + padding_twice        
    
    out = np.zeros((first_accumulator, second_accumulator, third_dimension), dtype = np.int32)

    out[padding:-padding, padding:-padding, :] = image_mem_view
    
    return out

cdef double multiply_sum2d(cnp.int32_t[:, ::1] extract_image, cnp.int32_t[:,::1] coeff_mem_view) nogil:
    cdef size_t i, I, j, J
    cdef double total = 0
    cdef signed long long x
    cdef signed long long y
    I = extract_image.shape[0]
    J = extract_image.shape[1]
    
    for i in range(I):
        for j in range(J):
                x = extract_image[i, j]
                y = coeff_mem_view[i, j]
                total += x*y

    return total/100000000


cdef cnp.float64_t [::1] multiply_sum3d(cnp.int32_t[:, :, ::1] extract_image, cnp.int32_t[:,::1] coeff_mem_view, cnp.float64_t [::1] sum3d_return) nogil:
    cdef size_t i, I, j, J
    cdef double total = 0
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