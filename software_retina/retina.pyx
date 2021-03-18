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
import errno
import os

cimport cython
cimport numpy as cnp

# Original code provided by Piotr Ozimek


default_loc_datadir = join(dirname(dirname(__file__)), 'cythonised_retina',
                           'data', '50k', 'nanoflann method',
                           '50k_rf_loc.pkl')
default_coeff_datadir = join(dirname(dirname(__file__)), "cythonised_retina",
                             "data", "50k", "nanoflann method",
                             "50k_rf_coeff.pkl")

cdef class Retina:
    cdef int N, width
    cdef cnp.float64_t[:, ::1] loc
    cdef cnp.int32_t[:, :, ::1] coeff
    cdef cnp.float64_t[::1] _V_gray
    cdef cnp.float64_t[:, ::1] _V_coloured

    def __init__(self):

        self.N = 0
        self.width = 0
        self.loc = np.load(default_loc_datadir, allow_pickle=True)
        self.coeff = np.load(default_coeff_datadir, allow_pickle=True)
        self._V_gray = np.zeros((1))
        self._V_coloured = np.zeros((1, 1))

    def load_loc(self, input):
        if isinstance(input, np.ndarray):
            if not (input.ndim == 2 and input.shape[1] == 7):
                raise ValueError('Must be a 2 dimensional array with each row'
                                 ' b having 7 columns of node attributes')

            else:
                self.loc = input
                self.N = len(self.loc)
                self.width = 2*int(np.abs(self.loc[:, :2]).max() +
                                   np.asarray(self.loc[:, 6]).max()/2.0)
        else:
            raise TypeError('This function only accepts numpy array')

    def load_loc_from_path(self, filename):
        if isinstance(filename, str):
            x = np.load(filename, allow_pickle=True)
            if not (x.ndim == 2 and x.shape[1] == 7):
                raise ValueError('Must be a 2 dimensional array with each row'
                                 ' having 7 columns of node attributes')

            else:
                self.loc = x
                self.N = len(self.loc)
                self.width = 2*int(np.abs(self.loc[:, :2]).max() +
                                   np.asarray(self.loc[:, 6]).max()/2.0)

        else:
            raise TypeError('This function only accepts string path'
                            ' of a pickled file')

    def load_coeff(self, input):
        if isinstance(input, np.ndarray):
            if not input.ndim == 3:
                raise ValueError('Must be 3 dimensional array')

            else:
                self.coeff = input
        else:
            raise TypeError('This function only accepts numpy array')

    def load_coeff_from_path(self, filename):
        if isinstance(filename, str):
            x = np.load(filename, allow_pickle=True)
            if not (x.ndim == 3 and x.shape[1] == x.shape[2]):
                raise ValueError('Must be a 3 dimensional array (n,k,k) with'
                                 ' n number of k*k sized kernels')

            else:
                self.coeff = x

        else:
            raise TypeError('This function only accepts string path'
                            ' of a pickled file')

    cpdef cnp.ndarray[cnp.float64_t, ndim=1] sample_grayscale(  # noqa: E225
            self,
            cnp.ndarray[cnp.uint8_t, ndim=2] image,
            (int, int) fixation):
        """Sample an image"""
        cdef cnp.float64_t[:, ::1] loc_memory_view = self.loc  # canremove
        cdef cnp.int32_t[:, :, ::1] coeff_memory_view = self.coeff  # canremove
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        cdef int p
        cdef cnp.int32_t[:, ::1] pic
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X  # noqa: E225
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y  # noqa: E225
        cdef cnp.ndarray[cnp.float64_t, ndim=1] V  # noqa: E225
        cdef Py_ssize_t i
        cdef float w
        cdef int y1
        cdef int x1
        cdef int y2
        cdef int x2

        p = self.width
        pic = pad_grayscaled(image, p)
        X = np.asarray(loc_memory_view[:, 0]) + fixation_x + p
        Y = np.asarray(loc_memory_view[:, 1]) + fixation_y + p
        V = np.empty((self.N), dtype=np.float64)

        with nogil, parallel():
            for i in prange(self.N):
                w = loc_memory_view[i, 6]
                y1 = int(Y[i] - w/2+0.5)
                y2 = int(Y[i] + w/2+0.5)
                x1 = int(X[i] - w/2+0.5)
                x2 = int(X[i] + w/2+0.5)
                V[i] = multiply_and_sum2d(pic[y1:y2, x1:x2],
                                          coeff_memory_view[i, :, :])

        self._V_gray = V

        return V

    cpdef cnp.ndarray[cnp.float64_t, ndim=2] sample_coloured(  # noqa: E225
            self,
            cnp.ndarray[cnp.uint8_t, ndim=3] image,
            (int, int) fixation):
        """Sample an image"""
        cdef cnp.float64_t[:, ::1] loc_memory_view = self.loc
        cdef cnp.int32_t[:, :, ::1] coeff_memory_view = self.coeff
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        cdef cnp.int32_t[:, :, ::1] pic
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X  # noqa: E225
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y  # noqa: E225
        cdef Py_ssize_t i
        cdef float w
        cdef int y1
        cdef int x1
        cdef int y2
        cdef int x2
        cdef cnp.float64_t[:, ::1] V = np.empty((self.N, 3))
        cdef cnp.float64_t[::1] sum3d_return = np.empty((3))

        pic = pad_coloured(image, self.width)
        X = np.asarray(loc_memory_view[:, 0]) + fixation_x + self.width
        Y = np.asarray(loc_memory_view[:, 1]) + fixation_y + self.width

        with nogil, parallel():
            for i in prange(self.N):
                w = loc_memory_view[i, 6]
                y1 = int(Y[i] - w/2+0.5)
                y2 = int(Y[i] + w/2+0.5)
                x1 = int(X[i] - w/2+0.5)
                x2 = int(X[i] + w/2+0.5)
                V[i] = multiply_and_sum3d(pic[y1:y2, x1:x2, :],
                                          coeff_memory_view[i, :, :],
                                          sum3d_return)

        self._V_coloured = V

        return np.asarray(V)

#######################################################################

cpdef cnp.int32_t[:, ::1] pad_grayscaled(cnp.ndarray[cnp.uint8_t, ndim=2] img,
                                         int padding):
    cdef cnp.int32_t[:, ::1] image_mem_view = img.astype(dtype=np.int32)
    cdef int first_dimension = img.shape[0]
    cdef int first_accumulator = 0
    cdef int second_dimension = img.shape[1]
    cdef int second_accumulator = 0
    cdef int padding_twice = 2*padding
    cdef cnp.int32_t[:, ::1] out
    cdef Py_ssize_t i
    cdef int img_dimension = img.ndim

    for i in range(img_dimension):
        if i == 1:
            first_accumulator += first_dimension + padding_twice

        else:
            second_accumulator += second_dimension + padding_twice

    out = np.zeros((first_accumulator, second_accumulator), dtype=np.int32)
    out[padding:-padding, padding:-padding] = image_mem_view

    return out

cpdef cnp.int32_t[:, :, ::1] pad_coloured(cnp.ndarray[cnp.uint8_t, ndim=3] img,
                                          int padding):
    cdef cnp.int32_t[:, :, ::1] image_mem_view = img.astype(dtype=np.int32)
    cdef int first_dimension = img.shape[0]
    cdef int first_accumulator = 0
    cdef int second_dimension = img.shape[1]
    cdef int second_accumulator = 0
    cdef int third_dimension = img.shape[2]
    cdef int padding_twice = 2*padding
    cdef cnp.int32_t[:, :, ::1] out
    cdef Py_ssize_t i
    cdef int img_dimension = img.ndim

    for i in range(img_dimension):
        if i == 1:
            first_accumulator += first_dimension + padding_twice

        if i == 2:
            second_accumulator += second_dimension + padding_twice

    out = np.zeros((first_accumulator, second_accumulator, third_dimension),
                   dtype=np.int32)
    out[padding:-padding, padding:-padding, :] = image_mem_view

    return out

cdef double multiply_and_sum2d(cnp.int32_t[:, ::1] image_extract,
                               cnp.int32_t[:, ::1] coeff_mem_view) nogil:
    cdef size_t i, first_dimension, j, second_dimension
    cdef double total = 0
    cdef signed long long x
    cdef signed long long y
    first_dimension = image_extract.shape[0]
    second_dimension = image_extract.shape[1]

    for i in range(first_dimension):
        for j in range(second_dimension):
            x = image_extract[i, j]
            y = coeff_mem_view[i, j]
            total += x*y

    return total/100000000


cdef cnp.float64_t[::1] multiply_and_sum3d(
                                        cnp.int32_t[:, :, ::1] image_extract,
                                        cnp.int32_t[:, ::1] coeff_mem_view,
                                        cnp.float64_t[::1] sum3d_return) nogil:
    cdef size_t i, first_dimension, j, second_dimension, k, third_dimension
    cdef double total = 0
    first_dimension = image_extract.shape[0]
    second_dimension = image_extract.shape[1]
    third_dimension = image_extract.shape[2]
    cdef signed long long x
    cdef signed long long y
    cdef signed long long column_0 = 0
    cdef signed long long column_1 = 0
    cdef signed long long column_2 = 0
    cdef double total_column_0 = 0
    cdef double total_column_1 = 0
    cdef double total_column_2 = 0

    for i in range(first_dimension):
        for j in range(second_dimension):
            for k in range(third_dimension):
                if k == 0:
                    x = image_extract[i, j, k]
                    y = coeff_mem_view[i, j]
                    total_column_0 += x*y

                if k == 1:
                    x = image_extract[i, j, k]
                    y = coeff_mem_view[i, j]
                    total_column_1 += x*y

                if k == 2:
                    x = image_extract[i, j, k]
                    y = coeff_mem_view[i, j]
                    total_column_2 += x*y

    sum3d_return[0] = total_column_0/100000000
    sum3d_return[1] = total_column_1/100000000
    sum3d_return[2] = total_column_2/100000000

    return sum3d_return
