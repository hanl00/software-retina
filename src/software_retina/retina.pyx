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

from src.software_retina.utils cimport multiply_and_sum2d
from src.software_retina.utils cimport multiply_and_sum3d
from src.software_retina.utils cimport pad_coloured
from src.software_retina.utils cimport pad_grayscaled

# Original code provided by Piotr Ozimek


default_node_attributes_datadir = join(dirname(dirname(__file__)), "..",
                           'data', '5k', '5k_rf_node_attributes.pkl')
default_coefficients_datadir = join(dirname(dirname(__file__)), "..",
                             "data", "5k", "5k_rf_coefficients.pkl")

cdef class Retina:

    cdef int N, width
    cdef cnp.float64_t[:, ::1] node_attributes
    cdef cnp.int32_t[:, :, ::1] coefficients
    cdef cnp.float64_t[::1] grayscale_intensity
    cdef cnp.float64_t[:, ::1] colour_intensity

    def __init__(self):

        self.N = 5000
        self.width = 206
        self.node_attributes = np.load(default_node_attributes_datadir, allow_pickle=True)
        self.coefficients = np.load(default_coefficients_datadir, allow_pickle=True)  # np.load(default_node_attributes_datadir, allow_pickle=True)
        self._V_gray = np.zeros((1))
        self._V_coloured = np.zeros((1, 1))

    def load_node_attributes(self, input):
        if isinstance(input, np.ndarray):
            if not (input.ndim == 2 and input.shape[1] == 7):
                raise ValueError('Must be a 2 dimensional array with each row'
                                 ' b having 7 columns of node attributes')

            else:
                self.node_attributes = input
                self.N = len(self.node_attributes)
                self.width = 2*int(np.abs(self.node_attributes[:, :2]).max() +
                                   np.asarray(self.node_attributes[:, 6]).max()/2.0)

        self.node_attributes = input_node_attributes
        self.coefficients = input_coefficients
        self.N = len(input_node_attributes)
        self.width = 2*int(np.abs(input_node_attributes[:, :2]).max() +
                           input_node_attributes[:, 6].max()/2.0)
        self.grayscale_intensity = np.zeros((1))
        self.colour_intensity = np.zeros((1, 1))

    def load_node_attributes_from_path(self, filename):

        if isinstance(filename, str):
            x = np.load(filename, allow_pickle=True)
            if not (x.ndim == 2 and x.shape[1] == 7):
                raise ValueError('Must be a 2 dimensional array with each row'
                                 ' having 7 columns of node attributes')

            else:
                self.node_attributes = x
                self.N = len(self.node_attributes)
                self.width = 2*int(np.abs(self.node_attributes[:, :2]).max() +
                                   np.asarray(self.node_attributes[:, 6]).max()/2.0)

        else:
            raise TypeError('This function only accepts string path'
                            ' of a pickled file')

    def load_coefficients(self, input):
        if isinstance(input, np.ndarray):
            if not input.ndim == 3:
                raise ValueError('Must be 3 dimensional array')

            else:
                self.coefficients = input
        else:
            raise TypeError('This function only accepts numpy array')

    def load_coefficients_from_path(self, filename):

        if isinstance(filename, str):
            x = np.load(filename, allow_pickle=True)
            if not (x.ndim == 3 and x.shape[1] == x.shape[2]):
                raise ValueError('Must be a 3 dimensional array (n,k,k) with'
                                 ' n number of k*k sized kernels')

            else:
                self.coefficients = x

        else:
            raise TypeError('This function only accepts string path'
                            ' of a pickled file')

    cpdef cnp.ndarray[cnp.float64_t, ndim=1] sample_grayscale(self, cnp.ndarray[cnp.uint8_t, ndim=2] image, (int, int) fixation):

        cdef cnp.float64_t[:, ::1] node_attributes_memory_view = self.node_attributes  # canremove
        cdef cnp.int32_t[:, :, ::1] coefficients_memory_view = self.coefficients  # canremove
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        cdef int p, y1, y2, x1, x2
        cdef cnp.int32_t[:, ::1] pic
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y
        cdef cnp.ndarray[cnp.float64_t, ndim=1] V
        cdef Py_ssize_t i
        cdef float w

        p = self.width
        pic = pad_grayscaled(image, p)
        X = np.asarray(node_attributes_memory_view[:, 0]) + fixation_x + p
        Y = np.asarray(node_attributes_memory_view[:, 1]) + fixation_y + p
        V = np.empty((self.N), dtype=np.float64)

        with nogil, parallel():
            for i in prange(self.N):
                w = node_attributes_memory_view[i, 6]
                y1 = int(Y[i] - w/2+0.5)
                y2 = int(Y[i] + w/2+0.5)
                x1 = int(X[i] - w/2+0.5)
                x2 = int(X[i] + w/2+0.5)
                V[i] = multiply_and_sum2d(pic[y1:y2, x1:x2],
                                          coefficients_memory_view[i, :, :])

        self.grayscale_intensity = V

        return V

    cpdef cnp.ndarray[cnp.float64_t, ndim=2] sample_coloured(  # noqa: E225
            self,
            cnp.ndarray[cnp.uint8_t, ndim=3] image,
            (int, int) fixation):

        cdef cnp.float64_t[:, ::1] node_attributes_memory_view = self.node_attributes
        cdef cnp.int32_t[:, :, ::1] coefficients_memory_view = self.coefficients
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        cdef int p, y1, y2, x1, x2
        cdef cnp.int32_t[:, :, ::1] pic
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X  # noqa: E225
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y  # noqa: E225
        cdef Py_ssize_t i
        cdef float w
        cdef cnp.float64_t[:, ::1] V = np.empty((self.N, 3))
        cdef cnp.float64_t[::1] sum3d_return = np.empty((3))

        p = self.width
        pic = pad_coloured(image, p)
        X = np.asarray(node_attributes_memory_view[:, 0]) + fixation_x + p
        Y = np.asarray(node_attributes_memory_view[:, 1]) + fixation_y + p

        with nogil, parallel():
            for i in prange(self.N):
                w = node_attributes_memory_view[i, 6]
                y1 = int(Y[i] - w/2+0.5)
                y2 = int(Y[i] + w/2+0.5)
                x1 = int(X[i] - w/2+0.5)
                x2 = int(X[i] + w/2+0.5)
                V[i] = multiply_and_sum3d(pic[y1:y2, x1:x2, :],
                                          coefficients_memory_view[i, :, :],
                                          sum3d_return)

        self.colour_intensity = V

        return np.asarray(V)