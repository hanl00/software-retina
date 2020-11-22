# cython: profile=True

import numpy as np
cimport numpy as cnp # Import for NumPY C-API
# from retina_utils import pad, sum2d
import sys
import itertools
from libc.math cimport isnan
from cython.parallel import prange

py = sys.version_info.major

if py == 2: import cPickle as pickle
elif py == 3: 
    import torch
    import pickle

from os.path import dirname, join

datadir = join(dirname(dirname(__file__)), "cython_test")

def loadPickle(path):
        with open(path, 'rb') as handle:
            if py == 3: 
                return pickle.load(handle, encoding='latin1')
            return pickle.load(handle)

coeff = 0
loc = 0

def loadCoeff():
        global coeff
        coeff = loadPickle(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_coeff.pkl"))

def getCoeff():
       global coeff
       return coeff

def loadLoc():
        global loc
        loc = loadPickle(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_loc.pkl"))

def getLoc():
       global loc
       return loc

# @cython.wraparound(False)
# @cython.boundscheck(False)  
cpdef cnp.float64_t[:, :] pad (cnp.float64_t[:, :] img, int padding, bint nans):
   
    cdef int firstDimension = img.shape[0]
    cdef int firstAccumulator = 0
    cdef int secondDimension = img.shape[1]
    cdef int secondAccumulator = 0
    cdef int paddingPower = 2*padding
    cdef cnp.float64_t[:, :] out
    cdef (int, int) size = (0, 0)
    cdef Py_ssize_t i
    cdef int imgDimension = img.ndim

    for i in prange(imgDimension, nogil = True):
        if i == 1: 
            firstAccumulator += firstDimension + paddingPower
        else:
            secondAccumulator += secondDimension + paddingPower
        # add for third dimension
    
    size = (firstAccumulator, secondAccumulator)
    
    if nans: 
        out = np.full(size, np.nan)
    else:
        out = np.zeros(size, dtype = np.float64)

    out[padding:-padding, padding:-padding] = img
    
    return out

   

# @cython.wraparound(False)
# @cython.boundscheck(False)
cpdef float sum2d(double[:, :] arr):
    cdef size_t i, j, I, J
    cdef float total = 0
    I = arr.shape[0]
    J = arr.shape[1]

    for i in range(I):
        for j in range(J):
                total += arr[i, j]
    return total

cpdef cnp.float64_t[:, :] cython_nan_to_num (cnp.float64_t[:, :] arr):

    cdef size_t i, j, I, J

    I = arr.shape[0]
    J = arr.shape[1]

    for i in range(I):
        for j in range(J):
            if isnan(arr[i][j]):
                arr[i][j] = 0.0

    return arr

cdef class Retina:
    
    cdef cnp.float64_t[:, :] _gaussNorm
    cdef cnp.float64_t[:, :] _gaussNormTight
    cdef cnp.float64_t[:] _V
    cdef int N, width
    cdef (int, int) _fixation
    cdef (int, int) _imsize
    cdef (int, int) _normFixation

    def __init__(self):

        self.N = 0
        self.width = 0

        self._fixation = (0,0)
        self._imsize = (720, 1280)
        self._gaussNorm = np.empty((720, 1280), dtype=float)
        self._gaussNormTight = np.empty((926, 926), dtype=float)
        self._normFixation = (0,0)
        self._V = np.zeros((1), dtype=float)
        # self._backproj = 0 not used first
        # self._backprojTight = 0 not used first

    cpdef updateLoc(self):
        global loc

        self.N = len(loc)
        self.width = 2*int(np.abs(loc[:,:2]).max() + loc[:,6].max()/2.0)

    # cpdef validate(self):
    #     global loc
    #     global coeff

    #     assert(len(loc) == len(coeff[0]))
    #     if self._gaussNormTight is 0: 
    #         self._normTight()

    # image is a numpy ndarray dtype uint8 dimension is 2 (3 if rgb) and fixation is a tuple
    cpdef cnp.ndarray[cnp.float64_t, ndim=2] sample (self, cnp.ndarray[cnp.float64_t, ndim=2] image, (int, int) fixation):
        """Sample an image"""
        cdef cnp.float64_t[:, :] imageMemoryView = image
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        cdef (int, int) fix = (fixation_y, fixation_x)
        cdef bint rgb
        cdef int p
        cdef cnp.float64_t[:, :] pic
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y
        cdef cnp.ndarray[cnp.float64_t, ndim=1] V
        cdef cnp.float64_t w
        cdef int y1
        cdef int x1
        cdef int y2
        cdef int x2
        cdef cnp.float64_t[:, :] extract
        cdef cnp.float64_t[:, :] nan_to_num_retval
        cdef cnp.ndarray[cnp.float64_t, ndim=2] c
        cdef cnp.ndarray[cnp.float64_t, ndim=2] kernel
        cdef cnp.ndarray[cnp.float64_t, ndim=2] m
        cdef cnp.float64_t f
        cdef Py_ssize_t i


        global loc
        global coeff

        # self.validate() to add later
        self._fixation = fixation

        # This will reset the image size only when it was changed.
        # if self._imsize != image.shape[:2]:
        #     self._imsize = image.shape

        rgb = imageMemoryView.ndim == 3 and imageMemoryView.shape[-1] == 3
        p = self.width
        pic = pad(image, p, True)
        
        X = loc[:,0] +  np.asarray(fixation_x, dtype=np.float64) + p
        Y = loc[:,1] + np.asarray(fixation_y, dtype=np.float64) + p
        
        if rgb: 
            V = np.zeros((self.N,3))
        else: 
            V = np.zeros((self.N))

        for i in range(0 ,self.N):
            w = loc[i,6]
            y1 = int(Y[i] - w/2+0.5)
            y2 = int(Y[i] + w/2+0.5)
            x1 = int(X[i] - w/2+0.5)
            x2 = int(X[i] + w/2+0.5)

            extract = pic[y1:y2,x1:x2]
            #np.unravel_index(np.where(data.ravel()==10),data.shape)
            c = coeff[0, i]
            
            if rgb: 
                kernel = np.dstack((c,c,c))
            else: 
                kernel = c

            m = np.where(np.isnan(extract), 0, 1.0)

            if rgb: 
                f = 1.0/np.sum(m*kernel, axis = (0,1))
            else: 
                f = 1.0/sum2d(m*kernel)

            nan_to_num_retval = cython_nan_to_num(extract)
            
            if rgb: 
                V[i] = np.sum(nan_to_num_retval*kernel, axis=(0,1)) * f
            else: 
                V[i] = sum2d(nan_to_num_retval*kernel) * f
       
        self._V = V
        
        return V
    
    