# cython: profile=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

import numpy as np
import sys
import itertools

cimport cython
cimport numpy as cnp 

from cython.parallel import parallel, prange
from os.path import dirname, join

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

coeff = 0
loc = 0

def loadCoeff():
        global coeff
        coeff = loadPickle(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_coeff.pkl"))
        output_list = []

        #get the largest size
        desired_rows, desired_columns = 0, 0

        for item in coeff[0]:
            if item.shape > (desired_rows, desired_columns):
                desired_rows, desired_columns = item.shape
    
        for x in coeff[0]:
            if x.shape != (desired_rows, desired_columns):
                b = np.pad(x, ((0, desired_rows-x.shape[0]), (0, desired_columns-x.shape[1])), 'constant', constant_values=0)
                output_list.append(b)
            else:
                output_list.append(x)

        coeff = (np.stack(output_list) * 10000).astype(int)

def loadLoc():
        global loc
        loc = loadPickle(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_loc.pkl"))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.profile(True)
cpdef cnp.int32_t[:,:] pad (cnp.ndarray[cnp.int32_t, ndim=2] img, int padding):
   
    cdef cnp.int32_t[:, :] image_mem_view = img
    cdef int firstDimension = img.shape[0]
    cdef int firstAccumulator = 0
    cdef int secondDimension = img.shape[1]
    cdef int secondAccumulator = 0
    cdef int padding_twice = 2*padding
    cdef cnp.int32_t[:,:] out
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

    out[padding:-padding, padding:-padding] = image_mem_view
    
    return out

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.profile(True)
cdef double multiply_sum2d(cnp.int32_t[:, :] extract_image, int[:,:] coeff_mem_view) nogil:
    cdef size_t i, I, j, J
    cdef double total = 0
    cdef int current = 0
    I = extract_image.shape[0]
    J = extract_image.shape[1]
    
    for i in range(I):
        for j in range(J):
            current = extract_image[i, j]*coeff_mem_view[i, j]
            total += current/10000

    return total

cdef class Retina:
    #to-do: tidy up

    cdef cnp.float64_t[:, :] _gaussNorm #can remove
    cdef cnp.float64_t[:, :] _gaussNormTight #can remove
    # cdef cnp.int32_t[:] _V
    cdef cnp.uint8_t[:, :] _backprojTight #can remove
    cdef int N, width
    cdef (int, int) _fixation
    cdef (int, int) _imsize
    cdef (int, int) _normFixation

    def __init__(self):

        self.N = 0
        self.width = 0

        self._fixation = (0,0)
        self._imsize = (1080, 1920)
        self._gaussNorm = np.empty((1080, 1920), dtype=float)
        self._gaussNormTight = np.zeros((926, 926), dtype=float) #can be further optimised using memory view initalisation
        self._normFixation = (0,0)
        # self._V = np.zeros((1), dtype=np.int32)
        # self._backproj = 0 not used first
        self._backprojTight = np.zeros((926, 926), dtype=np.uint8)

    cpdef updateLoc(self):
        global loc

        self.N = len(loc)
        self.width = 2*int(np.abs(loc[:,:2]).max() + loc[:,6].max()/2.0)

    # image is a numpy ndarray dtype int16 dimension is 2 (3 if rgb) and fixation is a tuple
    cpdef cnp.ndarray[cnp.float32_t, ndim=1] sample (self, cnp.ndarray[cnp.int32_t, ndim=2] image, (int, int) fixation):
        """Sample an image"""
        global loc
        global coeff

        cdef double [:,:] loc_memory_view = loc
        cdef int [:,:,:] coeff_memory_view = coeff 
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        cdef (int, int) fix = (fixation_y, fixation_x)
        cdef int p
        cdef cnp.int32_t[:, :] pic
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y
        cdef cnp.ndarray[cnp.float32_t, ndim=1] V
        cdef float w
        cdef int y1
        cdef int x1
        cdef int y2
        cdef int x2
        cdef Py_ssize_t i

        # self.validate()
        self._fixation = fixation

        # This will reset the image size only when it was changed.
        # if self._imsize != image.shape[:2]:
        #     self._imsize = image.shape
        p = self.width

        pic = pad(image, p)

        X = loc[:,0] +  np.asarray(fixation_x, dtype=np.float64) + p
        Y = loc[:,1] + np.asarray(fixation_y, dtype=np.float64) + p

        V = np.zeros((self.N), dtype=np.float32)
        
        with nogil, parallel():
            for i in prange(self.N):
                w = loc_memory_view[i, 6]
                y1 = int(Y[i] - w/2+0.5)
                y2 = int(Y[i] + w/2+0.5)
                x1 = int(X[i] - w/2+0.5)
                x2 = int(X[i] + w/2+0.5)
                V[i] = multiply_sum2d(pic[y1:y2,x1:x2], coeff_memory_view[i,:,:])

        # self._V = V
       
        return V


    ###############################################################################

    cpdef validate(self):
        global loc
        global coeff

        assert(len(loc) == len(coeff))
        _gaussNormTight = np.asarray(self._gaussNormTight)
        if np.all(_gaussNormTight==0): 
            self._normTight()

    def _normTight(self): 
        """Produce a tight-fitted Gaussian normalization image (width x width)"""
        global loc
        global coeff

        GI = np.zeros((self.width, self.width))         
        r = self.width/2.0
        for i in range(self.N - 1, -1, -1): 
            GI = project(coeff[0,i], GI, loc[i,:2][::-1] + r)
        
        self._gaussNormTight = GI

    def prepare(self, shape, fix):
        """Pre-compute fixation specific Gaussian normalization image """
        fix = (int(fix[0]), int(fix[1]))
        self.validate()
        self._normFixation = fix
        
        GI = np.zeros(shape[:2])
        _gaussNormTight = np.asarray(self._gaussNormTight)
        GI = project(_gaussNormTight, GI, fix)
        self._gaussNorm = GI

    # for testing, no need to cythonise
    def backproject_tight_last(self, n=True, norm=None):
        return self.backproject_tight(self._V, self._imsize, self._fixation, normalize=n, norm=norm)
    
    def backproject_tight(self, V, shape, fix, normalize=True, norm=None):
        """Produce a tight-fitted backprojection (width x width, lens only)"""
        global loc
        global coeff

        fix = (int(fix[0]), int(fix[1]))
        #TODO: look at the weird artifacts at edges when the lens is too big for the frame. CPU version
        
        self.validate()

        if fix != self._normFixation or shape[:2] != self._gaussNorm.shape: 
            self.prepare(shape, fix)
            
        rgb = len(shape) == 3 and shape[-1] == 3
        m = self.width
        r = m/2.0    
             
        if rgb: I1 = np.zeros((m, m, 3))
        else: I1 = np.zeros((m, m))
        
        for i in range(self.N - 1,-1,-1):
            c = coeff[0, i]
            if rgb: c = np.dstack((c,c,c))
            
            I1 = project(c*V[i], I1, loc[i,:2][::-1] + r)
    
        GI = self._gaussNormTight
        if rgb: GI = np.dstack((GI,GI,GI)) #TODO: fix invalid value warnings
        if normalize: 
            I1 = np.uint8(np.true_divide(I1,GI)) 
            self._backprojTight = I1
        return I1
    
    #TODO: add the crop function. Tightly crop original image using retinal lens
    
    