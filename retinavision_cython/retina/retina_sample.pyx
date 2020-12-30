# cython: profile=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp # Import for NumPY C-API
import sys
import itertools
from libc.math cimport isnan, round
# from libc.stdlib import malloc, free
cimport cython
from cython.parallel import parallel, prange

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
        coeff = coeff.squeeze()
        # output_list_of_lists = []

        # for x in coeff[0]:
        #     one_d = x.ravel()
        #     output_list_of_lists.append(one_d)

        # output_list = [item for sublist in output_list_of_lists for item in sublist]
        # coeff = np.array(output_list)


def loadLoc():
        global loc
        loc = loadPickle(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_loc.pkl"))




cpdef cnp.float64_t[:,:] pad (cnp.ndarray[cnp.float64_t, ndim=2] img, int padding):
   
    cdef cnp.float64_t[:, :] image_mem_view = img
    cdef int firstDimension = img.shape[0]
    cdef int firstAccumulator = 0
    cdef int secondDimension = img.shape[1]
    cdef int secondAccumulator = 0
    cdef int paddingPower = 2*padding
    cdef cnp.float64_t[:,:] out
    cdef (int, int) size = (0, 0)
    cdef Py_ssize_t i
    cdef int imgDimension = img.ndim

    for i in range(imgDimension):
        if i == 1: 
            firstAccumulator += firstDimension + paddingPower
        else:
            secondAccumulator += secondDimension + paddingPower
        # add for third dimension
    
    size = (firstAccumulator, secondAccumulator)
    out = np.zeros(size, dtype = np.float64)

    out[padding:-padding, padding:-padding] = image_mem_view
    
    return out

#OH BOY PYTHON 3 SURELY HURTS
def normal_round(n):
    if n - np.floor(np.abs(n)) < 0.5:
        return np.floor(n)
    return np.ceil(n)

#i = int, r = round.
def ir(val):
    return int(normal_round(val))

#Project the source image onto the target image at the given location
def project(source, target, location, v=False):
    sh, sw = source.shape[:2]
    th, tw = target.shape[:2]
    
    #target frame
    y1 = max(0, ir(location[0] - sh/2.0))
    y2 = min(th, ir(location[0] + sh/2.0))
    x1 = max(0, ir(location[1] - sw/2.0))
    x2 = min(tw, ir(location[1] + sw/2.0))
    
    #source frame
    s_y1 = - ir(min(0, location[0] - sh/2.0 + 0.5))
    s_y2 = s_y1 + (y2 - y1)
    s_x1 = - ir(min(0, location[1] - sw/2.0 + 0.5))
    s_x2 = s_x1 + (x2 - x1)
    
    try: target[y1:y2, x1:x2] += source[s_y1:s_y2, s_x1:s_x2]
    except Exception as E:
        print(y1, y2, x1, x2)
        print(s_y1, s_y2, s_x1, s_x2)
        print(source.shape)
        print(target.shape)
        print(location)
        raise E
    
    if v:
        print(y1, y2, x1, x2)
        print(s_y1, s_y2, s_x1, s_x2)
        print(source.shape)
        print(target.shape)
        print(location)
    
    return target



cpdef float multiply2d(cnp.float64_t[:, :] extract_image, cnp.float64_t[:, :] coeff_mem_view):
    cdef size_t i, I, j, J
    cdef float total = 0
    I = extract_image.shape[0]
    J = extract_image.shape[1]
    
    for i in range(I):
        for j in range(J):  
            total += extract_image[i, j]*coeff_mem_view[i, j]

    return total

# cpdef float multiply2d(cnp.float64_t[:, :] extracted_image, cnp.float64_t[:] coeff):
#     cdef size_t i, I, j, J
#     cdef float total = 0
#     cdef cnp.ndarray[cnp.float64_t, ndim=1] array_view = np.asarray(extracted_image).ravel()
#     J = array_view.shape[0]
#     for i in range(J):
#         total += array_view[i] + coeff[i]

#     return total

cdef class Retina:
    
    cdef cnp.float64_t[:, :] _gaussNorm
    cdef cnp.float64_t[:, :] _gaussNormTight
    cdef cnp.float64_t[:] _V
    cdef cnp.uint8_t[:, :] _backprojTight
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
        self._V = np.zeros((1), dtype=np.float64)
        # self._backproj = 0 not used first
        self._backprojTight = np.zeros((926, 926), dtype=np.uint8)

    cpdef updateLoc(self):
        global loc

        self.N = len(loc)
        self.width = 2*int(np.abs(loc[:,:2]).max() + loc[:,6].max()/2.0)

    def validate(self):
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

    # image is a numpy ndarray dtype int16 dimension is 2 (3 if rgb) and fixation is a tuple
    # cnp.ndarray[cnp.float64_t, ndim=1]
    cpdef cnp.ndarray[cnp.float64_t, ndim=1] sample (self, cnp.ndarray[cnp.float64_t, ndim=2] image, (int, int) fixation):
        """Sample an image"""
        global loc
        global coeff

        cdef double [:,:] loc_memory_view = loc
        cdef object [:] coeff_memory_view = coeff
        cdef cnp.ndarray[cnp.float64_t, ndim=2] imageMemoryView = image
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        cdef (int, int) fix = (fixation_y, fixation_x)
        cdef int p
        cdef cnp.float64_t[:, :] pic
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y
        cdef cnp.ndarray[cnp.float64_t, ndim=1] V
        cdef cnp.float64_t [:] extract_1d_memory_view
        cdef float w
        cdef int y1
        cdef int x1
        cdef int y2
        cdef int x2
        cdef int total_size_counter = 0
        cdef cnp.float64_t[:, :] extract

        cdef cnp.float64_t[:, :] kernel
        cdef Py_ssize_t i, j ,k
        cdef int total = 0

        # self.validate()
        self._fixation = fixation

        # This will reset the image size only when it was changed.
        # if self._imsize != image.shape[:2]:
        #     self._imsize = image.shape
        p = self.width

        pic = pad(image, p)

        X = loc[:,0] +  np.asarray(fixation_x, dtype=np.float64) + p
        Y = loc[:,1] + np.asarray(fixation_y, dtype=np.float64) + p

        V = np.zeros((self.N))

        with nogil, parallel():
            for i in prange(self.N):
                w = loc_memory_view[i, 6]
                y1 = int(Y[i] - w/2+0.5)
                y2 = int(Y[i] + w/2+0.5)
                x1 = int(X[i] - w/2+0.5)
                x2 = int(X[i] + w/2+0.5)
                # extract = pic[y1:y2,x1:x2]
                # kernel = coeff_memory_view[i]
                with gil:
                    # print(coeff_memory_view[i], pic[y1:y2,x1:x2])
                    V[i] = multiply2d(pic[y1:y2,x1:x2], coeff_memory_view[i])
                # total_size_counter = total_size_counter + len(pic[y1:y2,x1:x2])
                # V[i] = coeff_memory_view[i]

        self._V = V
       
        return V

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
    
    