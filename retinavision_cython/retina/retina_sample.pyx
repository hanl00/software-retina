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
from retina_utils import project
# from cython cimport view


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
coeff_padded_grayscale = 0
coeff_padded_colored = 0
loc = 0

def loadCoeff():
        global coeff
        global coeff_padded_grayscale
        global coeff_padded_colored

        coeff = loadPickle(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_coeff.pkl"))
        output_list_grayscale = []
        output_list_colored = []

        #get the largest size
        desired_rows, desired_columns = 0, 0

        for item in coeff[0]:
            if item.shape > (desired_rows, desired_columns):
                desired_rows, desired_columns = item.shape
    
        for x in coeff[0]:
            if x.shape != (desired_rows, desired_columns):
                b = np.pad(x, ((0, desired_rows-x.shape[0]), (0, desired_columns-x.shape[1])), 'constant', constant_values=0)
                b_stacked = np.dstack((b,b,b))
                output_list_grayscale.append(b)
                output_list_colored.append(b_stacked)
            else:
                x_stacked = np.dstack((x,x,x))
                output_list_grayscale.append(x)
                output_list_colored.append(x_stacked)

        coeff_padded_grayscale = (np.stack(output_list_grayscale)*100000000).astype(np.int32) #changed here 64 to 32
        coeff_padded_colored = (np.stack(output_list_colored)*100000000).astype(np.int32) #changed here 64 to 32

        return coeff, coeff_padded_grayscale, coeff_padded_colored #for testing only

def loadLoc():
        global loc      
        loc = loadPickle(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_loc.pkl"))

@cython.wraparound(False)
@cython.boundscheck(False)
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

    # image_mem_view =  out
    
    return out

@cython.wraparound(False)
@cython.boundscheck(False)
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

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
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


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
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

cdef class Retina:
    #to-do: tidy up

    cdef cnp.float64_t[:, :] _gaussNorm #can remove
    cdef cnp.float64_t[:, :] _gaussNormTight #can remove
    cdef cnp.float64_t[::1] _V
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
        self._V = np.zeros((1), dtype=np.float64)
        # self._backproj = 0 not used first
        self._backprojTight = np.zeros((926, 926), dtype=np.uint8)

    cpdef updateLoc(self): 
        global loc
        self.N = len(loc)
        self.width = 2*int(np.abs(loc[:,:2]).max() + loc[:,6].max()/2.0)

    # image is a numpy ndarray dtype int16 dimension is 2 (3 if rgb) and fixation is a tuple
    cpdef cnp.ndarray[cnp.float64_t, ndim=1] sample_grayscale (self, cnp.ndarray[cnp.uint8_t, ndim=2] image, (int, int) fixation):
                                                    # cnp.ndarray[cnp.int32_t, ndim=3] coeff_padded, 
                                                    # cnp.ndarray[cnp.float64_t, ndim=2] loc):
        """Sample an image"""
        global loc
        global coeff_padded_grayscale

        cdef cnp.float64_t [:,::1] loc_memory_view = loc
        cdef cnp.int32_t [:,:,::1] coeff_memory_view = coeff_padded_grayscale #changed here 64 to 32
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        cdef (int, int) fix = (fixation_y, fixation_x)
        cdef int p
        cdef cnp.int32_t[:, ::1] pic #changed here 64 to 32
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y
        cdef cnp.ndarray[cnp.float64_t, ndim=1] V
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

        pic = pad_grayscaled(image, p)

        X = loc[:,0] + fixation_x + p
        Y = loc[:,1] + fixation_y + p

        V = np.empty((self.N), dtype=np.float64)
        
        with nogil, parallel():
            for i in prange(self.N):
                w = loc_memory_view[i, 6]
                y1 = int(Y[i] - w/2+0.5)
                y2 = int(Y[i] + w/2+0.5)
                x1 = int(X[i] - w/2+0.5)
                x2 = int(X[i] + w/2+0.5)
                V[i] = multiply_sum2d(pic[y1:y2,x1:x2], coeff_memory_view[i,:,:])

        self._V = V
       
        return V

    cpdef cnp.ndarray[cnp.float64_t, ndim=2] sample_colored (self, cnp.ndarray[cnp.uint8_t, ndim=3] image, (int, int) fixation):
        """Sample an image"""
        global loc
        global coeff_padded_grayscale

        cdef cnp.float64_t [:,::1] loc_memory_view = loc
        cdef cnp.int32_t [:,:,::1] coeff_memory_view = coeff_padded_grayscale #changed here 64 to 32
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        cdef (int, int) fix = (fixation_y, fixation_x)
        cdef int p
        cdef cnp.int32_t[:,:, ::1] pic #changed here 64 to 32
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y
        # cdef cnp.ndarray[cnp.float64_t, ndim=2] V
        cdef float w
        cdef int y1
        cdef int x1
        cdef int y2
        cdef int x2
        cdef Py_ssize_t i
        cdef cnp.float64_t [:,::1] V = np.empty((self.N, 3), dtype=np.float64)
        cdef cnp.float64_t [::1] sum3d_return = np.empty((3), dtype=np.float64)

        # self.validate()
        self._fixation = fixation

        # This will reset the image size only when it was changed.
        # if self._imsize != image.shape[:2]:
        #     self._imsize = image.shape
        p = self.width

        pic = pad_colored(image, p)

        X = loc[:,0] + fixation_x + p
        Y = loc[:,1] + fixation_y + p

        # V = np.empty((self.N, 3), dtype=np.float64)
        
        with nogil, parallel():
            for i in prange(self.N):
                w = loc_memory_view[i, 6]
                y1 = int(Y[i] - w/2+0.5)
                y2 = int(Y[i] + w/2+0.5)
                x1 = int(X[i] - w/2+0.5)
                x2 = int(X[i] + w/2+0.5)
                V[i] = multiply_sum3d(pic[y1:y2,x1:x2,:], coeff_memory_view[i,:,:], sum3d_return)

        # self._V = V

        return np.asarray(V)

    cpdef cnp.int32_t[:, :] test_function (self, cnp.ndarray[cnp.uint8_t, ndim=2] image):
        cdef cnp.int32_t[:, ::1] image_mem_view_1 = image.astype(dtype=np.int32)
        cdef cnp.int32_t[:, :] image_mem_view_2 = image.view(dtype=np.int32)[:,::4]

        print("test")
        print(image_mem_view_1.shape)
        print(image_mem_view_2.shape)
        return image_mem_view_2
    ###############################################################################

    cpdef validate(self):
        global loc
        global coeff

        assert(len(loc) == len(coeff[0]))
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
        global loc
        global coeff

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
    
    