# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

import numpy as np
import sys
import itertools

cimport cython
cimport numpy as cnp 

from cython.parallel import parallel, prange
from os.path import dirname, join

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

cdef class Retina:
    cdef int N, width
    # cdef (int, int) _fixation only required in backprojection
    # cdef (int, int) _imsize only required in backprojection
    # cdef (int, int) _normFixation only required in backprojection
    cdef cnp.float64_t[:, ::1] loc
    cdef cnp.int32_t[:, :, ::1] coeff
    # cdef cnp.int32_t[:, :, :, ::1] coeff_colored
    cdef cnp.float64_t[::1] _V_gray
    cdef cnp.float64_t[:,::1] _V_colored

    def __init__(self):

        self.N = 0
        self.width = 0
        self.loc = np.zeros((1,1), dtype=np.float64)
        self.coeff = np.zeros((1,1,1), dtype=np.int32)
        # self.coeff_colored = np.zeros((1,1,1,1), dtype=np.int32)
        self._V_gray = np.zeros((1), dtype=np.float64)
        self._V_colored = np.zeros((1,1), dtype=np.float64)


    def loadLoc(self, input): #only accept path or array
        if isinstance(input, str):
            self.loc = loadPickle(input)
        else:
            self.loc = input
        self.N = len(self.loc)
        self.width = 2*int(np.abs(self.loc[:,:2]).max() + np.asarray(self.loc[:,6]).max()/2.0) ##issue here

    def loadCoeff(self, input):
        if isinstance(input, str):
            self.coeff = loadPickle(input)
        elif isinstance(input, np.ndarray):
            self.coeff = input
        else:
            print("Only accepting pickled/ normal arrays")
            

    # def loadCoeffColor(self, input):
    #     if isinstance(input, str):
    #         self.coeff_colored = loadPickle(input)
    #     else:
    #         self.coeff_colored = input


    # image is a numpy ndarray dtype int16 dimension is 2 (3 if rgb) and fixation is a tuple
    cpdef cnp.ndarray[cnp.float64_t, ndim=1] sample_grayscale (self, cnp.ndarray[cnp.uint8_t, ndim=2] image, (int, int) fixation):
                                                    # cnp.ndarray[cnp.int32_t, ndim=3] coeff_padded, 
                                                    # cnp.ndarray[cnp.float64_t, ndim=2] loc):
        """Sample an image"""
        cdef cnp.float64_t [:,::1] loc_memory_view = self.loc
        cdef cnp.int32_t [:,:,::1] coeff_memory_view = self.coeff #changed here 64 to 32
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        # cdef (int, int) fix = (fixation_y, fixation_x)
        cdef int p
        cdef cnp.int32_t[:, ::1] pic #changed here 64 to 32
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y
        cdef cnp.ndarray[cnp.float64_t, ndim=1] V

        cdef Py_ssize_t i
        cdef float w
        cdef int y1
        cdef int x1
        cdef int y2
        cdef int x2


        # self.validate()
        # self._fixation = fixation

        p = self.width

        pic = pad_grayscaled(image, p)
    
        X = np.asarray(loc_memory_view[:,0]) + fixation_x + p
        Y = np.asarray(loc_memory_view[:,1]) + fixation_y + p

        #print(np.asarray(loc_memory_view[:,0]))# [ 15.5   3.    6.  ... -35.5 -39.   -4.5]
        #print(np.asarray(loc_memory_view[:,1])) #[ 43.5 -46.   -4.  ...  20.5   8.    9.5]
        V = np.empty((self.N), dtype=np.float64)
        
        with nogil, parallel():
            for i in prange(self.N):
                w = loc_memory_view[i, 6]
                y1 = int(Y[i] - w/2+0.5)
                y2 = int(Y[i] + w/2+0.5)
                x1 = int(X[i] - w/2+0.5)
                x2 = int(X[i] + w/2+0.5)
                # with gil:
                #     print(w, Y[i], y1, y2, X[i], x1, x2)
                V[i] = multiply_sum2d(pic[y1:y2,x1:x2], coeff_memory_view[i,:,:])

        self._V_gray = V
       
        return V

    ############################################
    # color
    ############################################
    cpdef cnp.ndarray[cnp.float64_t, ndim=2] sample_colored (self, cnp.ndarray[cnp.uint8_t, ndim=3] image, (int, int) fixation):
        """Sample an image"""
        cdef cnp.float64_t [:,::1] loc_memory_view = self.loc
        cdef cnp.int32_t [:,:,::1] coeff_memory_view = self.coeff #changed here 64 to 32
        cdef int fixation_y = fixation[0]
        cdef int fixation_x = fixation[1]
        # cdef (int, int) fix = (fixation_y, fixation_x)
        # cdef int p
        cdef cnp.int32_t[:,:, ::1] pic #changed here 64 to 32
        cdef cnp.ndarray[cnp.float64_t, ndim=1] X
        cdef cnp.ndarray[cnp.float64_t, ndim=1] Y
        # cdef cnp.ndarray[cnp.float64_t, ndim=2] V
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

        # This will reset the image size only when it was changed.
        # if self._imsize != image.shape[:2]:
        #     self._imsize = image.shape
        # p = self.width

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
                # with gil:
                #     print(np.asarray(pic[y1:y2,x1:x2,:]))
                #     print("kernel")
                #     print(np.asarray(coeff_memory_view[i,:,:]))
                V[i] = multiply_sum3d(pic[y1:y2,x1:x2,:], coeff_memory_view[i,:,:], sum3d_return)

        self._V_colored = V

        return np.asarray(V)  #convert mem view to array

    
cpdef cnp.int32_t[:,::1] pad_grayscaled (cnp.ndarray[cnp.uint8_t, ndim=2] img, int padding):
    cdef cnp.int32_t[:, ::1] image_mem_view = img.astype(dtype=np.int32)
    cdef int firstDimension = img.shape[0]
    cdef int firstAccumulator = 0
    cdef int secondDimension = img.shape[1]
    cdef int secondAccumulator = 0
    cdef int padding_twice = 2*padding
    cdef cnp.int32_t[:,::1] out
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

cpdef cnp.int32_t[:,:,::1] pad_colored (cnp.ndarray[cnp.uint8_t, ndim=3] img, int padding):
    cdef cnp.int32_t[:,:,::1] image_mem_view = img.astype(dtype=np.int32) 
    cdef int firstDimension = img.shape[0]
    cdef int firstAccumulator = 0
    cdef int secondDimension = img.shape[1]
    cdef int secondAccumulator = 0
    cdef int thirdDimension = img.shape[2]
    cdef int padding_twice = 2*padding
    cdef cnp.int32_t[:,:,::1] out
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