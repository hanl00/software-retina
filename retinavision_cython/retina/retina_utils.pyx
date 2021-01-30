import numpy as np
cimport numpy as cnp # Import for NumPY C-API
cimport cython
from cython.parallel import prange
import cv2

@cython.wraparound(False)
@cython.boundscheck(False)  
cpdef cnp.float64_t[:, :] pad (cnp.float64_t[:, :] img, int padding):
   
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
    
    out = np.zeros(size, dtype = np.float64)

    out[padding:-padding, padding:-padding] = img
    
    return out

   

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef float sum2d(double[:, :] arr):
    cdef size_t i, j, I, J
    cdef float total = 0
    I = arr.shape[0]
    J = arr.shape[1]

    for i in range(I):
        for j in range(J):
                total += arr[i, j]
    return total

"""Camera and visualisation functions""" 
def camopen():
    cap = 0
    camid = 0
    cap = cv2.VideoCapture(camid)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    
    while not cap.isOpened():
        print(str(camid) + ' failed, retrying\n')
        cv2.VideoCapture(camid).release()
        cap = cv2.VideoCapture(camid)
        camid += 1  
    return cap


#Run this if cam stops working. Does not work in py3 (cam never closes?)
def camclose(cap):
    cap.release()
    cv2.destroyAllWindows()

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