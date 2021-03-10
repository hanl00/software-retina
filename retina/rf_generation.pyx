# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, exp, pi
from scipy.spatial import distance

# Original code provided by Piotr Ozimek

def rf_ozimek_cython(tessellation, double kernel_ratio, sigma_base, sigma_power, min_rf, min_kernel=3):

    rf_loc = np.zeros((len(tessellation), 7))
    rf_coeff = np.ndarray((1, len(tessellation)),dtype='object')

    row_length = 0
    column_length = 0
    output_list = []
    
    neighbourhood = 5 #5 closest nodes
    
    print("rf generation - might take a while...")
  
    length = len(tessellation)
    chunk = 5000
    num = length/chunk
    if length%chunk != 0:
        num += 1

    dist_5 = np.zeros(length, dtype='float64')
    dist = dist_5

    print(str(chunk) + " nodes in one chunk.")

    for i in range(int(num)):
        
        d = distance.cdist(tessellation[i*chunk:(i+1)*chunk], tessellation)
        s = np.sort(d)
        dist_5[i*chunk:(i+1)*chunk] = np.mean(s[:,1:neighbourhood], 1)
    
    fov_dist_5 = np.min(dist_5[:20])
    rf_loc[:,:2] = tessellation*(1/fov_dist_5)*min_rf
    print(rf_loc[:,:2])
    dist_5 = dist_5*(1/fov_dist_5)*min_rf
    rf_loc[:,3] = np.arctan2(rf_loc[:,1],rf_loc[:,0])
    print(rf_loc[:,:3])
    rf_loc[:,4] = dist_5
    print(rf_loc[:,:4])

    print("All chunks done.")
    correction = 0
    rf_loc[:,5] = sigma_base * (dist_5+correction)**sigma_power
    
    for i in range(len(tessellation)):
        k_width = max(min_kernel, int(np.ceil(kernel_ratio*rf_loc[i,4])))
        rf_loc[i,6] = k_width
        cx, cy = xy_sumitha_cython(rf_loc[i,0], rf_loc[i,1], k_width)
        rx = rf_loc[i][0] - cx
        ry = rf_loc[i][1] - cy
        loc = np.array([rx, ry])
        rf_loc[i,2] = np.linalg.norm(rf_loc[i,:2])
        rf_loc[i,0] = cx
        rf_loc[i,1] = cy
        rf_coeff[0,i] = gausskernel_cython(k_width, loc, rf_loc[i,5])
        rf_coeff[0,i] /= np.sum(rf_coeff[0,i]) ###NORMALIZATION
        if rf_coeff[0,i].shape > (row_length, column_length):
            row_length, column_length = rf_coeff[0,i].shape

    print("Padding kernels now")
    for x in rf_coeff[0]:
        if x.shape != (row_length, column_length):
            b = np.pad(x, ((0, row_length-x.shape[0]), (0, column_length-x.shape[1])), 'constant', constant_values=0)
            output_list.append(b)
        else:
            output_list.append(x)

        rf_coeff = (np.stack(output_list)*100000000).astype(np.int32) #changed here 64 to 32
    
    return rf_loc, rf_coeff, fov_dist_5


cpdef cnp.float64_t gauss_cython (cnp.float64_t sigma, cnp.float64_t x,cnp.float64_t y, int mean=0):
    cdef cnp.float64_t d

    d = sqrt(x*x + y*y)
    return exp(-(d-mean)**2/(2*sigma**2))/sqrt(2*pi*sigma**2)


cpdef cnp.ndarray[cnp.float64_t, ndim=2] gausskernel_cython(cnp.int_t width, cnp.ndarray[cnp.float64_t, ndim=1] loc, cnp.float64_t sigma):
  
    cdef cnp.ndarray[cnp.float64_t, ndim=2] k
    cdef double w, shift, dx, dy
    cdef int x, y

    w = float(width)
    #location is passed as np array [x,y]
    k = np.zeros((width, width))    
    
    shift = (w-1)/2.0

    #subpixel accurate coords of gaussian centre
    dx = loc[0] - int(loc[0])
    dy = loc[1] - int(loc[1])    
    
    for x in range(width):
        for y in range(width):
            k[y,x] = gauss_cython(sigma,(x-shift)-dx,(y-shift)-dy)
    
    return k    

def xy_sumitha_cython(x,y,k_width): # least important
    k_width = int(k_width) #this will change                 #d
    
    #if odd size mask -> round coordinates
    if k_width%2 != 0:
        cx = round(x)
        cy = round(y)
        
    #else if even size mask -> 1 decimal point coordinates (always .5)
    else:
        cx = round(x) + np.sign(x-round(x))*0.5
        cy = round(y) + np.sign(y-round(y))*0.5
    
    return cx, cy