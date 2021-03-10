import timeit
import numpy as np
import skimage.io
import cv2
from os.path import dirname, join
import os
import pickle
import time

from tessellation.ssnn import *
from rf_generation import *
from sample import *

def loadPickle(path):
        with open(path, 'rb') as handle:
            if py == 3: 
                return pickle.load(handle, encoding='latin1')
            return pickle.load(handle)

# datadir = join(dirname(dirname(__file__)), "tessellation")


# nanoflann_retina = SSNN(n_nodes = 5000, fovea = 0.1, method = "nanoflann")
# nanoflann_retina.fit()
# nanoflann_tessellation = nanoflann_retina.weights
# pickle.dump( nanoflann_tessellation, open( "data/5k/nanoflann_5k_tessellation.pkl", "wb" ) )
# arr = os.listdir()
# print(arr)

# nanoflann_tessellation = np.load('data/5k/current/nanoflann_5k_tessellation.pkl', allow_pickle=True)
# rf_ozimek_cython(nanoflann_tessellation, kernel_ratio = 3, sigma_base = 0.5, sigma_power = 1, min_rf = 1)

# pickle.dump( rf_loc, open( "data/5k/nanoflann_5k_rf_loc.pkl", "wb" ) )
# pickle.dump( rf_coeff_grayscale, open( "data/5k/nanoflann_5k_rf_coeff_grayscale.pkl", "wb" ) )
# pickle.dump( rf_coeff_colored, open( "data/5k/nanoflann_5k_rf_coeff_colored.pkl", "wb" ) )
# pickle.dump( fov_dist_5, open( "data/5k/nanoflann_5k_fov_dist_5.pkl", "wb" ) )

# print(rf_loc.shape)
# print(rf_coeff_grayscale.shape)
# print(rf_coeff_grayscale.ndim)
# print(rf_coeff_colored.shape)
# print(rf_coeff_colored.ndim)
# print(fov_dist_5)

# file0 = np.load('data/5k/nanoflann_5k_rf_loc.pkl', allow_pickle=True)
# file1 = np.load('data/5k/current/nanoflann_5k_rf_loc.pkl', allow_pickle=True)


# for index, (first, second) in enumerate(zip(file0, file1)):
#     for idx in range(len(first)):
#         if first[idx] != second[idx]:
#             print(idx, first[idx], second[idx])
########################################################
# SAMPLING
########################################################

R = Retina()
# R.info()
R.loadLoc(join(datadir, "data", "5k", "current", "nanoflann_5k_rf_loc.pkl"))
R.loadCoeff(join(datadir, "data", "5k", "current", "nanoflann_5k_rf_coeff_grayscale.pkl"))
# R.loadCoeffColor(join(datadir, "data", "5k", "nanoflann_5k_rf_coeff_colored.pkl"))

# #Prepare retina
# x = campic.shape[1]/2
# y = campic.shape[0]/2
# fixation = (y,x)

# R.prepare(campic.shape, fixation)

baseline_image = skimage.io.imread('dock.jpg')

# img = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY)
# # V = R.sample(img, fixation)
now = time.time()
V = R.sample_colored(baseline_image, (360.0, 640.0))
print(time.time()-now)


########################################################
# TIMEIT SAMPLE
########################################################

print(timeit.timeit('R.sample_colored(baseline_image, (360.0, 640.0))'))
# print("done")
# pickle.dump( V, open( "dock_sample_gray_cythonised.pkl", "wb" ) )


########################################################
# COMPARING OUTPUT
########################################################

# original = np.load('dock_sample_gray.pkl', allow_pickle=True)
# cythonised = np.load('dock_sample_gray_cythonised.pkl', allow_pickle=True)

# for index, (first, second) in enumerate(zip(original, cythonised)):
#     if abs(int(first) - int(second)) > 1:
#         print(index, first, second)