import timeit
import numpy as np
import skimage.io
import cv2
from os.path import dirname, join
import os
import pickle
import time
import scipy.io

from tessellation.ssnn import *
from rf_generation import *
from retina import *
import utils

def loadPickle(path):
        with open(path, 'rb') as handle:
            if py == 3: 
                return pickle.load(handle, encoding='latin1')
            return pickle.load(handle)

# datadir = join(dirname(dirname(__file__)), "tessellation")

########################################################
# GENERATING 5K TESSELLATION
########################################################

# nanoflann_retina = SSNN(n_nodes = 100000, fovea = 0.1, method = "nanoflann")
# nanoflann_retina.fit()
# nanoflann_tessellation = nanoflann_retina.weights
# pickle.dump(nanoflann_tessellation, open("data/100k/nanoflann_100k_tessellation.pkl", "wb"))

########################################################
# GENERATING KERNELS AND NODE ATTRIBUTES (LOC)
########################################################

# nanoflann_tessellation = np.load('data/100k/nanoflann_100k_tessellation.pkl', allow_pickle=True)
# now = time.time()
# rf_loc, rf_coeff, fov_dist_5 = rf_ozimek(nanoflann_tessellation, kernel_ratio = 3, sigma_base = 0.5, sigma_power = 1, min_rf = 1)
# print("time taken " )
# print(time.time() - now)
# pickle.dump(rf_loc, open("data/100k/100k_rf_loc.pkl", "wb"))
# pickle.dump(rf_coeff, open("data/100k/100k_rf_coeff.pkl", "wb"))
# pickle.dump(fov_dist_5, open("data/100k/100k_fov_dist_5.pkl", "wb"))

########################################################
# KERNEL GENERATION - VALIDATION (RF_COEFF IS PADDED THEREFORE WILL RETURN FALSE)
########################################################

# rf_loc = np.load('data/5k/nanoflann_5k_rf_loc.pkl', allow_pickle=True)
# rf_coeff = np.load('data/5k/nanoflann_5k_rf_coeff.pkl', allow_pickle=True)
# fov_dist_5 = np.load('data/5k/nanoflann_5k_fov_dist_5.pkl', allow_pickle=True)

# rf_loc_original = np.load('nanoflann_5k_rf_loc_original.pkl', allow_pickle=True)
# fov_dist_5_original = np.load('nanoflann_5k_fov_dist_5_original.pkl', allow_pickle=True)

# print(np.array_equal(rf_loc, rf_loc_original))
# print(np.array_equal(fov_dist_5, fov_dist_5_original))

########################################################
# RETINA SAMPLING - TIME TAKEN
########################################################

R = Retina()
R.load_loc_from_path('data/50k/ozimek original/50k_rf_loc(generated).pkl')
R.load_coeff_from_path('data/50k/ozimek original/50k_rf_coeff(generated).pkl')

# # #Prepare retina
# # x = campic.shape[1]/2
# # y = campic.shape[0]/2
# # fixation = (y,x)

# # R.prepare(campic.shape, fixation)

baseline_image = skimage.io.imread('dock.jpg')

# baseline_image = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY)

if baseline_image.ndim == 2:
    print("Sampling grayscale")
    now = time.time()
    V = R.sample_grayscale(baseline_image, (360.0, 640.0))
    print(time.time()-now)
    # pickle.dump( V, open( "dock_sample_gray.pkl", "wb" ) )
else: 
    print("Sampling colored")
    now = time.time()
    V = R.sample_coloured(baseline_image, (360.0, 640.0))
    print(time.time()-now)
    # pickle.dump( V, open( "dock_sample_colored.pkl", "wb" ) )

########################################################
# RETINA SAMPLING - VALIDATION
########################################################

# original_gray_V = np.load('dock_sample_gray_original.pkl', allow_pickle=True)
# original_color_V = np.load('dock_sample_colored_original.pkl', allow_pickle=True)
# gray_V = np.load('dock_sample_gray.pkl', allow_pickle=True)
# color_V = np.load('dock_sample_colored.pkl', allow_pickle=True)

# 1d array
# for index, (first, second) in enumerate(zip(original_gray_V, gray_V)): 
#     if abs(first-second) > 0.0005:
#         print(index, first, second)

# 2d array
# for i in range(len(color_V)):
#     for index, (first, second) in enumerate(zip(original_color_V[i], color_V[i])):
#         if abs(first-second) > 0.0005:
#             print(index, first, second)

# x = np.asarray(utils.pad_colored(baseline_image, 100))
# pickle.dump( x, open( "pad_V2_color.pkl", "wb" ) )

# pad = np.load('pad.pkl', allow_pickle=True)
# pad2 = np.load('pad_V2_color.pkl', allow_pickle=True)
# pad_original = np.load('pad_original_color.pkl', allow_pickle=True)

# print((pad_original==pad2).all())
