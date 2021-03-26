import timeit
import numpy as np
import skimage.io
import cv2
from os.path import dirname, join
import os
import pickle
import time
import scipy.io

from src.software_retina.retina import *

node_attributes_50k = np.load('data/50k/50k_rf_node_attributes.pkl', allow_pickle=True)
coefficients_50k = np.load('data/50k/50k_rf_coefficients.pkl', allow_pickle=True)

R = Retina(node_attributes_50k, coefficients_50k)

baseline_image = skimage.io.imread('data/dock.jpg')

baseline_image = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY)

if baseline_image.ndim == 2:
    print("Sampling grayscale")
    now = time.time()
    V = R.sample_grayscale(baseline_image, (360.0, 640.0))
    print(time.time()-now)
    pickle.dump( V, open( "validation testing/50k_dock_sample_gray_cython.pkl", "wb" ) )
else: 
    print("Sampling colored")
    now = time.time()
    V = R.sample_coloured(baseline_image, (360.0, 640.0))
    print(time.time()-now)
    pickle.dump( V, open( "validation testing/50k_dock_sample_colored_cython.pkl", "wb" ) )

########################################################
# RETINA SAMPLING - VALIDATION - GRAY
########################################################

# print("validation gray")
# original_gray = np.load("validation testing/ozimek's values/dock_sample_gray_50k.pkl", allow_pickle=True)
# gray = np.load('validation testing/50k_dock_sample_gray_cython.pkl', allow_pickle=True)

# for index, (first, second) in enumerate(zip(original_gray, gray)): 
#     if abs(first-second) > 0.0005:
#         print(index, first, second)

# print("validation done")

########################################################
# RETINA SAMPLING - VALIDATION - COLOUR
########################################################

# print("validation colour")
# original_colour = np.load("validation testing/ozimek's values/dock_sample_colored_50k.pkl", allow_pickle=True)
# colour = np.load('validation testing/50k_dock_sample_colored_cython.pkl', allow_pickle=True)

# for i in range(len(colour)):
#     for index, (first, second) in enumerate(zip(original_colour[i], colour[i])):
#         if abs(first-second) > 0.0005:
#             print(index, first, second)

# print("validation done")
