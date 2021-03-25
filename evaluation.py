import timeit
import numpy as np
import skimage.io
import cv2
from os.path import dirname, join
import os
import pickle
import time
import scipy.io

from src.software_retina_generation.ssnn import *
from src.software_retina.rf_generation import *
from src.software_retina.retina import *

def loadPickle(path):
        with open(path, 'rb') as handle:
            if py == 3: 
                return pickle.load(handle, encoding='latin1')
            return pickle.load(handle)

########################################################
# Testing nanoflann multiprocessing
########################################################

print("5000")

nanoflann_retina = SelfSimilarNeuralNetwork(node_count = 5000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann")
nanoflann_retina.fit()

nanoflann_retina_4 = SelfSimilarNeuralNetwork(node_count = 5000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann_multi_4")
nanoflann_retina_4.fit()

nanoflann_retina_8 = SelfSimilarNeuralNetwork(node_count = 5000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann_multi_8")
nanoflann_retina_8.fit()

########################################################
# Testing nanoflann multiprocessing
########################################################

print("10000")

nanoflann_retina = SelfSimilarNeuralNetwork(node_count = 10000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann")
nanoflann_retina.fit()

nanoflann_retina_4 = SelfSimilarNeuralNetwork(node_count = 10000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann_multi_4")
nanoflann_retina_4.fit()

nanoflann_retina_8 = SelfSimilarNeuralNetwork(node_count = 10000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann_multi_8")
nanoflann_retina_8.fit()


########################################################
# Testing nanoflann multiprocessing
########################################################

print("20000")

nanoflann_retina = SelfSimilarNeuralNetwork(node_count = 20000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann")
nanoflann_retina.fit()


nanoflann_retina_4 = SelfSimilarNeuralNetwork(node_count = 20000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann_multi_4")
nanoflann_retina_4.fit()

nanoflann_retina_8 = SelfSimilarNeuralNetwork(node_count = 20000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann_multi_8")
nanoflann_retina_8.fit()


########################################################
# Testing nanoflann multiprocessing
########################################################

print("50000")

nanoflann_retina = SelfSimilarNeuralNetwork(node_count = 50000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann")
nanoflann_retina.fit()


nanoflann_retina_4 = SelfSimilarNeuralNetwork(node_count = 50000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann_multi_4")
nanoflann_retina_4.fit()

nanoflann_retina_8 = SelfSimilarNeuralNetwork(node_count = 50000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann_multi_8")
nanoflann_retina_8.fit()

########################################################
# Testing nanoflann multiprocessing
########################################################

print("100000")

nanoflann_retina = SelfSimilarNeuralNetwork(node_count = 100000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann")
nanoflann_retina.fit()


nanoflann_retina_4 = SelfSimilarNeuralNetwork(node_count = 100000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann_multi_4")
nanoflann_retina_4.fit()

nanoflann_retina_8 = SelfSimilarNeuralNetwork(node_count = 100000,
                                            foveal_region_size = 0.1,
                                            nearest_neighbour_method = "nanoflann_multi_8")
nanoflann_retina_8.fit()




# datadir = join(dirname(dirname(__file__)), "tessellation")

########################################################
# GENERATING 10K TESSELLATION
########################################################

# nanoflann_retina = SelfSimilarNeuralNetwork(node_count = 10000, foveal_region_size = 0.1, nearest_neighbour_method = "nanoflann")
# nanoflann_retina.fit()
# nanoflann_tessellation = nanoflann_retina.weights
# pickle.dump(nanoflann_tessellation, open("data/10k/nanoflann_10k_tessellation.pkl", "wb"))

########################################################
# GENERATING KERNELS AND NODE ATTRIBUTES (LOC)
########################################################

# nanoflann_tessellation = np.load('data/10k/nanoflann_10k_tessellation.pkl', allow_pickle=True)
# rf_loc, rf_coeff, fov_dist_5 = rf_generation(nanoflann_tessellation, kernel_ratio = 3, sigma_base = 0.5, sigma_power = 1, min_rf = 1)
# pickle.dump(rf_loc, open("data/10k/10k_rf_node_attributes.pkl", "wb"))
# pickle.dump(rf_coeff, open("data/10k/10k_rf_coefficients.pkl", "wb"))
# pickle.dump(fov_dist_5, open("data/10k/10k_fov_dist_5.pkl", "wb"))

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
# rf_loc = np.load('data/5k/5k_rf_loc.pkl', allow_pickle=True)
# rf_coeff = np.load('data/5k/5k_rf_coeff.pkl', allow_pickle=True)

# R = Retina()
# R.load_loc(rf_loc)
# R.load_coeff(rf_coeff)

# # #Prepare retina
# # x = campic.shape[1]/2
# # y = campic.shape[0]/2
# # fixation = (y,x)

# # R.prepare(campic.shape, fixation)

# baseline_image = skimage.io.imread('data/dock.jpg')

# baseline_image = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY)

# if baseline_image.ndim == 2:
#     print("Sampling grayscale")
#     now = time.time()
#     V = R.sample_grayscale(baseline_image, (360.0, 640.0))
#     print(time.time()-now)
#     pickle.dump( V, open( "validation testing/5k_dock_sample_gray_cython.pkl", "wb" ) )
# else: 
#     print("Sampling colored")
#     now = time.time()
#     V = R.sample_coloured(baseline_image, (360.0, 640.0))
#     print(time.time()-now)
#     pickle.dump( V, open( "validation testing/5k_dock_sample_colored_cython.pkl", "wb" ) )

########################################################
# RETINA SAMPLING - VALIDATION - GRAY
########################################################

# original_gray = np.load("validation testing/ozimek's values/dock_sample_gray_5k.pkl", allow_pickle=True)
# gray = np.load('validation testing/5k_dock_sample_gray_cython.pkl', allow_pickle=True)

# for index, (first, second) in enumerate(zip(original_gray, gray)): 
#     if abs(first-second) > 0.0005:
#         print(index, first, second)


########################################################
# RETINA SAMPLING - VALIDATION - COLOUR
########################################################

# original_colour = np.load("validation testing/ozimek's values/dock_sample_colored_5k.pkl", allow_pickle=True)
# colour = np.load('validation testing/5k_dock_sample_colored_cython.pkl', allow_pickle=True)

# for i in range(len(colour)):
#     for index, (first, second) in enumerate(zip(original_colour[i], colour[i])):
#         if abs(first-second) > 0.0005:
#             print(index, first, second)

# x = np.asarray(utils.pad_colored(baseline_image, 100))
# pickle.dump( x, open( "pad_V2_color.pkl", "wb" ) )

# pad = np.load('pad.pkl', allow_pickle=True)
# pad2 = np.load('pad_V2_color.pkl', allow_pickle=True)
# pad_original = np.load('pad_original_color.pkl', allow_pickle=True)

# print((pad_original==pad2).all())



########################################################
# RF LOC - VALIDATION
########################################################

# original_gray = np.load('validation testing/nanoflann_50k_rf_loc.pkl', allow_pickle=True)
# gray = np.load('data/50k/nanoflann nearest_neighbour_method/50k_rf_loc.pkl', allow_pickle=True)

# for index, (first, second) in enumerate(zip(original_gray, gray)): 
#     if abs(first-second) > 0.0005:
#         print(index, first, second)

########################################################
# RF COEFF - VALIDATION
########################################################

# original_gray = np.load('validation testing/nanoflann_50k_rf_coeff_original.pkl', allow_pickle=True)
# gray = np.load('data/50k/nanoflann nearest_neighbour_method/50k_rf_coeff.pkl', allow_pickle=True)

# gray = gray.astype(np.float64)

# gray = gray/100000000

# rf_coeff_original = original_gray[0]
# for i in range(1):
#         kernel = rf_coeff_original[i]
#         for j in range(kernel.shape[0]):
#                 for k in range(kernel.shape[1]):
#                         if kernel[j, k] != gray[i, j, k]:
#                                 print(kernel[j, k], gray[i, j, k] )
