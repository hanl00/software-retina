# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

from scipy.spatial import distance

from src.software_retina.utils cimport gauss
from src.software_retina.utils cimport gausskernel

# Original code provided by Piotr Ozimek


def node_attribute_kernel_generation(tessellation, kernel_ratio, sigma_base,
                                     sigma_power, min_rf, min_kernel=3):

    """ Generates node attributes and kernels for each node
        Parameters
        ----------
        tessellation: raw node locations [x,y] array
        kernel_ratio: the ratio of kernel to local node density (dist_5)
        sigma_base: the base sigma, or global sigma scaling factor (lambda)
        sigma_power: the power term applied to sigma scaling with eccentricity
        min_rf: sets the min distance between the 20 most central nodes
        min_kernel: imposes a minimum kernel width for the receptive fields
                    (default value: 3)

        Return
        rf_node_attributes - a num_of_nodes x 7 array that describes
                             each node as follows:
        [x, y, d, angle (radians), dist_5, rf_sigma, rf_width]
        rf_coefficients - an padded array of gaussian receptive field kernels

    """

    rf_node_attributes = np.zeros((len(tessellation), 7))
    rf_coefficients_unpadded = np.ndarray((1, len(tessellation)),
                                          dtype='object')

    row_length = 0
    column_length = 0
    output_list = []
    neighbourhood = 6

    print("rf generation - might take a while...")

    length = len(tessellation)
    chunk = 5000
    num = length/chunk
    if length % chunk != 0:
        num += 1

    dist_5 = np.zeros(length, dtype='float64')
    print(str(chunk) + " nodes in one chunk.")

    for i in range(int(num)):
        print("Processing chunk " + str(i))
        d = distance.cdist(tessellation[i*chunk:(i + 1)*chunk],
                           tessellation)
        s = np.sort(d)
        dist_5[i*chunk:(i + 1)*chunk] = np.mean(s[:, 1:neighbourhood], 1)

    fov_dist_5 = np.min(dist_5[:20])
    rf_node_attributes[:, :2] = tessellation*(1/fov_dist_5)*min_rf
    dist_5 = dist_5*(1/fov_dist_5)*min_rf
    rf_node_attributes[:, 3] = np.arctan2(rf_node_attributes[:, 1],
                                          rf_node_attributes[:, 0])
    rf_node_attributes[:, 4] = dist_5

    print("All chunks done.")

    correction = 0
    rf_node_attributes[:, 5] = sigma_base*(dist_5+correction)**sigma_power

    for i in range(len(tessellation)):
        kernel_width = max(min_kernel, int(np.ceil(kernel_ratio *
                                           rf_node_attributes[i, 4])))
        rf_node_attributes[i, 6] = kernel_width
        cx, cy = return_offsets(rf_node_attributes[i, 0],
                                rf_node_attributes[i, 1],
                                kernel_width)

        rx = rf_node_attributes[i][0] - cx
        ry = rf_node_attributes[i][1] - cy
        loc = np.array([rx, ry])
        rf_node_attributes[i, 2] = np.linalg.norm(rf_node_attributes[i, :2])

        rf_node_attributes[i, 0] = cx
        rf_node_attributes[i, 1] = cy
        rf_coefficients_unpadded[0, i] = gausskernel(kernel_width, loc,
                                                     rf_node_attributes[i, 5])
        rf_coefficients_unpadded[0, i] /= np.sum(
                                            rf_coefficients_unpadded[0, i])

        if rf_coefficients_unpadded[0, i].shape > (row_length, column_length):
            row_length, column_length = rf_coefficients_unpadded[0, i].shape

    print("Padding kernels now")

    rf_coefficients = np.zeros((len(tessellation), row_length, row_length))

    for i in range(len(tessellation)):
        x, y = rf_coefficients_unpadded[0][i].shape
        rf_coefficients[i, :x, :y] = rf_coefficients_unpadded[0][i]

    rf_coefficients = (rf_coefficients*100000000).astype(np.int32)

    return rf_node_attributes, rf_coefficients


def return_offsets(x, y, kernel_width):
    kernel_width = int(kernel_width)

    if kernel_width % 2 != 0:
        cx = round(x)
        cy = round(y)

    else:
        cx = round(x) + np.sign(x-round(x))*0.5
        cy = round(y) + np.sign(y-round(y))*0.5

    return cx, cy
