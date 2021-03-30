import cv2
import numpy as np
import skimage.io

from src.software_retina.retina import *

def test_sample_grayscale():
    node_attributes_5k = np.load('data/5k/5k_rf_node_attributes.pkl', allow_pickle=True)
    coefficients_5k = np.load('data/5k/5k_rf_coefficients.pkl', allow_pickle=True)

    R = Retina(node_attributes_5k, coefficients_5k)

    baseline_image = skimage.io.imread('data/dock.jpg')
    baseline_image = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY)

    V = R.sample_grayscale(baseline_image, (360.0, 640.0))

    value = np.load('tests/5k_dock_sample_gray_cython.pkl', allow_pickle=True)

    assert np.all(V == value)


def test_sample_colour():
    node_attributes_5k = np.load('data/5k/5k_rf_node_attributes.pkl', allow_pickle=True)
    coefficients_5k = np.load('data/5k/5k_rf_coefficients.pkl', allow_pickle=True)

    R = Retina(node_attributes_5k, coefficients_5k)

    baseline_image = skimage.io.imread('data/dock.jpg')

    V = R.sample_colour(baseline_image, (360.0, 640.0))

    value = np.load('tests/5k_dock_sample_colour_cython.pkl', allow_pickle=True)

    assert np.all(V == value)