from src.software_retina.retina import *
from src.software_retina.rf_generation import *
import skimage.io
import cv2

# Load a pre-generated retina tessellation from the data folder/ alternatively you can generate one using "software_retina_generation"
example_5k_tessellation = np.load('data/5k/nanoflann_5k_tessellation.pkl', allow_pickle=True)

# Generate node attributes(rf_node_attributes) and kernel coefficients (rf_coefficient)
node_attributes, coefficients = rf_generation(example_5k_tessellation, kernel_ratio = 3, sigma_base = 0.5, sigma_power = 1, min_rf = 1)

# Retina initialisation and sampling
R = Retina()
R.load_node_attributes(node_attributes) 
R.load_coefficients(coefficients)

baseline_image = skimage.io.imread('data/dock.jpg')

baseline_image = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY) 

if baseline_image.ndim == 2:
    print("Sampling grayscale")
    V = R.sample_grayscale(baseline_image, (360.0, 640.0))
    
else: 
    print("Sampling colored")
    V = R.sample_coloured(baseline_image, (360.0, 640.0))

