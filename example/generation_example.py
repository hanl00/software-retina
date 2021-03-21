from ssnn import SSNN
from fibonacci import fibonacci_retina

# Generating a software retina tessellation with 10000 nodes using nanoflann library
nanoflann_retina = SSNN(n_nodes = 10000, fovea = 0.1, method = "nanoflann")
nanoflann_retina.fit()
nanoflann_tessellation = nanoflann_retina.weights

# Generating a software retina tessellation with 10000 nodes using points from the fibonacci sunflower
fibonacci_10k_tessellation = fibonacci_retina(10000, 0.1, 5)
