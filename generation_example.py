from src.software_retina_generation.ssnn import SelfSimilarNeuralNetwork
from src.software_retina_generation.fibonacci import fibonacci_retina


# Generating a software retina tessellation with 10000 nodes using nanoflann library
nanoflann_retina = SelfSimilarNeuralNetwork(node_count = 20000, foveal_region_size = 0.1, nearest_neighbour_method = "nanoflann")
nanoflann_retina.fit()
nanoflann_tessellation = nanoflann_retina.weights

# Generating a software retina tessellation with 10000 nodes using points from the fibonacci sunflower
fibonacci_10k_tessellation = fibonacci_retina(10000, 0.1, 5)
