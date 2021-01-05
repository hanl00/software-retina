from .fibonacci import *
from .utils import *
from .ssnn import *

def fib_ssnn_hybrid(n_nodes, fovea, foveal_density, verbose=True):
	
	""" A hybrid approach to generating retina tessellations;
		uses the fibnacci retina with a sierpinski node generation
		applied as an initialization for the SSNN.

		Parameters
		----------
		n_nodes:
		fovea:
		foveal_density:

		Return: retina tessallation.
	"""

	retina = fibonacci_retina(n_nodes//4, fovea, foveal_density)

	shake = retina
	shake = randomize(shake, 0.23)
	shake = point_gen(shake, mode='sierpinski', concatenate=True)
	shake = randomize(shake, 0.23)

	frying = SSNN(100, 0.1)
	frying.set_weights(shake)

	frying.fit(3000, 0.033, 0.0005, verbose)

	weights = frying.weights

	return normalize(weights)