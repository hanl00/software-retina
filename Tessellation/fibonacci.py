import numpy as np
from .ssnn import SSNN
from .utils import *

# Author: George Killick

def fibonacci_sunflower(n_nodes):
	""" Generates points using the golden ratio

		Parameters
		----------
		n_nodes: number of points to be generated

		Return: numpy array of points

	"""

	# Golden ratio (can use any value here for different
	# tessellation mosaics)
	g_ratio = (np.sqrt(5) + 1) / 2

	nodes = np.arange(1,n_nodes+1)

	# calculate rho for each point
	# sqrt to maintain uniform density
	# offset by -0.5 to fill centre gap
	# normalize by sqrt of total nodes to constrain to unit circle
	rho = np.sqrt(nodes-0.5)/np.sqrt(n_nodes)

	# Rotate each point by the golden ratio
	# being the most unrationable number means the points get
	# optimal spacing
	theta = np.pi * 2 * g_ratio * nodes

	# convert to cartesian coordinates
	x = rho * np.cos(theta)
	y = rho * np.sin(theta)

	# return points in standard format
	return np.array([x,y]).T

def fibonacci_retina(n_nodes, fovea, foveal_density):

	""" Generates points using the fibonacci sunflower
		and dilates them with the dilate function found in utils.
		See README for more description of this dilate function.

		Parameters
		----------
		n_nodes: number of nodes in tessellation
		fovea: size of foveal region in tessellation; 0 < fovea <= 1
		fovea_density: scaling factor to affect the ratio of nodes
		in and outside the fovea.

		Return: numpy array of points

	"""
	x = fibonacci_sunflower(n_nodes)

	x = normalize(x)
	x = cart2pol(x)
	x[:,1] *= (1/(fovea + ((2*np.pi*fovea)/foveal_density)) ** x[:,1] ** foveal_density)
	x = pol2cart(x)

	return normalize(x)

def hybrid(n_nodes, fovea, foveal_density, verbose=True):
	
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

	# small randomization and barycentre generation to help 
	# promote pseudo uniform tessellation
	shake = retina
	shake = randomize(shake, 0.23)
	shake = point_gen(shake, mode='sierpinski', concatenate=True)
	shake = randomize(shake, 0.23)

	# annealling the points
	frying = SSNN(100, 0.1)
	frying.set_weights(shake)

	frying.fit(3000, 0.033, 0.0005, verbose)

	weights = frying.weights

	return normalize(weights)
	




