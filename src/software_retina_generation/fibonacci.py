import numpy as np

from .utils import normalize, cartesian_to_polar, polar_to_cartesian

# Original code provided by George Killick


def fibonacci_sunflower(node_count):

    golden_ratio = (np.sqrt(5) + 1) / 2
    nodes = np.arange(1, node_count+1)
    rho = np.sqrt(nodes-0.5)/np.sqrt(node_count)
    theta = np.pi*2*golden_ratio*nodes
    x = rho*np.cos(theta)
    y = rho*np.sin(theta)

    return np.array([x, y]).T


def fibonacci_retina(node_count, foveal_region_size, foveal_density):

    x = fibonacci_sunflower(node_count)
    x = normalize(x)
    x = cartesian_to_polar(x)
    x[:, 1] *= (1/(foveal_region_size + ((2*np.pi*foveal_region_size) /
                foveal_density)) ** x[:, 1] ** foveal_density)
    x = polar_to_cartesian(x)

    return normalize(x)
