import time
import sys
import multiprocessing

import numpy as np
import pynanoflann
from scipy.spatial.distance import cdist

from .utils import normalize, cartesian_to_polar, polar_to_cartesian

# Original code provided by George Killick

total_threads = multiprocessing.cpu_count()


class SelfSimilarNeuralNetwork:

    def __init__(self, node_count, foveal_region_size,
                 nearest_neighbour_method='auto'):

        self.node_count = node_count
        self.foveal_region_size = foveal_region_size
        self.weights = SelfSimilarNeuralNetwork.__init_weights(node_count)
        self.nearest_neighbour_method = nearest_neighbour_method
        self.nanoflann = pynanoflann.KDTree(n_neighbors=1, metric='L2',
                                            radius=1)

    def fit(self, num_iters=20000, initial_learning_rate=0.1,
            final_learning_rate=0.0005, verbose=True):

        learning_rate = SelfSimilarNeuralNetwork.__alpha_schedule(
            initial_learning_rate, final_learning_rate,
            num_iters, num_iters//4)
        start = time.time()
        get_neighbours = self.__select_nearest_neighbour_method(
            self.nearest_neighbour_method)

        for i in range(num_iters):
            alpha = learning_rate[i]

            input_vectors = np.copy(self.weights)
            input_vectors = cartesian_to_polar(input_vectors)

            d = np.exp((2*np.random.uniform() - 1)*np.log(8))

            input_vectors[:, 1] *= d
            input_vectors = polar_to_cartesian(input_vectors)

            delta_theta = 2*np.random.uniform()*np.pi
            delta_rho = np.random.uniform() * self.foveal_region_size

            input_vectors[:, 0] += np.cos(delta_theta)*delta_rho
            input_vectors[:, 1] += np.sin(delta_theta)*delta_rho
            input_vectors = cartesian_to_polar(input_vectors)
            input_vectors[:, 0] += 2*np.random.uniform()*np.pi

            cull = np.where(input_vectors[:, 1] <= 1)[0]

            input_vectors = polar_to_cartesian(input_vectors)
            input_vectors = input_vectors[cull]

            index = get_neighbours(input_vectors, self.weights)
            self.weights[index] -= ((self.weights[index] - input_vectors)
                                    * alpha)
            if (verbose):
                sys.stdout.write('\r' + str(i + 1) + "/" + str(num_iters))

        normalize(self.weights)
        if(verbose):
            print("\nFinished.")
            print("Time taken: " + str(time.time()-start))

    def set_weights(self, X):

        self.node_count = X.shape[0]
        self.weights = X

    def __select_nearest_neighbour_method(self, nearest_neighbour_method):

        if (nearest_neighbour_method == 'brute_force'):
            print("Using bruteforce.")
            return self.__brute_force_neighbours

        elif(nearest_neighbour_method == 'nanoflann'):
            print("Using nanoflann method.")
            return self.__pynanoflann_neighbours

        elif(nearest_neighbour_method == 'nanoflann_multi_jobs'):
            print("Using nanoflann method with " + str(total_threads) + " threads.")
            return self.__pynanoflann_multi_neighbours

        elif(nearest_neighbour_method == 'auto'):
            if(self.node_count <= 256):
                print("Using bruteforce.")
                return self.__brute_force_neighbours

            elif(self.node_count <= 14999):
                print("Using nanoflann method.")
                return self.__pynanoflann_neighbours

            else:
                print("Using nanoflann method with " + str(total_threads) + " threads.")
                return self.__pynanoflann_multi_neighbours

        else:
            print("Unknown nearest_neighbour_method, using nanoflann.")

    def __brute_force_neighbours(self, update_vectors, network_weight):

        dists = cdist(update_vectors, network_weight)
        indices = np.argmin(dists, axis=1)

        return indices

    def __pynanoflann_neighbours(self, update_vectors, network_weight):

        self.nanoflann.fit(network_weight)
        distances, indices = self.nanoflann.kneighbors(update_vectors)

        return indices.flatten()

    def __pynanoflann_multi_neighbours(self, update_vectors, network_weight):

        self.nanoflann.fit(network_weight)
        distances, indices = self.nanoflann.kneighbors(update_vectors,
                                                       n_jobs=total_threads)

        return indices.flatten()

    @staticmethod
    def __init_weights(node_count):

        r = np.random.uniform(1, 0, node_count)
        th = 2*np.pi*np.random.uniform(1, 0, node_count)

        return polar_to_cartesian(np.array([th, r]).T)

    @staticmethod
    def __alpha_schedule(initial_learning_rate, final_learning_rate,
                         num_iters, split):

        """ Creates a learning rate schedule for the SelfSimilarNeuralNetwork.
            Constant learning rate for first "split" of iterations
            and then linearly annealing for the remainder.

            Parameters
            ----------
            initial_learning_rate: starting learning rate.
            final_learning_rate: final learning rate.
            num_iters: number of iterations, equivalent to number
            of iterations to train the network for.
            split: decides when to start annealing, typical after 25%
            oftotal iterations.

            Return: Numpy array for learning rate; length num_iters.

        """

        static = split
        decay = num_iters - static
        static_lr = np.linspace(initial_learning_rate,
                                initial_learning_rate,
                                static)
        decay_lr = np.linspace(initial_learning_rate,
                               final_learning_rate,
                               decay)

        return np.concatenate((static_lr, decay_lr))
