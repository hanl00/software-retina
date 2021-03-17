
import time
import numpy as np
from scipy.spatial.distance import cdist
import pynanoflann
import sys
from .utils import normalize, cart2pol, pol2cart

# Original code provided by George Killick


class SSNN:

    def __init__(self, n_nodes, fovea, method='auto'):

        """ Self Similar Neural Network for generation retina
            tessellations.

            Parameters
            ----------
            n_nodes: number of nodes desired in the tessellation
            fovea: size of the foveal region of the retina tessellation.
            method: which backend to use when computing nearest
            neighbours.

        """

        self.n_nodes = n_nodes
        self.fovea = fovea
        self.weights = SSNN.__init_weights(n_nodes)
        self.method = method
        self.nanoflann = pynanoflann.KDTree(n_neighbors=1, metric='L2',
                                            radius=1)

    def fit(self, num_iters=20000, initial_alpha=0.1, final_alpha=0.0005,
            verbose=True):

        """ Uses the Self-Similar-Neural-Network algorithm to organize n
            number of nodes into a retina style tessellation. See README
            for more information.

            Parameters
            ----------
            num_iters: Number of iterations to run the algorithm for
            initial_alpha: initial learning rate
            final_alpha: final learning rate at end of training.
            verbose: displays helpful stats during the fit process

            Return: None.
        """

        learning_rate = SSNN.__alpha_schedule(initial_alpha, final_alpha,
                                              num_iters, num_iters//4)
        start = time.time()
        get_neighbours = self.__select_method(self.method)

        for i in range(num_iters):
            alpha = learning_rate[i]

            input_vectors = np.copy(self.weights)
            input_vectors = cart2pol(input_vectors)

            d = np.exp((2*np.random.uniform() - 1)*np.log(8))

            input_vectors[:, 1] *= d
            input_vectors = pol2cart(input_vectors)

            delta_theta = 2*np.random.uniform()*np.pi
            delta_rho = np.random.uniform() * self.fovea

            input_vectors[:, 0] += np.cos(delta_theta)*delta_rho
            input_vectors[:, 1] += np.sin(delta_theta)*delta_rho
            input_vectors = cart2pol(input_vectors)
            input_vectors[:, 0] += 2*np.random.uniform()*np.pi

            cull = np.where(input_vectors[:, 1] <= 1)[0]

            input_vectors = pol2cart(input_vectors)
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
        self.n_nodes = X.shape[0]
        self.weights = X

    def __select_method(self, method):

        """ Selects the nearest neighbour backend to use.

            Parameters
            ----------

            method:
                - 'default': Uses a Scipy brute force search
                - 'nanoflann': Uses nanoflann
                - 'auto': Selects best backend based on number
                of nodes and available hardware.

            Returns: Function to compute nearest neighbours

        """

        if (method == 'default'):
            print("Using bruteforce.")
            return self.__bf_neighbours

        elif(method == 'nanoflann'):
            print("Using nanoflann.")
            return self.__pynanoflann_neighbours

        elif(method == 'auto'):
            if(self.n_nodes <= 256):
                print("Using bruteforce.")
                return self.__bf_neighbours

            else:
                print("Using nanoflann.")
                return self.__pynanoflann_neighbours

        else:
            print("Unknown method, using nanoflann backend.")

    def __bf_neighbours(self, X, Y):

        """ Canonical wrapper for scipy's bruteforce nearest
            neighbour search.

            Parameters
            ----------
            X: update vectors
            Y: network weights

            Return: index of nearest neighbour for each
            update vector.
        """

        dists = cdist(X, Y)
        indeces = np.argmin(dists, axis=1)

        return indeces

    def __pynanoflann_neighbours(self, X, Y):

        """ Canonical wrapper for nanoflann nearest neighbour
            search.

            Parameters
            ----------
            X: update vectors
            Y: network weights

            Return: index of nearest neighbour for each
            update vector.
        """
        self.nanoflann.fit(Y)
        distances, indices = self.nanoflann.kneighbors(X)

        return indices.flatten()

    @staticmethod
    def __init_weights(n_nodes):

        """ Initializes weights for the SSNN

            Parameters
            ----------
            n_nodes: number of nodes in tessellation i.e.
            number of nodes in SSNN.

            Return: Initialized weights as a numpy array.

        """

        r = np.random.uniform(1, 0, n_nodes)
        th = 2*np.pi*np.random.uniform(1, 0, n_nodes)

        return pol2cart(np.array([th, r]).T)

    @staticmethod
    def __alpha_schedule(initial_alpha, final_alpha, num_iters, split):

        """ Creates a learning rate schedule for the SSNN.
            Constant learning rate for first "split" of iterations
            and then linearly annealing for the remainder.

            Parameters
            ----------
            initial_alpha: starting learning rate.
            final_alpha: final learning rate.
            num_iters: number of iterations, equivalent to number
            of iterations to train the network for.
            split: decides when to start annealing, typical after 25%
            oftotal iterations.

            Return: Numpy array for learning rate; length num_iters.

        """

        static = split
        decay = num_iters - static
        static_lr = np.linspace(initial_alpha, initial_alpha, static)
        decay_lr = np.linspace(initial_alpha, final_alpha, decay)

        return np.concatenate((static_lr, decay_lr))
