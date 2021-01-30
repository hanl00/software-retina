
import time
import numpy as np
# from pyflann import *
from cyflann import *
from scipy.spatial.distance import cdist

from utils import *

# Authors: George Killick 


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
        self.flann = FLANNIndex()
                      
    
    def fit(self, num_iters=20000, initial_alpha=0.1, final_alpha=0.0005, verbose=True):

        """ Uses the Self-Similar-Neural-Network algorithm to organize n number of nodes
            into a retina style tessellation. See README for more information.

            Parameters
            ----------
            num_iters: Number of iterations to run the algorithm for
            initial_alpha: initial learning rate
            final_alpha: final learning rate at end of training.
            verbose: displays helpful stats during the fit process

            Return: None.
        """
        
        learning_rate = SSNN.__alpha_schedule(initial_alpha, final_alpha, num_iters, num_iters//4)

        start = time.time()

        get_neighbours = self.__select_method(self.method)
        
        for i in range(num_iters):
            
            alpha = learning_rate[i]
            
            I = np.copy(self.weights)
                     
            I = cart2pol(I)
   
            # Dilation
            d = np.exp((2 * np.random.uniform() -1) * np.log(8))
            I[:,1] *= d
            
            I = pol2cart(I)
            
            # Translation 
            delta_theta = 2 * np.random.uniform() * np.pi
            delta_rho = np.random.uniform() * self.fovea
            
            # Changes in the x and y direction
            I[:,0] += np.cos(delta_theta) * delta_rho
            I[:,1] += np.sin(delta_theta) * delta_rho
            
            # Random rotation
            I = cart2pol(I)
            I[:,0] += 2 * np.random.uniform() * np.pi
            
            # Remove the input vectors which are outside the bounds of the retina
            cull = np.where(I[:,1] <= 1)[0]
            I = pol2cart(I)
            I = I[cull]
            
            # Gets the nearest update vector for each weight
            index = get_neighbours(I, self.weights)
            
            # Update the weights 
            self.weights[index] -= (self.weights[index] - I) * alpha
 
            # Display progress
            if (verbose):
                sys.stdout.write('\r'+str(i + 1) +"/" + str(num_iters))
        
        # Constrain to unit circle
        normalize(self.weights)

        if(verbose):
            print("\nFinished.")
            print("Time taken: " + str(time.time()-start))

    def generative_fit(self, steps, num_iters=20000, initial_alpha=0.1, final_alpha=0.0005, verbose=True):

        """ Uses a point generation method based on delaunay triangulation
            to decrease training times. See README for more info.

            Parameters
            ----------
            steps: Number of iterations for generating points.

            -- remaining parameters same as normal fit function.

            Return: None

        """
        
        start = time.time()

        # Calculates the required starting node size for number of
        # generative iterations. 
        # Node count x 4 for Sierpinski. (approx)
        # Node count x 3 for Barycentre. (approx)

        self.n_nodes = self.n_nodes // (4 ** steps)
        self.weights = SSNN.__init_weights(self.n_nodes)

        SSNN.fit(self, num_iters, initial_alpha, final_alpha, verbose)
        
        # Number of training iterations after point generation
        g_iters = 3000

        # Calculates a new initial alpha
        g_alpha = ((g_iters/(num_iters * 0.75)) * initial_alpha) + final_alpha

        for i in range(steps):

            print("\nAnnealing new points...")
            self.weights = randomize(self.weights)
            self.weights = point_gen(self.weights, 'sierpinski') #point_gen method from utils.py
            self.weights = randomize(self.weights)
            self.n_nodes = self.weights.shape[0]

            if(i == steps-1):
                # Slightly increase number of iterations on final run,
                # almost like polishing everything up.
                g_iters += 2000

            self.fit(g_iters,0.033, 0.0005, verbose)
        
        if(verbose):
            print("\nFinal node count: " + str(self.weights.shape[0]))
            print("\nTotal time taken: " + str(time.time()-start))
        
        return 

    def set_weights(self, X):
        self.n_nodes = X.shape[0]
        self.weights = X


    def __select_method(self, method):

        """ Selects the nearest neighbour backend to use.

            Parameters
            ----------

            method:
                - 'default': Uses a Scipy brute force search
                - 'flann': Uses FLANN
                - 'auto': Selects best backend based on number
                of nodes and available hardware.

            Returns: Function to compute nearest neighbours

        """

        if (method == 'default'):
            print("Using Scipy.")
            return self.__bf_neighbours

        elif(method == 'flann'):
            print("Using FLANN.")
            return self.__flann_neighbours

        elif(method == 'auto'):
            if(self.n_nodes <= 256):
                print("Using Scipy.")
                return self.__bf_neighbours

            else:
                print("Using FLANN.")
                return self.__flann_neighbours
        else:
        	print("Unknown method, using FLANN backend.")

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

    def __flann_neighbours(self, X, Y):

        """ Canonical wrapper for FLANN nearest neighbour 
            search.
            
            Parameters
            ----------
            X: update vectors
            Y: network weights

            Return: index of nearest neighbour for each
            update vector.
        """
        indeces, dists = self.flann.nn(Y, X, 1, algorithm="kdtree", branching=16, iterations=5, checks=16) 
        return indeces

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
        th = 2 * np.pi * np.random.uniform(1, 0, n_nodes)
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




