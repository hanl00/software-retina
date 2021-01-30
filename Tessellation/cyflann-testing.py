import cyflann
from ssnn import *

retina = SSNN(n_nodes = 2048, fovea=0.1)
retina.fit()