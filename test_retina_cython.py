import timeit
import numpy as np
from os.path import dirname, join
import sys
import cProfile, pstats, io
import time
from retinavision import *
from retinavision_cython.retina import *

datadir = join(dirname(dirname(__file__)), "cythonised_retina")

# uint8 input
baseline_image = np.loadtxt('data.csv', delimiter=',')

# generate a random 2 dimensional array 
input_img =  np.random.randint(256, size=(1080, 1920), dtype=np.uint8)

# generate 10 different 2d arrays for multiple frame testing
test_list = []
for x in range(10):
    input_img =  np.random.randint(256, size=(1080, 1920), dtype=np.uint8)
    test_list.append(input_img)

#old fixation was 640, 360
fixation = (360.0, 640.0)

# campicshape has to match input img shape for original retina
# campicShape = (720, 1280, 3) #old
campicShape = (1080, 1920, 3)


def runOriginalPad():
    padding = 1
    return retinavision.utils.pad(input_img, padding, True)

def runCythonPad():
    image = input_img.astype(np.int32)
    padding = 1
    return retina_sample.pad(image, padding, True)

def originalRetinaSample():
    R = retinavision.Retina()
    R.loadLoc(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_loc.pkl"))
    R.loadCoeff(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_coeff.pkl"))
    R.prepare(campicShape, fixation)
    v = R.sample(baseline_image, fixation)
    return v

def cythonRetinaSample():
    R = retina_sample.Retina()
    retina_sample.loadCoeff()
    retina_sample.loadLoc()
    R.updateLoc()
    img = baseline_image.astype(np.int32)
    v = R.sample(img, fixation)
    return v


###################################
# SAMPLING COMPARISON

# sample 1 image each test
def compareRetinaSample():
    OR = retinavision.Retina()
    OR.loadLoc(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_loc.pkl"))
    OR.loadCoeff(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_coeff.pkl"))
    OR.prepare(campicShape, fixation)   
    original_start_time = time.time()
    OR.sample(baseline_image, fixation)
    original_sample_time = time.time() - original_start_time
    
    CR = retina_sample.Retina()
    retina_sample.loadCoeff()
    retina_sample.loadLoc()
    CR.updateLoc()
    img = baseline_image.astype(np.int32)
    cython_start_time = time.time()
    CR.sample(img, fixation)
    cython_sample_time = time.time() - cython_start_time

    return original_sample_time, cython_sample_time

# sample a list of 10 images each test
def compareRetinaSample10():
    OR = retinavision.Retina()
    OR.loadLoc(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_loc.pkl"))
    OR.loadCoeff(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_coeff.pkl"))
    OR.prepare(campicShape, fixation)   
    original_time = 0

    for input_img in test_list:
        original_start_time = time.time()
        OR.sample(input_img, fixation)
        original_sample_time = time.time() - original_start_time
        original_time += original_sample_time
    
    CR = retina_sample.Retina()
    retina_sample.loadCoeff()
    retina_sample.loadLoc()
    CR.updateLoc()
    cython_time = 0

    for input_img in test_list:
        img = input_img.astype(np.int32)
        cython_start_time = time.time()
        CR.sample(img, fixation)
        cython_sample_time = time.time() - cython_start_time
        cython_time += cython_sample_time

    return original_time, cython_time

# run each sample test above X amount of times
def compareWithLoops(x, testFunction):
    original_total, cython_total = 0, 0
    for count in range(x):
        original_sample_time, cython_sample_time = testFunction()
        original_total += original_sample_time
        cython_total += cython_sample_time

    print("Function tested: " + str(testFunction))
    print("Total time taken for original code : {original_total}".format(original_total =  original_total))
    print("Total time taken for cython code : {cython_total}".format(cython_total = cython_total))
    print("Average time taken for original to sample {number} times: {value}".format(number = x, value = (original_total/x)))
    print("Average time taken for cython to sample {number} times: {value}".format(number = x, value = (cython_total/x)))
    print('Cython is {}x faster'.format(original_total/cython_total))


###################################
# INDIVIDUAL TIMING

def timeCythonRetinaSample(x):
    total_time = 0
    CR = retina_sample.Retina()
    retina_sample.loadCoeff()
    retina_sample.loadLoc()
    CR.updateLoc()
    img = baseline_image.astype(np.int32)

    for i in range(x):
        cython_start_time = time.time()
        CR.sample(img, fixation)
        cython_sample_time = time.time() - cython_start_time
        total_time += cython_sample_time

    print("Average time taken for to sample baseline image {number} times: {value}".format(number = x, value = (total_time/x)))

def timeOriginalRetinaSample(x):
    total_time = 0
    OR = retinavision.Retina()
    OR.loadLoc(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_loc.pkl"))
    OR.loadCoeff(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_coeff.pkl"))
    OR.prepare(campicShape, fixation)   

    for i in range(x):
        original_start_time = time.time()
        OR.sample(baseline_image, fixation)
        original_sample_time = time.time() - original_start_time
        total_time += original_sample_time

    print("Average time taken to sample baseline image {number} times: {value}".format(number = x, value = (total_time/x)))


###################################
# PROFILING

# profiling original code
def originalRetinaProfile():
    R = retinavision.Retina()
    R.loadLoc(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_loc.pkl"))
    R.loadCoeff(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_coeff.pkl"))
    R.prepare(campicShape, fixation)
    cProfile.runctx("R.sample(baseline_image, fixation)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("cumtime").print_stats()

# profiling current code
def cythonRetinaProfile():
    R = retina_sample.Retina()
    retina_sample.loadCoeff()
    retina_sample.loadLoc()
    R.updateLoc()
    img = baseline_image.astype(np.int32)
    cProfile.runctx("R.sample(img, fixation)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("cumtime").print_stats()


###################################
# TESTING ACCURACY

# check output
def testOutput():
    print("testing output")
    original = originalRetinaSample()
    cythonised = cythonRetinaSample()

    for index, (first, second) in enumerate(zip(original, cythonised)):
        if abs(first-second) > 0.05:
            print(index, first, second)
        if index == 49999:
            print(index, first, second)

# print(originalRetinaSample())
compareWithLoops(2, compareRetinaSample)
# compareWithLoops(10, compareRetinaSample10)
# testOutput()
# originalRetinaProfile()
# cythonRetinaProfile()
# timeCythonRetinaSample(100)
# timeOriginalRetinaSample(10)
