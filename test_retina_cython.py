import timeit
import numpy as np
from os.path import dirname, join
import sys
import cProfile, pstats, io
import time

sys.path.append('C:\\Users\\Nicholas\\Documents\\University @ Glasgow\\Year 5\\cythonised_retina\\retinavision')
import retinavision

sys.path.append('C:\\Users\\Nicholas\\Documents\\University @ Glasgow\\Year 5\\cythonised_retina\\retinavision_cython\\memory view')
from memoryView import sum3d

sys.path.append('C:\\Users\\Nicholas\\Documents\\University @ Glasgow\\Year 5\\cythonised_retina\\retinavision_cython\\retina')
import retina_sample
import retina_utils


datadir = join(dirname(dirname(__file__)), "cythonised_retina")

# define data, save to csv file (for comparision with previous cythonised improvement)
# baseline_image = np.random.randint(256, size=(1080, 1920), dtype=np.uint8)
# np.savetxt('data.csv', baseline_image, delimiter=',')
baseline_image = np.loadtxt('data.csv', delimiter=',')


# generate a random 2 dimensional array (for comparision with original code)
input_img =  np.random.randint(256, size=(1080, 1920), dtype=np.uint8)

# generate 10 different 2d arrays for multiple frame testing (for comparision with original code)
test_list = []
for x in range(10):
    input_img =  np.random.randint(256, size=(1080, 1920), dtype=np.uint8)
    test_list.append(input_img)

#old fixation was 640, 360
fixation = (360.0, 640.0)

#campicshape has to match input img shape for original retina
#campicShape = (720, 1280, 3) #old
campicShape = (1080, 1920, 3)


def runOriginalPad():
    padding = 1
    return retinavision.utils.pad(input_img, padding, True)

def runCythonPad():
    image = input_img.astype(np.float64)
    padding = 1
    return retina_sample.pad(image, padding, True)

def runCPadMemView():
    padding = 1
    return padMemoryView(input_img, padding, False)

# for comparing output, change baseline/ random input here
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
    img = baseline_image.astype(np.float64)
    v = R.sample(img, fixation)
    return v

def testCython():
    py = timeit.timeit('''runOriginal()''',setup="from __main__ import runOriginal",number=100)
    cy = timeit.timeit('''runCPad()''',setup="from __main__ import runCPad", number=100)

    print(cy, py)
    print('Cython is {}x faster'.format(py/cy))

def testMemoryView():
    py = timeit.timeit('''runCPad()''',setup="from __main__ import runCPad",number=100)
    cy = timeit.timeit('''runCPadMemView()''',setup="from __main__ import runCPadMemView", number=100)

    print(cy, py)
    print('Memory View is {}x faster'.format(py/cy))

def compareRetinaPad():
    py = timeit.timeit('''runOriginalPad()''',setup="from __main__ import runOriginalPad",number=10)
    cy = timeit.timeit('''runCythonPad()''',setup="from __main__ import runCythonPad", number=10)

    print(cy, py)
    print('Cython is {}x faster'.format(py/cy))

def compareRetinaSampleOld():  #old one takes account of Retina initialisation and loading
    py = timeit.timeit('''originalRetinaSample()''',setup="from __main__ import originalRetinaSample",number=10)
    cy = timeit.timeit('''cythonRetinaSample()''',setup="from __main__ import cythonRetinaSample", number=10)

    print(cy, py)
    print('Cython is {}x faster'.format(py/cy))

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
    img = baseline_image.astype(np.float64)
    cython_start_time = time.time()
    CR.sample(img, fixation)
    cython_sample_time = time.time() - cython_start_time

    return original_sample_time, cython_sample_time

# sample 10 images each test
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
    img = input_img.astype(np.float64)
    cython_time = 0

    for input_img in test_list:
        cython_start_time = time.time()
        CR.sample(img, fixation)
        cython_sample_time = time.time() - cython_start_time
        cython_time += cython_sample_time

    return original_time, cython_time

# run sample 1 test 10 loop
def compareWithLoops(x, testFunction):
    original_total, cython_total = 0, 0
    for count in range(x):
        original_sample_time, cython_sample_time = testFunction()
        original_total += original_sample_time
        cython_total += cython_sample_time

    print("Function tested: " + str(testFunction))
    print(original_total, cython_total)
    print('Cython is {}x faster'.format(original_total/cython_total))

def timeCythonRetinaSample(x):
    total_time = 0
    CR = retina_sample.Retina()
    retina_sample.loadCoeff()
    retina_sample.loadLoc()
    CR.updateLoc()
    img = baseline_image.astype(np.float64)

    for i in range(x):
        cython_start_time = time.time()
        CR.sample(img, fixation)
        cython_sample_time = time.time() - cython_start_time
        total_time += cython_sample_time

    print("Average time taken to sample baseline image {number} times: {value}".format(number = x, value = (total_time/x)))

# profiling original code
def originalRetinaProfile():
    R = retinavision.Retina()
    R.loadLoc(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_loc.pkl"))
    R.loadCoeff(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_coeff.pkl"))
    R.prepare(campicShape, fixation)
    cProfile.runctx("R.sample(input_img, fixation)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("cumtime").print_stats()

# profiling current code
def cythonRetinaProfile():
    R = retina_sample.Retina()
    retina_sample.loadCoeff()
    retina_sample.loadLoc()
    R.updateLoc()
    img = baseline_image.astype(np.float64)
    cProfile.runctx("R.sample(img, fixation)", globals(), locals(), "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("cumtime").print_stats()

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

# testOutput()
# originalRetinaSample()
compareWithLoops(10, compareRetinaSample)
# compareWithLoops(10, compareRetinaSample10)
# originalRetinaProfile()
# cythonRetinaProfile()
# timeCythonRetinaSample(100)