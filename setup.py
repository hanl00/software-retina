from Cython.Build import cythonize
from setuptools import setup
from distutils.core import setup, Extension
import numpy

ext_modules = [
    Extension(
        "utils", 
        ["retina/utils.pyx"], 
    ),
    Extension(
        "sample",
        ["retina/sample.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "rf_generation",
        ["retina/rf_generation.pyx"],
    ),
]

setup(
    name="fast_retina",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)

# python setup.py build_ext --inplace