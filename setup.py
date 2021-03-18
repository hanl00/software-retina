from Cython.Build import cythonize
from setuptools import setup
from distutils.core import setup, Extension
import numpy

ext_modules = [
    Extension(
        "utils", 
        ["software_retina/utils.pyx"], 
    ),
    Extension(
        "retina",
        ["software_retina/retina.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "rf_generation",
        ["software_retina/rf_generation.pyx"],
    ),
]

setup(
    name="software_retina",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)

# python setup.py build_ext --inplace