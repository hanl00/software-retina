from Cython.Build import cythonize
from setuptools import setup
from distutils.core import setup, Extension
import numpy

ext_modules = [
    Extension(
        "src.software_retina.utils", 
        ["src/software_retina/utils.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "src.software_retina.retina",
        ["src/software_retina/retina.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=['pxd'],
    ),
    Extension(
        "src.software_retina.rf_generation",
        ["src/software_retina/rf_generation.pyx"],
    ),
]

setup(
    name="software_retina",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)

# python setup.py build_ext --inplace