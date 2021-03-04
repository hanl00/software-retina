from Cython.Build import cythonize
from setuptools import setup
from distutils.core import setup, Extension
import numpy

ext_modules = [
    Extension(
        "retina_sample",
        ["retinavision_cython/retina/retina_sample.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "retina_utils", 
        ["retinavision_cython/retina/retina_utils.pyx"], 
    ),
    # Extension(
    #     "tessellation.ssnn_cyflann", 
    #     ["tessellation/ssnn_cyflann.pyx"], 
    # )
]

setup(
    name="cheese",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)


# cython compile 
# python setup.py build_ext --inplace