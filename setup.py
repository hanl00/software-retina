from Cython.Build import cythonize
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
    )
]

setup(
    name="cheese",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
