## Here is the following structure for the repository

* Tessellation - Killick's retina generator
* bin - RetinaCUDA library developed by Balog
* examples - examples provided in Ozimek's code
* retinavision - Ozimek's code
* retinavision_cython - Cythonised version of Ozimek's code, mainly on the retina sampling algorithm (retinavision_cython/retina/retina_sample.pyx)
* data.csv - a 1920 x 1080 array of uint8 datatype for testing purposes (baseline_image)
* setup.py - required for compiling .pyx files, command can be found below
* test_retina_cython.py - testing/benchmarking takes places here

### Notes

* generated files (.pyd and .c) from compiling .pyx files have been added into gitignore
* .html and .cpp files have been added to gitignore and older versions in the repository are no longer tracked
* .pkl files have been removed from gitignore
* repository visibility is now public

### Compiling .pyx code

* Cython version 0.29.21
* navigate to base directory, $python setup.py build_ext --inplace


* https://github.com/georgeKillick90/Retina
* https://github.com/Pozimek/RetinaVision