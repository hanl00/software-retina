## Software-retina

### What is a software retina?

A software retina is made up of n-number of nodes that follows the density distribution of photoreceptor cells (rods and cones) found in the human retina. 

<p float="left">
  <img src="https://raw.github.com/hanl00/software-retina/main/data/readme_pics/8k.png" height="300" width="300">
  <img src="https://raw.github.com/hanl00/software-retina/main/data/readme_pics/rods_cones.png">
</p>

### Why use a software retina?

The high density of the cone cells results provides a high visual acuity in the central region of the human view known as the fovea centralis or fovea. This gives human what is known as a foveal vision, a sharp central vision which allows humans to perform activities where visual detail is the utmost importance such as driving, reading and sewing. It is estimated that if the human eye sampled at everything in the field of view at foveal resolution, the human cortex would need to be orders of magnitude larger to accommodate this. This showed that the biological structure of the human retina reduces the amount of information being sent to the brain. This could potentially be applied in image processing where the information around the image edges are compressed while leaving the centre untouched to allow for salient features to be extracted. This pre-processing approach has been validated in previous works such as [[1]](https://doi.org/10.1007/978-3-030-66415-2_32) and [[2]](http://eprints.gla.ac.uk/148797/7/148797.pdf).

### What is included in this package?

This package includes:
* software retina
* software retina generation

The software retina consists of the retina class itself which allow users to sample image/video inputs in grayscale or colour.

The software retina generation allows users to generate software retinas of varying sizes.

### How to install?

Software retina can be installed using:


    pip install software-retina


### References and sources
<a id="1">[1]</a> 
Samagaio A.M., Siebert J.P. (2020) An Investigation of Deep Visual Architectures Based on Preprocess Using the Retinal Transform. In: Bartoli A., Fusiello A. (eds) Computer Vision – ECCV 2020 Workshops. ECCV 2020. Lecture Notes in Computer Science, vol 12535. Springer, Cham. <br />
<a id="http://eprints.gla.ac.uk/148797/7/148797.pdf">[2]</a> 
Ozimek P., Siebert J.P. (2017) Integrating a Non-Uniformly Sampled Software Retina with a Deep CNN Model
* https://github.com/georgeKillick90/Retina
* https://github.com/Pozimek/RetinaVision
