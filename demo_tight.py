#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:51:21 2019

Tight retina image demo trial
@author: Piotr Ozimek
"""
import cv2
import numpy as np
import sys
import retinavision
from os.path import dirname, join

from retinavision.cortex import Cortex
from retinavision import datadir, utils
from os.path import join

#Open webcam
cap = utils.camopen() #cap is the capture object (global)
ret, campic = cap.read()
datadir = join(dirname(dirname(__file__)), "cythonised_retina")

#Create and load retina
R = retinavision.Retina()
R.info()
R.loadLoc(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_loc.pkl"))
R.loadCoeff(join(datadir, "retinavision_cython", "data", "retinas", "ret50k_coeff.pkl"))

#Prepare retina
x = campic.shape[1]/2
y = campic.shape[0]/2
fixation = (y,x)
print('campic.shape')
print(campic.shape)
print('fixation')
print(fixation)
R.prepare(campic.shape, fixation)

while True:
    ret, img = cap.read()
    if ret is True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print('img.shape')
        print(img.shape)
        print('fixation')
        print(fixation)
        V = R.sample(img, fixation)
        tight = R.backproject_tight_last()
        
        cv2.namedWindow("inverted", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("inverted", tight) 
        
        cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("input", img) 
        
        key = cv2.waitKey(10)
        
        if key == 43: #+
            print('')
#        elif key == 45: #-
#            print ''
#        elif key == 97: #a
#            print 'switching autoscaling...'
#            cv2.destroyAllWindows()
#            autoscale = not autoscale
#        elif key == 105: #'i
#            cv2.destroyWindow("inverted")
#            showInverse = not showInverse            
#        elif key == 99: #c
#            showCortex = not showCortex
#            cv2.destroyWindow("cortex")
#        elif key == 119: #w
#            imShrink += 1
#            imShrink = min(6, imShrink)
#            prep()
#        elif key == 115: #s
#            imShrink -= 1
#            imShrink = max(imShrink, 1)
#            prep()
        elif key == 27: #esc
            break
#        elif key != -1:
#            print key
            
utils.camclose(cap)
