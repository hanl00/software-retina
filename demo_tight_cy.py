#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:51:21 2019

Tight retina image demo trial
@author: Piotr Ozimek
"""
import cv2
import numpy as np
# from retinavision.retina import Retina
# from retinavision.cortex import Cortex
# from retinavision import utils
from os.path import join
import sys
import time

sys.path.append('C:\\Users\\Nicholas\\Documents\\University @ Glasgow\\Year 5\\cythonised_retina\\retinavision_cython\\retina')
import retina_sample
import retina_utils

#Open webcam
cap = retina_utils.camopen() #cap is the capture object (global)
ret, campic = cap.read()

#Create and load retina
R = retina_sample.Retina()
# R.info()
# R.loadLoc(join(datadir, "retinas", "ret50k_loc.pkl"))
# R.loadCoeff(join(datadir, "retinas", "ret50k_coeff.pkl"))
retina_sample.loadCoeff()
retina_sample.loadLoc()
R.updateLoc()

#Prepare retina
x = campic.shape[1]/2
y = campic.shape[0]/2
fixation = (y,x)
R.prepare(campic.shape, fixation)

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0

while True:
    ret, img = cap.read()
    if ret is True:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = input_img.astype(np.float64)
        V = R.sample(img, fixation)
        #tight = R.backproject_tight_last()
        tight = R.backproject_tight_last()

        # font which we will be using to display FPS 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # time when we finish processing for this frame 
        new_frame_time = time.time() 
  
        # Calculating the fps 
  
        # fps will be number of frame processed in given time frame 
        # since their will be most of time error of 0.001 second 
        # we will be subtracting it to get more accurate result 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
  
        # converting the fps into integer 
        # fps = int(fps) 
  
        # converting the fps to string so that we can display it on frame 
        # by using putText function 
        fps = str(fps) 

        # puting the FPS count on the frame 
        cv2.putText(tight, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        
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
            
retina_utils.camclose(cap)
