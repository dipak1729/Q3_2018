# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 12:32:24 2018

@author: q1012576
"""

# Import dependencies #
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image and convert to grayscale #
im = cv2.imread("sample_image.jpg")
im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Sampling function definition #
def sampler(image, r_size, c_size, r_step, c_step, padding, pad_val):
    
    # Original image shape #
    im_r_size, im_c_size = image.shape
    
    # Residual row and column counts #
    res_r_size = (im_r_size - r_size) % r_step
    res_c_size = (im_c_size - c_size) % c_step
    
    if padding == True:
        
        # Padding row and column counts to be added #
        pad_r_size = r_step - res_r_size
        pad_c_size = c_step - res_c_size
        
        # Padding the image with specified value #
        pad_im = np.zeros((im_r_size+pad_r_size, im_c_size+pad_c_size), image.dtype)
        pad_im[:im_r_size, :im_c_size] = image
        pad_im[im_r_size:im_r_size+pad_r_size, :im_c_size] = pad_val
        pad_im[:im_r_size, im_c_size:im_c_size+pad_c_size] = pad_val
        pad_im[im_r_size:im_r_size+pad_r_size, im_c_size:im_c_size+pad_c_size] = pad_val
        
        # Padded image shape #
        pad_im_r_size, pad_im_c_size = pad_im.shape
        
        # Sampling loop counts #
        final_im = pad_im
        loop_r_count = 1 + ((pad_im_r_size - r_size) / r_step)
        loop_c_count = 1 + ((pad_im_c_size - c_size) / c_step)
        
        # Saving the padded image #
        cv2.imwrite("Padded_image.jpg", pad_im)
        
    elif padding == False:
        
        # Truncated row and column counts to be deleted #
        trun_r_size = res_r_size
        trun_c_size = res_c_size
        
        # Truncating the image #
        trun_im = image[:im_r_size-trun_r_size, :im_c_size-trun_c_size]
        
        # Truncated image shape #
        trun_im_r_size, trun_im_c_size = trun_im.shape
        
        # Sampling loop counts #
        final_im = trun_im
        loop_r_count = 1 + ((trun_im_r_size - r_size) / r_step)
        loop_c_count = 1 + ((trun_im_c_size - c_size) / c_step)
        
        # Saving the padded image #
        cv2.imwrite("Truncated_image.jpg", trun_im)
        
    # Sampling the image #
    im_list = list()
    for i in range(int(loop_r_count)):
        for j in range(int(loop_c_count)):
            im_s = final_im[i*r_step:i*r_step+r_size, j*c_step:j*c_step+c_size]
            im_list.append(im_s)
    return im_list 

# Calling the sampling function and saving the sampled images #
loop_c = sampler(im_g, 200, 200, 100, 100, True, 20)
for c, k in enumerate(loop_c):
    cv2.imwrite(str(c)+".jpg", k)

