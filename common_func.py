#!/usr/bin/env python
# -*- coding: utf-8 -*-

'A conmmon functions module'

__author__ = 'KoalaZB'

import sys
import cv2
import numpy as np
import matplotlib.mlab as mlab    
import matplotlib.pyplot as plt  

'''
    Draw histograph of an image

    Parameters:
        -arr: a array of source datas
        -name: a title of histograph 
'''
def draw_histo(arr,name):  
	fig = plt.figure()  
	n, bins, patches = plt.hist(arr, bins=256, normed=1,edgecolor='None',facecolor='red')  
	plt.xlabel("X-axis")  
	plt.ylabel("Y-axis")  
	plt.title(name)  
	plt.show()    
	
'''
    Print image pixels in a matrix way

    Parameters:
        -img_matrix: a cv2 instance of image img_matrix
'''
def print_Image_Matrix(img_matrix):
	rows = len(img_matrix)
	cols = len(img_matrix[0])
	for i in range(10):
		tmp_row = []
		for j in range(10):
			tmp_row.append(img_matrix[i][j])
		print tmp_row