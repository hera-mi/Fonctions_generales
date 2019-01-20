# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:07:44 2018

@author: villa
"""

import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import os
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage.transform import resize
import random
import scipy.stats
import skimage.morphology.selem
from math import sqrt
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from sklearn.cluster import KMeans
from skimage import data
from skimage import filters
from skimage import exposure
from skimage.transform import rotate
from skimage.segmentation import active_contour
import math
#from Filtres import *
import cv2
from skimage import img_as_ubyte

from Fct_generales import *





IMDIR="D:/Documents/Travail/DATASIm/Projet/datasim-prj-phantoms-dataset-201812061411/digital/2.16.840.1.113669.632.20.20110707.94917643.9.2/"
#filename = get_testdata_files(IMDIR+"2.16.840.1.113669.632.20.20110707.95031041.200083.50.dcm")[0]
ds = pydicom.dcmread(IMDIR+"2.16.840.1.113669.632.20.20110707.95031041.200083.50.dcm")
#plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
im= ds.pixel_array
#plt.imshow(im, cmap="hot")
#
#im2=ndimage.convolve(im,gradlap()[2])
#im3=rotate(im2,180)
#plt.imshow(im3)


    
masses=isoler(rotation(im),[0.3,0.46],[0.01,0.3])  
masses_l=ndimage.convolve(masses,gaussianKernel(10,7)) #Sinon utiliser scipy.ndimage.filters.gaussian_filter
masses_ll=ndimage.convolve(masses_l,gaussianKernel(10,7))
masses_c=ndimage.convolve(masses,gradlap()[2]) 
#masses_co=otsu(masses_c).astype(int)
#masses_lo=otsu(masses_l).astype(int)

    
s = np.linspace(0, 2*np.pi, 400)
x = 280 + 50*np.cos(s)
y = 500 + 50*np.sin(s)
init = np.array([x, y]).T

snake = active_contour(massesbis_l,init, alpha=0.015, beta=10, gamma=0.001)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(massesbis_l, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, masses.shape[1], masses.shape[0], 0])

from skimage import img_as_ubyte
cv_image = img_as_ubyte(masses)
#np.uint8(masses)
thresh=cv2.adaptiveThreshold(cv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1)
#https://pythonprogramming.net/thresholding-image-analysis-python-opencv-tutorial/

def trouver_masses(im):
    cv_im = img_as_ubyte(im)
    #plt.imshow(cv_im)
    #imgray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY) 
    im_gauss = cv2.GaussianBlur(cv_im, (5, 5), 0) 
    thresh = cv2.adaptiveThreshold(im_gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1)
# get contours 
    #Yplt.imshow(thresh)
    imb, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(imb, cmap=plt.cm.gray)
    #ax.plot(contours[:, 0], contours[:, 1], '-b', lw=3)
    return(imb,contours)
#https://stackoverrun.com/fr/q/11610670    
    
    
dirbis="D:/Documents/Travail/DATASIm/Projet/"
imbis=io.imread(dirbis+"2.16.840.1.113669.632.20.20120425.94850094.200083.40.dcm.jpg")
massesbis=isoler(rotation(imbis),[0.3,0.48],[0.01,0.36]) 

massesbis_l=ndimage.convolve(massesbis,gaussianKernel(10,7))

s = np.linspace(0, 2*np.pi, 400)
xb = 190 + 50*np.cos(s)
yb = 220 + 50*np.sin(s)
initb = np.array([xb, yb]).T

snake = active_contour(massesbis_l,initb, alpha=0.015, beta=10, gamma=0.001)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(massesbis_l, cmap=plt.cm.gray)
ax.plot(initb[:, 0], initb[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, massesbis.shape[1], massesbis.shape[0], 0])