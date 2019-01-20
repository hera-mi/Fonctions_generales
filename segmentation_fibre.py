# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:11:42 2019

Segmentation des fibres

@author: Gauthier Frecon


"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
from scipy import ndimage
import math
from scipy import signal
import skimage.io as io
from skimage.transform import rotate

#ouverture de l'image
IMDIR="D:/Documents/Projet Mammographie/datasim-prj-phantoms-planmed-dataset-201812061411/datasim-prj-phantoms-dataset-201812061411/digital/2.16.840.1.113669.632.20.20130917.192726317.17.381"
nom_im="2.16.840.1.113669.632.20.20130917.192819263.200064.384.dcm"
ds = pydicom.dcmread(IMDIR+ "/" + nom_im)
im=ds.pixel_array
[n,p]=np.shape(ds.pixel_array)
#plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 


zone_fibres=isoler(im, [0.1,0.41], [0.50,0.95]) #partie de l'image correspondant aux fibres
plt.imshow(zone_fibres, cmap=plt.cm.bone) 
fibre_F1=isoler(im, [0.15,0.23], [0.80,0.91]) #partie de l'image correspondant à la fibre F1
plt.imshow(fibre_F1, cmap=plt.cm.bone) 
fibre_F1_inverted=linear(fibre_F1, -1, 10000) #fibre F1 avec pixels inversé (blanc=noir et inversement)
plt.imshow(fibre_F1_inverted, cmap=plt.cm.bone)
plt.grid()
plt.show()




mask1=np.zeros_like(fibre_F1_inverted)
#mask1[120:124,:]=np.ones([4,236])
#mask1[118:126,100:136]=np.ones([8,36])
mask1[120:124,105:131]=np.ones([4,26])
mask1=rotate(mask1, 45)
plt.imshow(mask1, cmap='gray')
plt.show()


corr_mask1=signal.correlate(fibre_F1_inverted,mask1, mode='same')
plt.imshow(corr_mask1, cmap='gray')
plt.show()

max_corr_mask1=np.max(corr_mask1)
min_corr_mask1=np.min(corr_mask1)
y, x = np.histogram(corr_mask1, bins=np.arange(min_corr_mask1,max_corr_mask1))
fig, ax = plt.subplots()
plt.plot(x[:-1], y)
plt.show()

plt.imshow(corr_mask1>10.39, cmap='gray') 
plt.show()

#####
mask2=np.zeros_like(fibre_F1_inverted)
mask2[120:124,100:136]=np.ones([4,36])
mask2=rotate(mask2, 135)
mask2[0,0]=np.max(mask2)*np.ones([1,1])
plt.imshow(mask2, cmap='gray')
plt.title("mask")
plt.show()
corr_mask2=signal.correlate(fibre_F1_inverted, mask2, mode='same')
plt.imshow(corr_mask2, cmap='gray')
plt.show()

max_corr_mask2=np.max(corr_mask2)
min_corr_mask2=np.min(corr_mask2)
y, x = np.histogram(corr_mask2, bins=np.arange(min_corr_mask2,max_corr_mask2))
fig, ax = plt.subplots()
plt.plot(x[:-1], y)
plt.show()

plt.imshow(corr_mask2>14.3, cmap='gray')
plt.show()

