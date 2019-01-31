# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
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
from skimage.exposure import equalize_adapthist

#ouverture de l'image
IMDIR="D:/Documents/Projet Mammographie/datasim-prj-phantoms-planmed-dataset-201812061411/datasim-prj-phantoms-dataset-201812061411/digital/2.16.840.1.113669.632.20.20130917.192726317.17.381"
nom_im="2.16.840.1.113669.632.20.20130917.192819263.200064.384.dcm"
ds = pydicom.dcmread(IMDIR+ "/" + nom_im)
im=ds.pixel_array
[n,p]=np.shape(ds.pixel_array)

zone_fibres=isoler(im, [0.13,0.40], [0.55,0.91]) #partie de l'image correspondant aux fibres
zone_fibres_inverted=linear(zone_fibres, -1, 10000)

#fibre F1
fibre_F1=isoler(im, [0.15,0.23], [0.80,0.91]) #partie de l'image correspondant à la fibre F1
fibre_F1_inverted=linear(fibre_F1, -1, 10000)
fibre_F1_inverted=equalize_adapthist(fibre_F1_inverted) #fibre F1 avec pixels inversé (blanc=noir et inversement)

#fibre F2
fibre_F5=isoler(im, [0.24,0.31], [0.80,0.90])
fibre_F5_inverted=linear(fibre_F5, -1, 10000)
fibre_F5_inverted=equalize_adapthist(fibre_F5_inverted) 
#plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 


