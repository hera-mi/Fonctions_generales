# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:54:45 2019

@author: Gauthier Frecon

#à faire: 

-faire pipeline pour toutes fibres
-Etiquettage des branches
-scoring : on a trouvé 5 fibres 
-gérer le redimensionnement
-canal vert = segm, canal rouge= mask
-jouer sur longueur Lx,Ly dans corrélation
-spacing seb



#facultatif:
-valeurs moyenne de la glande et de la graisse peut-etre à exploiter
-mettre au propre redim_im (la fonction marche dnc pas obligé)

#remarques:
-fuji: angle de 150° mieux que 135 car la fibre est mal positionné.
- yen est adpaté à la zone de la fibre
-lire papier de yen: compliqué...


"""

import skimage 
import scipy
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
from Fct_generales import *

plt.close('all')



IMDIR_mask=r"C:\Users\Gauthier Frecon\Documents\GitHub\Fonctions_generales\MASK"
IMDIR_phantom=r"C:\Users\Gauthier Frecon\Documents\GitHub\Fonctions_generales\Phantoms"

nom_imref_ge="ge-0001-0000-00000000.dcm"
nom_imref_hologic="hologic-MG02.dcm"
nom_imref_planmed="2.16.840.1.113669.632.20.20130917.192819263.200064.384.dcm"
nom_imref_fuji="1.2.392.200036.9125.4.0.2718896371.50333032.466243176.dcm"





#ge
print('\n', 'ge:')
chemin_ge=IMDIR_phantom + "/" + nom_imref_ge
ds = pydicom.dcmread(chemin_ge)
imref_ge=ds.pixel_array
mask_ge=plt.imread(IMDIR_mask + "/" + "GE_mask.png")[:,:,3]
[F1_ge, F1_ge_mask, mesures_ge]=pipeline_segm_fibre(imref_ge, mask_ge, zone_fibre_n=[0.10,0.21], zone_fibre_p=[0.66,0.83])
#F5_ge=pipeline_segm_fibre(imref_ge,zone_fibre_n=[0.21,0.31], zone_fibre_p=[0.66,0.80])



# hologic

chemin=IMDIR_phantom + "/" + nom_imref_hologic
ds = pydicom.dcmread(chemin)
imref_hologic=ds.pixel_array
mask_hologic=plt.imread(IMDIR_mask + "/" + "Hologic_mask.png")[:,:,3]
print('\n', 'hologic:')
[F1_hologic, F1_hologic_mask, mesures_hologic]=pipeline_segm_fibre(imref_hologic, mask_hologic,zone_fibre_n=[0.09,0.20], zone_fibre_p=[0.68,0.84]) 

#fuji
print('\n', 'fuji:')
chemin=IMDIR_phantom + "/" + nom_imref_fuji
ds = pydicom.dcmread(chemin)
imref_fuji=ds.pixel_array
mask_fuji=plt.imread(IMDIR_mask + "/" + "FUJI_mask.png")[:,:,3]
[F1_fuji, F1_fuji_mask, mesures_fuji]=pipeline_segm_fibre(imref_fuji ,mask_fuji, zone_fibre_n=[0.07,0.15], zone_fibre_p=[0.66,0.81]) #, seuil2=80) #mettre un angle de 150 pour faire un truc propre (angle de 60 deg au lieu de 45 deg)

#planmed
print('\n', 'planmed:')
chemin_planmed=IMDIR_phantom + "/" + nom_imref_planmed
ds = pydicom.dcmread(chemin_planmed)
imref_planmed=ds.pixel_array
mask_planmed=plt.imread(IMDIR_mask + "/" + "PM_mask.png")[:,:,3]
[F1_planmed, F1_planmed_mask, mesures_ge]=pipeline_segm_fibre(imref_planmed, mask_planmed)




##
#plt.close('all')
#plt.figure(1)
#
#ims=skimage.filters.sobel(imref_hologic)
#plt.imshow(ims)
#plt.show()
#
#plt.figure(2)
#plt.imshow(redim_im(-imref_planmed+np.max(imref_planmed)), cmap='gray')
#plt.show()
#









