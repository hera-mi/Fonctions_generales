# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:54:45 2019

@author: Gauthier Frecon

à faire: 
    
-voir seuil générale
-fair une étude détaillée sur une iamge de ref (à moitié déja fait)

-calculer pr chaque image les valeurs moyenne de la glande et de la graisse (àmoitié ok)
-seuil en fonctio du tiret au dessus (pas à faire si yen marche)

-mettre au propre redim

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


IMDIR_fuji=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-fujifilm-dataset-201812130858\datasim-prj-phantoms-fuji-dataset-201812130858\1.2.392.200036.9125.3.1602111935212811.64878483013.3092121"
IMDIR_planmed=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-planmed-dataset-201812061411\datasim-prj-phantoms-dataset-201812061411\digital\2.16.840.1.113669.632.20.20140513.192554394.19.415"
IMDIR_hologic=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-hologic-20190125-mg-proc"
IMDIR_ge=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-ge-20190125-mg-proc"

nom_imref_ge="ge-0001-0000-00000000.dcm"
nom_imref_hologic="hologic-MG02.dcm"
nom_imref_planmed="2.16.840.1.113669.632.20.20140513.202406491.200064.424.dcm"
nom_imref_fuji="1.2.392.200036.9125.4.0.3826927078.1023464352.885066624.dcm"



#planmed
chemin=IMDIR_planmed + "/" + nom_imref_planmed
ds = pydicom.dcmread(chemin)
imref_planmed=ds.pixel_array
F1_planmed=pipeline_segm_fibre(imref_planmed)

#ge
chemin=IMDIR_ge + "/" + nom_imref_ge
ds = pydicom.dcmread(chemin)
imref_ge=ds.pixel_array
F1_ge=pipeline_segm_fibre(imref_ge,zone_fibre_n=[0.10,0.21], zone_fibre_p=[0.66,0.83])

# hologic
chemin=IMDIR_hologic + "/" + nom_imref_hologic
ds = pydicom.dcmread(chemin)
imref_hologic=ds.pixel_array
F1_hologic=pipeline_segm_fibre(imref_hologic,zone_fibre_n=[0.09,0.20], zone_fibre_p=[0.68,0.84]) 

#fuji
chemin=IMDIR_fuji + "/" + nom_imref_fuji
ds = pydicom.dcmread(chemin)
imref_fuji=ds.pixel_array
np.save("imref_fuji.npy", imref_fuji)
F1_fuji=pipeline_segm_fibre(imref_fuji,zone_fibre_n=[0.07,0.15], zone_fibre_p=[0.66,0.81]) #, seuil2=80)




plt.close('all')
plt.figure(1)
plt.imshow(redim_im(imref_ge), cmap='gray')
plt.show()

plt.figure(2)
plt.imshow(redim_im(-imref_planmed+np.max(imref_planmed)))
plt.show()










