# -*- coding: utf-8 -*-
"""
Gauthier

-isolement de la zone fibre
-isolement de F1 et isolement de F1
-inversion des pixels pour avoir les fibres en blanc et le fond en noir
-égalisation des histogrammes

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
IMDIR=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-planmed-dataset-201812061411\datasim-prj-phantoms-dataset-201812061411\digital\2.16.840.1.113669.632.20.20130917.192726317.17.381"
nom_im1="2.16.840.1.113669.632.20.20130917.192819263.200064.384.dcm"
IMDIR2=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-planmed-dataset-201812061411\datasim-prj-phantoms-dataset-201812061411\digital\2.16.840.1.113669.632.20.20120425.94758942.2456.10"
nom_im2="2.16.840.1.113669.632.20.20120425.94850094.200083.40.dcm"

IMDIR_fuji="D:\Documents\Projet Mammographie\datasim-prj-phantoms-fujifilm-dataset-201812130858\datasim-prj-phantoms-fuji-dataset-201812130858"
IMDIR_planmed="D:\Documents\Projet Mammographie\datasim-prj-phantoms-planmed-dataset-201812061411\datasim-prj-phantoms-dataset-201812061411\digital"
IMDIR_hologic=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-hologic-20190125-mg-proc"
IMDIR_ge=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-ge-20190125-mg-proc"

nom_im1_ge="ge-0001-0000-00000000.dcm"
nom_im1_hologic="hologic-MG02.dcm"


chemin=IMDIR + "/" + nom_im1
ds = pydicom.dcmread(chemin)
im=ds.pixel_array
[n,p]=np.shape(ds.pixel_array)

#zones fibres 


zone_fibres=isoler(im, [0.13,0.40], [0.55,0.91]) #partie de l'image correspondant aux fibres
zone_fibres_inverted=linear(zone_fibres, -1, 10000)
max_zone_fibres=np.max(zone_fibres)
min_zone_fibres=np.min(zone_fibres)


#fibre F1
fibre_F1=isoler(im, [0.15,0.23], [0.80,0.91]) #partie de l'image correspondant à la fibre F1
fibre_F1_inverted=linear(fibre_F1, -1, 10000)
fibre_F1_inverted=equalize_adapthist(fibre_F1_inverted) #fibre F1 avec pixels inversé (blanc=noir et inversement)

#fibre F2
fibre_F5=isoler(im, [0.24,0.31], [0.80,0.90])
fibre_F5_inverted=linear(fibre_F5, -1, 10000)
fibre_F5_inverted=equalize_adapthist(fibre_F5_inverted) 
#plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 


