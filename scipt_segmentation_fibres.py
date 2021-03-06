# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:54:45 2019

@author: Gauthier Frecon

Script qui affiche les resultats pour les  4 images de références de la segmentation de la zones des fibres (PARTIE 1) et de la fibre F1 (PARTIE 2)


!!!!  Attention le script bloc par bloc (F9) pour que le script de soit pas trop long à tourner et ne pas afficher trop d'images !!!!


"""


import numpy as np
import matplotlib.pyplot as plt
import pydicom
from Fct_generales import *



#### chemins d'accès #######

IMDIR_mask=r"C:\Users\Gauthier Frecon\Documents\GitHub\Fonctions_generales\MASK"  ### A completer ####
IMDIR_phantom=r"C:\Users\Gauthier Frecon\Documents\GitHub\Fonctions_generales\Phantoms" ### A completer ####

nom_imref_ge="ge-0001-0000-00000000.dcm"
nom_imref_hologic="hologic-MG02.dcm"
nom_imref_planmed="2.16.840.1.113669.632.20.20130917.192819263.200064.384.dcm"
nom_imref_fuji="1.2.392.200036.9125.4.0.2718896371.50333032.466243176.dcm"





''' PARTIE 1:  Résultats segmentation toutes fibres (pipeline_toute_fibre)  '''





plt.close('all')
#### ge 
print('\n', 'ge:')
chemin_ge=IMDIR_phantom + "/" + nom_imref_ge
ds = pydicom.dcmread(chemin_ge)
imref_ge=ds.pixel_array
mask_ge=plt.imread(IMDIR_mask + "/" + "GE_mask.png")[:,:,3]
[fibres_ge, fibres_ge_mask, mesures_ge]=pipeline_toute_fibre(imref_ge, mask_ge)


plt.close('all')
#### hologic
chemin=IMDIR_phantom + "/" + nom_imref_hologic
ds = pydicom.dcmread(chemin)
imref_hologic=ds.pixel_array
mask_hologic=plt.imread(IMDIR_mask + "/" + "Hologic_mask.png")[:,:,3]
print('\n', 'hologic:')
[fibres_hologic, fibres_hologic_mask, mesures_hologic]=pipeline_toute_fibre(imref_hologic, mask_hologic)


plt.close('all')
#### fuji
print('\n', 'fuji:')
chemin=IMDIR_phantom + "/" + nom_imref_fuji
ds = pydicom.dcmread(chemin)
imref_fuji=ds.pixel_array
mask_fuji=plt.imread(IMDIR_mask + "/" + "FUJI_mask.png")[:,:,3]
[fibres_fuji, fibres_fuji_mask, mesures_fuji]=pipeline_toute_fibre(imref_fuji ,mask_fuji, zone_fibre_n=[0.05,0.39], zone_fibre_p=[0.25,0.80]) 


plt.close('all')
#### planmed
print('\n', 'planmed:')
chemin_planmed=IMDIR_phantom + "/" + nom_imref_planmed
ds = pydicom.dcmread(chemin_planmed)
imref_planmed=ds.pixel_array
mask_planmed=plt.imread(IMDIR_mask + "/" + "PM_mask.png")[:,:,3]
[fibres_planmed, fibres_planmed_mask, mesures_ge]=pipeline_toute_fibre(imref_planmed, mask_planmed,zone_fibre_n=[0.11, 0.44], zone_fibre_p=[0.295, 0.85])






'''  PARTIE2: Résultats segmenation Fibre F1 (pipeline_segm_fibre)  '''





plt.close('all')
#### ge
print('\n', 'ge:')
chemin_ge=IMDIR_phantom + "/" + nom_imref_ge
ds = pydicom.dcmread(chemin_ge)
imref_ge=ds.pixel_array
mask_ge=plt.imread(IMDIR_mask + "/" + "GE_mask.png")[:,:,3]
[F1_ge, F1_ge_mask, mesures_ge]=pipeline_segm_fibre(imref_ge, mask_ge, zone_fibre_n=[0.10,0.21], zone_fibre_p=[0.66,0.83])
#F5_ge=pipeline_segm_fibre(imref_ge,zone_fibre_n=[0.21,0.31], zone_fibre_p=[0.66,0.80])


plt.close('all')
#### hologic
chemin=IMDIR_phantom + "/" + nom_imref_hologic
ds = pydicom.dcmread(chemin)
imref_hologic=ds.pixel_array
mask_hologic=plt.imread(IMDIR_mask + "/" + "Hologic_mask.png")[:,:,3]
print('\n', 'hologic:')
[F1_hologic, F1_hologic_mask, mesures_hologic]=pipeline_segm_fibre(imref_hologic, mask_hologic,zone_fibre_n=[0.09,0.20], zone_fibre_p=[0.68,0.84]) 


plt.close('all')
#### fuji
print('\n', 'fuji:')
chemin=IMDIR_phantom + "/" + nom_imref_fuji
ds = pydicom.dcmread(chemin)
imref_fuji=ds.pixel_array
mask_fuji=plt.imread(IMDIR_mask + "/" + "FUJI_mask.png")[:,:,3]
[F1_fuji, F1_fuji_mask, mesures_fuji]=pipeline_segm_fibre(imref_fuji ,mask_fuji, zone_fibre_n=[0.07,0.15], zone_fibre_p=[0.66,0.81]) #, seuil2=80) #mettre un angle de 150 pour faire un truc propre (angle de 60 deg au lieu de 45 deg)


plt.close('all')
#### planmed
print('\n', 'planmed:')
chemin_planmed=IMDIR_phantom + "/" + nom_imref_planmed
ds = pydicom.dcmread(chemin_planmed)
imref_planmed=ds.pixel_array
mask_planmed=plt.imread(IMDIR_mask + "/" + "PM_mask.png")[:,:,3]
[F1_planmed, F1_planmed_mask, mesures_ge]=pipeline_segm_fibre(imref_planmed, mask_planmed)








