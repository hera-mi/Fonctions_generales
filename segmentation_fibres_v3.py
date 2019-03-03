# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 21:44:15 2019

@author: Gauthier Frecon


script à appliquer à partir des zones des fibres inversées et égalisées:
    
-filtrage passe haut pour enlever le gradient
-corrélation des deux mask
-OU logique
-pas besoin de faire d'étiquettage




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

from Fct_generales import *

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

plt.close('all')



#redim
[n,p]=np.shape(im)
if np.mean(im[n//2-10:n//2+10 , 0:20]) > np.mean(im[n//2-10:n//2+10 , p-21:p-1]) :
     im=-im+np.max(im)
  
im_red=redim_im(im)
[n,p]=np.shape(im_red)

#isolement fibre f1
 
fibre_F1=isoler(im_red, [0.11,0.22], [0.70,0.86])
         
#np.array([s])


# traitement fibre F1



fibre_F1=equalize_adapthist(fibre_F1)
fftc_highpass=highpass_filter(fibre_F1,Dc=5)
fft_highpass=np.fft.ifftshift(fftc_highpass)
invfft_highpass=np.real(np.fft.ifft2(fft_highpass))


im_highpass=invfft_highpass 

im_highpass=scipy.signal.medfilt(skimage.restoration.denoise_nl_means(invfft_highpass))


im_corr_I1=correlation_mask_I(im_highpass,4,40, seuil=31, angle=45) 
im_corr_I2=correlation_mask_I(im_highpass,5,40, seuil=26, angle=135) #4,20, seuil=193, angle=135)

im_segmentation_F1= (im_corr_I1+im_corr_I2)
plt.figure(2)
plt.imshow(im_segmentation_F1, cmap='gray')
plt.show()

#traitement fibre F5

#fibre F5
fibre_F5=isoler(im, [0.24,0.31], [0.80,0.90])
fibre_F5=equalize_adapthist(fibre_F5) 
#plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 

fftc_highpass=highpass_filter(fibre_F5,Dc=5)
fft_highpass=np.fft.ifftshift(fftc_highpass)
invfft_highpass=np.real(np.fft.ifft2(fft_highpass))

im_highpass=invfft_highpass
invfft_highpass=scipy.signal.medfilt(skimage.restoration.denoise_nl_means(invfft_highpass)) 

im_corr_I1=correlation_mask_I(im_highpass,3,40, seuil=28, angle=45) 
im_corr_I2=correlation_mask_I(im_highpass,4,60, seuil=26, angle=135) 
im_segmentation_F5= (im_corr_I1+im_corr_I2)
plt.figure(4)
plt.imshow(im_segmentation_F5, cmap='gray')
plt.show()

# traitement toute fibre
#fftc_highpass=highpass_filter(zone_fibres_inverted,Dc=5)
#fft_highpass=np.fft.ifftshift(fftc_highpass)
#invfft_highpass=np.real(np.fft.ifft2(fft_highpass))
#
#im_highpass=invfft_highpass /10000
#im_corr_I1=correlation_mask_I(im_highpass,3,40, seuil=5, angle=45) 

# zones carrés noir et blanc

m=n//2
x_bas=525
x_haut=575
y_matblanc_bas=m-30
y_matblanc_haut=m+30
y_matnoir_bas=m+125-30
y_matnoir_haut=m+125+30



moyenne_matblanc=np.mean(im_red[y_matblanc_bas: y_matblanc_haut, x_bas:x_haut])
moyenne_matnoir=np.mean(im_red[y_matnoir_bas: y_matnoir_haut, x_bas:x_haut])              

plt.figure()
plt.imshow(im_red, cmap='gray')
plt.plot([0,p], [m, m])
plt.plot([x_bas, x_bas,x_haut, x_haut, x_bas], [y_matblanc_bas, y_matblanc_haut, y_matblanc_haut, y_matblanc_bas,y_matblanc_bas])
plt.plot([x_bas, x_bas,x_haut, x_haut, x_bas], [y_matnoir_bas, y_matnoir_haut, y_matnoir_haut, y_matnoir_bas, y_matnoir_bas])
plt.show()



