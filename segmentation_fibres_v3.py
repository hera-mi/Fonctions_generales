# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 21:44:15 2019

@author: Gauthier Frecon



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
from skimage.restoration import denoise_bilateral
from Fct_generales import *

#ouverture de l'image
IMDIR=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-planmed-dataset-201812061411\datasim-prj-phantoms-dataset-201812061411\digital\2.16.840.1.113669.632.20.20130917.192726317.17.381"
nom_im1="2.16.840.1.113669.632.20.20130917.192819263.200064.384.dcm"
IMDIR2=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-planmed-dataset-201812061411\datasim-prj-phantoms-dataset-201812061411\digital\2.16.840.1.113669.632.20.20120425.94758942.2456.10"
nom_im2="2.16.840.1.113669.632.20.20120425.94850094.200083.40.dcm"

IMDIR_fuji=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-fujifilm-dataset-201812130858\datasim-prj-phantoms-fuji-dataset-201812130858\1.2.392.200036.9125.3.1602111935212811.64878483013.3092121"
IMDIR_planmed=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-planmed-dataset-201812061411\datasim-prj-phantoms-dataset-201812061411\digital\2.16.840.1.113669.632.20.20140513.192554394.19.415"
IMDIR_hologic=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-hologic-20190125-mg-proc"
IMDIR_ge=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-ge-20190125-mg-proc"

nom_imref_ge="ge-0001-0000-00000000.dcm"
nom_imref_hologic="hologic-MG02.dcm"
nom_imref_planmed="2.16.840.1.113669.632.20.20140513.202406491.200064.424.dcm"
nom_imref_fuji="1.2.392.200036.9125.4.0.3826927078.1023464352.885066624.dcm"



chemin=IMDIR + "/" + nom_im1
ds = pydicom.dcmread(chemin)
im=ds.pixel_array

plt.close('all')



#redim
[n,p]=np.shape(im)
if np.mean(im[n//2-10:n//2+10 , 0:20]) > np.mean(im[n//2-10:n//2+10 , p-21:p-1]) :
     im=-im+np.max(im)
  
[im_red, m]=redim_im(im, np.zeros_like(im))
[n,p]=np.shape(im_red)

#isolement fibre f1
 
[fibre_F1, m]=isoler(im_red, np.zeros_like(im_red), [0.11,0.22], [0.70,0.86])
         
#np.array([s])


# traitement fibre F1
fibre_F1=equalize_adapthist(fibre_F1)
plt.figure()
plt.imshow(fibre_F1, cmap='gray')
plt.show()
zone_sigma,m=isoler(fibre_F1, np.zeros_like(fibre_F1),[0,0.1], [0,0.1])
sigma=np.mean(zone_sigma)
denoised = denoise_bilateral(fibre_F1,sigma_color=sigma, sigma_spatial=4, multichannel=False)
plt.figure()
plt.imshow(denoised, cmap='gray')
plt.show()

fftc_highpass=highpass_filter(denoised,Dc=3)
fft_highpass=np.fft.ifftshift(fftc_highpass)
invfft_highpass=np.real(np.fft.ifft2(fft_highpass))

plt.figure()
plt.imshow(invfft_highpass, cmap='gray')
plt.show()


im_filtree=skimage.restoration.denoise_nl_means(invfft_highpass)
plt.figure()
plt.imshow(im_filtree, cmap='gray')
plt.show()


im_corr_I1=correlation_mask_I(im_filtree,4,40, seuil=31, angle=45) 
im_corr_I2=correlation_mask_I(im_filtree,5,40, seuil=26, angle=135) #4,20, seuil=193, angle=135)

im_segmentation_F1= (im_corr_I1+im_corr_I2)
plt.figure()
plt.imshow(im_segmentation_F1, cmap='gray')
plt.show()

#traitement fibre F5

#fibre F5

fibre_F5,m =isoler(im, np.zeros_like(im), [0.24,0.31], [0.80,0.90])


fibre_F5=equalize_adapthist(fibre_F5) 
plt.figure()
plt.imshow(fibre_F5, cmap='gray')
plt.show()

#plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 
zone_sigma,m=isoler(fibre_F5, np.zeros_like(fibre_F5),[0,0.1], [0,0.1])
sigma=np.mean(zone_sigma)
denoised = denoise_bilateral(fibre_F5,win_size=5, sigma_color=sigma, sigma_spatial=4, multichannel=False)
plt.figure()
plt.imshow(denoised, cmap='gray')
plt.show()

fftc_highpass=highpass_filter(denoised,Dc=3)
fft_highpass=np.fft.ifftshift(fftc_highpass)
invfft_highpass=np.real(np.fft.ifft2(fft_highpass))
plt.figure()
plt.imshow(invfft_highpass, cmap='gray')
plt.show()

im_filtree=skimage.restoration.denoise_nl_means(invfft_highpass)
plt.figure()
plt.imshow(im_filtree, cmap='gray')
plt.show()

im_corr_I1=correlation_mask_I(im_filtree,3,40, seuil=28, angle=45) 
im_corr_I2=correlation_mask_I(im_filtree,4,60, seuil=26, angle=135) 
im_segmentation_F5= (im_corr_I1+im_corr_I2)
plt.figure()
plt.imshow(im_segmentation_F5, cmap='gray')
plt.show()



# traitement toute fibre  (marche pas parce que tous les bords du phantom perturbe l'algp)
'''
plt.close('all')

zone_fibres,m=isoler(im_red, np.zeros_like(im_red),[0.12,0.42], [0.29,0.85]) 
[n,p]=np.shape(zone_fibres)
for i in range(n):
    for j in range(p):
        if (-i)>(-370 +(380/370)*j) or (-i)<((n-500)/(p-350)*j-900) :
            zone_fibres[i,j]=0


#plt.figure()
#plt.imshow(zone_fibres, cmap='gray')
#plt.show()
#plt.close('all')
zone_fibres=equalize_adapthist(zone_fibres)
plt.figure()
plt.imshow(zone_fibres, cmap='gray')
plt.show()

zone_sigma,m=isoler(zone_fibres, np.zeros_like(fibre_F5),[0.8,0.9], [0.4,0.5])
sigma=np.mean(zone_sigma)
denoised = denoise_bilateral(zone_fibres,win_size=3, sigma_color=0.2, sigma_spatial=4, multichannel=False)
plt.figure()
plt.imshow(denoised, cmap='gray')
plt.show()
fftc_highpass=highpass_filter(denoised,Dc=3)
fft_highpass=np.fft.ifftshift(fftc_highpass)
invfft_highpass=np.real(np.fft.ifft2(fft_highpass))
plt.figure()
plt.imshow(invfft_highpass, cmap='gray') # zone au milieu des fibres
plt.show()
im_filtree=skimage.restoration.denoise_nl_means(denoised, patch_distance=2)#scipy.signal.medfilt(skimage.restoration.denoise_nl_means(  zone_fibres))#invfft_highpass)) 
plt.figure()
plt.imshow(im_filtree, cmap='gray')
plt.show()
im_corr_I1=correlation_mask_I(im_filtree,3,40, seuil=5, angle=45) 
im_corr_I2=correlation_mask_I(im_filtree,4,60, seuil=26, angle=135) 


# zones carrés noir et blanc

m=n//2
x_bas=525
x_haut=575
y_matblanc_bas=m-30
y_matblanc_haut=m+30
y_matnoir_bas=m+125-30
y_matnoir_haut=m+125+30



#moyenne_matblanc=np.mean(im_red[y_matblanc_bas: y_matblanc_haut, x_bas:x_haut])
#moyenne_matnoir=np.mean(im_red[y_matnoir_bas: y_matnoir_haut, x_bas:x_haut])              
#
#plt.figure()
#plt.imshow(im_red, cmap='gray')
#plt.plot([0,p], [m, m])
#plt.plot([x_bas, x_bas,x_haut, x_haut, x_bas], [y_matblanc_bas, y_matblanc_haut, y_matblanc_haut, y_matblanc_bas,y_matblanc_bas])
#plt.plot([x_bas, x_bas,x_haut, x_haut, x_bas], [y_matnoir_bas, y_matnoir_haut, y_matnoir_haut, y_matnoir_bas, y_matnoir_bas])
#plt.show()
'''


