# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:16:42 2019

@author: villa
"""

from Fct_generales import *

import matplotlib.pyplot as plt
import pydicom
import os
import skimage.io as io
from scipy import ndimage
from scipy import misc
import skimage.morphology.selem
from skimage.morphology import disk
from skimage import filters
from skimage import exposure
from skimage.restoration import denoise_nl_means, estimate_sigma

def normalisation(im):
    m=np.max(im)
    return(im/m)

# ======== Chargement des masques =======

DIR="D:/Documents/Travail/DATASIm/Projet/Fonctions_generales/Phantoms/"
DIRMASK="D:/Documents/Travail/DATASIm/Projet/Fonctions_generales/MASK/"

im1=pydicom.dcmread(DIR+"2.16.840.1.113669.632.20.20130917.192819263.200064.384.dcm")
m1=io.imread(DIRMASK+"PM_mask.png")
[masses1,mask1]=redim_im_bis(im1.pixel_array,m1)
[masses1,mask1]=isoler(masses1,mask1,[0.603,0.722],[0.436,0.9])
mask1=normalisation(mask1[:,:,3])
masses1=normalisation(masses1)

im2=pydicom.dcmread(DIR+"hologic-MG02.dcm")
m2=io.imread(DIRMASK+"Hologic_mask.png")
[masses2,mask2]=redim_im_bis(im2.pixel_array,m2)
[masses2,mask2]=isoler(masses2,mask2,[0.603,0.722],[0.436,0.9])
mask2=normalisation(mask2[:,:,3])
masses2=normalisation(masses2)
masses2=1-masses2

im3=pydicom.dcmread(DIR+"ge-0001-0000-00000000.dcm")
m3=io.imread(DIRMASK+"GE_mask.png")
[masses3,mask3]=redim_im_bis(im3.pixel_array,m3)
[masses3,mask3]=isoler(masses3,mask3,[0.603,0.722],[0.436,0.9])
mask3=normalisation(mask3[:,:,3])
masses3=normalisation(masses3)
masses3=1-masses3

im4=pydicom.dcmread(DIR+"1.2.392.200036.9125.4.0.2718896371.50333032.466243176.dcm")
m4=io.imread(DIRMASK+"FUJI_mask.png")
[masses4,mask4]=redim_im_bis(im4.pixel_array,m4)
[masses4,mask4]=isoler(masses4,mask4,[0.603,0.722],[0.436,0.9])#0.605
mask4=normalisation(mask4[:,:,3])
masses4=normalisation(masses4)

"""
Création des fonds

fond1=ndimage.convolve(masses1,meanKernel(101))
np.save("fond_PM.npy",fond1)
fond2=ndimage.convolve(masses2,meanKernel(101))
np.save("fond_Holo.npy",fond2)
fond3=ndimage.convolve(masses3,meanKernel(101))
np.save("fond_GE.npy",fond3)
fond4=ndimage.convolve(masses4,meanKernel(101))
np.save("fond_fuji.npy",fond4)


"""
Fond1=np.load("fond1.npy")
Fond2=np.load("fond2.npy")
Fond2=1-Fond2
Fond3=np.load("fond3.npy")
Fond3=1-Fond3
Fond4=np.load("fond4.npy")

#========== Partie traitements =========

def pipeline_corr(im,block_size = 8,disk_size=5):
    im_eq=skimage.exposure.equalize_hist(im)
    im=skimage.filters.sobel(im_eq)-im_eq  
    sigma_est = np.mean(estimate_sigma(im, multichannel=True))
    im_nl = denoise_nl_means(im, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
    im_med=scipy.signal.medfilt(im_nl,7)
    im=ndimage.convolve(im_med,gaussianKernel(block_size,max(block_size//4,1)))
    im=scipy.signal.medfilt(im,7)
    im_corr=scipy.signal.correlate(im,skimage.morphology.selem.disk(disk_size))
    t=skimage.filters.threshold_minimum(im_corr)
    imt=(im_corr>t).astype(int)
    return(imt)
    
patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)

def pipeline_finale(taille_masse=0.85):
    D4=round(taille_masse/im4.ImagerPixelSpacing[0])
    #On fixe le bon rayon à avoir pour im4, et ensuite on ajuste les autres proportionellement au carré
    #du rapport des pixels, parceque l'on considère des volumes
    D1=round(D4*(1/im1.ImagerPixelSpacing[0]*im4.ImagerPixelSpacing[0])**2)
    D2=round(D4*(1/im2.ImagerPixelSpacing[0]*im4.ImagerPixelSpacing[0])**2)
    D3=round(D4*(1/im3.ImagerPixelSpacing[0]*im4.ImagerPixelSpacing[0])**2)
    res1=pipeline_corr(masses1-Fond1,block_size=D1//2,disk_size=D1)#Application de la pipeline
    res2=pipeline_corr(masses2-Fond2,block_size=D2//2,disk_size=D2)
    res3=pipeline_corr(masses3-Fond3,block_size=D3//2,disk_size=D3)
    res4=pipeline_corr(masses4-Fond4,block_size=D4//2,disk_size=D4)
    res1=res1[D1:-D1,D1:-D1]
    res2=res2[D2:-D2,D2:-D2]
    res3=res3[D3:-D3,D3:-D3]#Redimensionnement pour enlever l'effet de la corrélation sur la taille
    res4=res4[D4:-D4,D4:-D4]#et ainsi pouvoir comparer au mask
    [l1,n1]=scipy.ndimage.measurements.label(res1) #Fonction pour compter le nombre de masses détectées
    [l2,n2]=scipy.ndimage.measurements.label(res2)#Apporte un autre point de vue du score
    [l3,n3]=scipy.ndimage.measurements.label(res3)
    [l4,n4]=scipy.ndimage.measurements.label(res4)
    M1=ajoutermask(res1,mask1) #Pour pouvoir afficher les résultats sur le masque
    M2=ajoutermask(res2,mask2)
    M3=ajoutermask(res3,mask3)
    M4=ajoutermask(res4,mask4)
    resultats_masses(res1,mask1) #Affichage des résultats chiffrés
    resultats_masses(res2,mask2)
    resultats_masses(res3,mask3)
    resultats_masses(res4,mask4)
    plt.subplot(4,1,1) #Il reste à afficher les masses et les mask sur les mêmes images
    plt.imshow(M1)
    plt.subplot(4,1,2)
    plt.imshow(M2)
    plt.subplot(4,1,3)
    plt.imshow(M3)
    plt.subplot(4,1,4)
    plt.imshow(M4)
    return(l1,l2,l3,l4)
    
def resultats_masses(im,mask):
    [im1,m1]=isoler(im,mask,[0.5,1],[0.738,1])
    [im2,m2]=isoler(im,mask,[0.5,1],[0.424,0.738])
    [im3,m3]=isoler(im,mask,[0.5,1],[0.157,0.424])
    [im4,m4]=isoler(im,mask,[0,1],[0,0.157])
    [im5,m5]=isoler(im,mask,[0,0.5],[0.738,1])
    [im6,m6]=isoler(im,mask,[0,0.5],[0.424,0.738])
    [im7,m7]=isoler(im,mask,[0,0.5],[0.157,0.424])
    print ('Les résultats globaux sont :')
    resultat(im,mask)
    print("Les résultats de la masse n°1 sont :")
    resultat(im1,m1)
    print("Les résultats de la masse n°2 sont :")
    resultat(im2,m2)
    print("Les résultats de la masse n°3 sont :")
    resultat(im3,m3)
    print("Les résultats de la masse n°4 sont :")
    resultat(im4,m4)
    print("Les résultats de la masse n°5 sont :")
    resultat(im5,m5)
    print("Les résultats de la masse n°6 sont :")
    resultat(im6,m6)
    print("Les résultats de la masse n°7 sont :")
    resultat(im7,m7)
    
def ajoutermask(im,mask): #Pour la visualisation des deux en superposé
    [n,p]=np.shape(im)
    M=np.zeros([n,p,3])
    M[:,:,0]=im
    M[:,:,2]=mask
    return(M)
