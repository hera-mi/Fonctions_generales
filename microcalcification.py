#LAANAITI ABDERRAHMANE
#DETECTION DES MICRO-CALICIFICATIONS

import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import os
from PIL import Image
from pylab import *
import skimage.io as io
from skimage.transform import resize
from scipy import ndimage
import numpy as np
import matplotlib.image as mpimg
import cv2 
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
import numpy as np
from skimage import measure, morphology, segmentation
from skimage.exposure import equalize_adapthist
from Fct_generales import *
import time

# Debut du decompte du temps
start_time = time.time()
plt.close('all')

######### -------- CHARGEMENT DES FANTÔMES -------  #######

IMDIR1= "/Users/reda/Downloads/EI2/PROJET/Fonctions_generales/Phantoms/1.2.392.200036.9125.4.0.2718896371.50333032.466243176.dcm"
IMDIR2= "/Users/reda/Downloads/EI2/PROJET/Fonctions_generales/Phantoms/2.16.840.1.113669.632.20.20130917.192819263.200064.384.dcm"
IMDIR3= "/Users/reda/Downloads/EI2/PROJET/Fonctions_generales/Phantoms/ge-0001-0000-00000000.dcm"
IMDIR4 = "/Users/reda/Downloads/EI2/PROJET/Fonctions_generales/Phantoms/hologic-MG02.dcm"

######### -------- CHARGEMENT DES MASQUES -------  #######

mask1 ="/Users/reda/Downloads/EI2/PROJET/Fonctions_generales/MASK/FUJI_MASK.png" 
mask2 ="/Users/reda/Downloads/EI2/PROJET/Fonctions_generales/MASK/PM_mask.png" 
mask3 ="/Users/reda/Downloads/EI2/PROJET/Fonctions_generales/MASK/GE_mask.png"
mask4 ="/Users/reda/Downloads/EI2/PROJET/Fonctions_generales/MASK/Hologic_mask.png" 

######### -------- CROP DES IMAGES -------  #######

ligne1, col1 = [0.74,0.92],[0.4,0.88]
ligne2, col2 = [0.74,0.89],[0.43,0.88]
ligne3, col3 = [0.74,0.90],[0.4,0.88]
ligne4, col4 = [0.74,0.90],[0.4,0.88]

def normalisation(im):
    m=np.max(im)
    return(im/m)

######### -------- FONCTION DE TRAITEMENT -------  #######

def reda(IMDIR_im,IMDIR_mask,s,ligne,col):
    ds = pydicom.dcmread(IMDIR_im)
    im= ds.pixel_array
    mask = plt.imread(IMDIR_mask)[:,:,3] #lecture du mask
    
    #redimensionnement
    [n,p]=np.shape(im)
    if np.mean(im[n//2-10:n//2+10 , 0:20]) > np.mean(im[n//2-10:n//2+10 , p-21:p-1]) :
         im=-im+np.max(im)
    [im_red, im_mask]=redim_im(im, mask)
    [n,p]=np.shape(im_red)
    
    #filtrage 
    
    zone_bruit, m=isoler(im_red, np.zeros_like(im_red),[0.68,0.69],[0.81,0.82])
    moy_bruit =np.mean(zone_bruit)
    denoised = denoise_bilateral(im_red,win_size=3, sigma_color=moy_bruit, sigma_spatial=4, multichannel=False)
    
    if IMDIR_im == IMDIR2:
        lowpass = ndimage.gaussian_filter(im_red, 1000)
        denoised = im_red - lowpass
        #fftc_highpass=highpass_filter(denoised,Dc=200)
        #fft_highpass=np.fft.ifftshift(fftc_highpass)
        #denoised=np.real(np.fft.ifft2(fft_highpass))
    
    [invfft_highpass,i_mask]=isoler(denoised,im_mask, ligne, col)
    plt.figure()
    plt.imshow(invfft_highpass, cmap=plt.cm.gray)
    plt.title('image filtrée')
    i_mask=normalisation(i_mask)
    invfft_highpass=normalisation(invfft_highpass)
    
    #Correlation
    pas = 9
    kernel = np.zeros((2*pas+1,2*pas+1))
    for i in range(2*pas+1):
        for j in range(2*pas+1):
            if (i-9)**2+(j-9)**2<1:
                kernel[i,j]=1;
                
    #correlation avec 6 micro-calcifications ne marche pas           
    corr_mask=signal.correlate(invfft_highpass, kernel, mode='same')
    plt.figure()
    plt.imshow(corr_mask, cmap=plt.cm.gray)
    plt.title('image après corrélation')
    
    # Seuillage 
    
    t = skimage.filters.threshold_yen(corr_mask)
    Is= corr_mask > t
    
    #Labeling and counting
    structure = [[1,1,1],[1,1,1],[1,1,1]]
    [labeled_array1, num_features1]=scipy.ndimage.measurements.label(Is,structure)

    [n,p]=np.shape(Is)
    comparaison=np.ones((n,p,3))
    comparaison[:,:,0]=i_mask
    comparaison[:,:,1]=Is
    comparaison = rgb2gray(comparaison)
    
    return(num_features1,Is,i_mask,comparaison)# parametres 8 pixels)
    

######### -------- AFFICHAGE DES IMAGES TRAITEES -------  #######

[score1,IM1,mask1,sup1] = reda(IMDIR1,mask1,s_FUJI,ligne1, col1)
[score2,IM2,mask2,sup2] = reda(IMDIR2,mask2,s_PM,ligne2, col2)
[score3,IM3,mask3,sup3] = reda(IMDIR3,mask3,s_GE,ligne3, col3)
[score4,IM4,mask4,sup4] = reda(IMDIR4,mask4,s_HOLOGIC,ligne4, col4)


plt.figure()
plt.subplot(4,2,1)
plt.imshow(IM1,cmap=plt.cm.gray)
plt.axis('off')
plt.title("les images traitées")
plt.subplot(4,2,3)
plt.imshow(IM2,cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(4,2,5)
plt.imshow(IM3,cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(4,2,7)
plt.imshow(IM4,cmap=plt.cm.gray)
plt.axis('off')


plt.subplot(4,2,2)
plt.imshow(sup1)
plt.axis('off')
plt.title("superposition du mask et image traitée ")
plt.subplot(4,2,4)
plt.imshow(sup2)
plt.axis('off')
plt.subplot(4,2,6)
plt.imshow(sup3)
plt.axis('off')
plt.subplot(4,2,8)
plt.imshow(sup4)
plt.axis('off')

######### -------- COMPTER LES MICRO_CALCIFICATIONS -------  #######

print("le nombre de micro-calcifications FUJI est :",score1)
print("le nombre de micro-calcifications PM est :",score2)
print("le nombre de micro-calcifications GE est :",score3)
print("le nombre de micro-calcifications HOLOGIC est :",score4)

print("---------------------------------------------------------")
print("---------------------------------------------------------")

######### -------- AFFICHAGE DU DICE -------  #######

resultat(IM1,mask1)
resultat(IM2,mask2)
resultat(IM3,mask3)
resultat(IM4,mask4)

######### -------- TEMPS D'EXECUTION -------  #######

# Affichage du temps d execution
print("Temps d execution : %s secondes ---" % (time.time() - start_time))





























































######### -------- Brouillon -------  #######


#print("le nombre de micro-calcification après traitement dans le mask", num_features2-17)#17


"""
ds = pydicom.dcmread(IMDIR)
im= ds.pixel_array
mask = mpimg.imread("/Users/reda/Downloads/CODE_PY/GE_mask.png") #lecture du mask
plt.figure(10)
plt.imshow(mask, cmap=plt.cm.gray)
plt.axis('off')


#redimensionnement

[n,p]=np.shape(im)
if np.mean(im[n//2-10:n//2+10 , 0:20]) > np.mean(im[n//2-10:n//2+10 , p-21:p-1]) :
     im=-im+np.max(im)

im_red=redim_im(im)
[n,p]=np.shape(im_red)


#filtrage 
fftc_highpass=highpass_filter(im_red,Dc=200)
fft_highpass=np.fft.ifftshift(fftc_highpass)
invfft_highpass=np.real(np.fft.ifft2(fft_highpass))
invfft_highpass=isoler(invfft_highpass, [0.78,0.90], [0.4,0.87])

#Correlation
pas = 9
kernel = np.zeros((2*pas+1,2*pas+1))
for i in range(2*pas+1):
    for j in range(2*pas+1):
        if (i-9)**2+(j-9)**2<1:
            kernel[i,j]=1;
           
#plt.imshow(kernel, cmap=plt.cm.gray)
corr_mask=signal.correlate(invfft_highpass, kernel, mode='same')


# Seuillage 
#Is= corr_mask > 246
Is= corr_mask > 100
plt.figure(3)
plt.imshow(Is, cmap=plt.cm.gray)
plt.axis('off')

#Labeling and counting
labeled_array, num_features =scipy.ndimage.measurements.label(Is)
print("le nombre de micro-calcification après traitement",num_features)
labeled_array, num_features =scipy.ndimage.measurements.label(mask)
print("le nombre de micro-calcification après traitement dans le mask", num_features-17)#17 le nombre d'elements hors microcalcification 
"""


"""
mask=isoler(mask, [0.72,0.90], [0.4,0.97])
plt.figure(10)
plt.imshow(mask, cmap=plt.cm.gray)
plt.axis('off')
mask = mask>0
"""
######@

#sharpening+ filtering
#f = im_red
#blurred_f = ndimage.gaussian_filter(f, sigma=1)
#filter_blurred_f = ndimage.gaussian_filter(blurred_f, sigma=8)
#alpha = 60
#sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
#sharpened=isoler(f, [0.78,0.90], [0.42,0.867])
#plt.figure(1)
#plt.imshow(sharpened, cmap=plt.cm.gray)
#plt.axis('off')
#####
#fftc_highpass=highpass_filter(f,Dc=200)
#####
#plt.figure(2)
#plt.imshow(invfft_highpass, cmap=plt.cm.gray)
#plt.axis('off')

#######
#plt.figure(2)
#plt.imshow(corr_mask, cmap=plt.cm.gray)
#plt.axis('off')
#plt.figure(7)
#plt.imshow(corr_mask, cmap=plt.cm.gray)
#plt.axis('off')
#t=skimage.filters.threshold_li(corr_mask)
    #t=skimage.filters.threshold_local(corr_mask,3)
    #t=skimage.filters.threshold_niblack(corr_mask)
    #t=skimage.filters.threshold_otsu(corr_mask)
    #t = skimage.filters.threshold_isodata(corr_mask)