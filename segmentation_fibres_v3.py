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
from skimage.exposure import equalize_adapthist

def correlation_mask_I(im, Lx, Ly, seuil, angle=45):
    ''' correlation de l'image avec un mask en I de taille 2*Lx   * 2*Ly puis seuillage à seuil'''
    
    mask=np.zeros_like(fibre_F1_inverted)
    mask[122-Lx:122+Lx,118-Ly:118+Ly]=np.ones([2*Lx,2*Ly])
    mask=rotate(mask, angle)
    mask[0,0]=np.max(mask)*np.ones([1,1])
    plt.imshow(mask, cmap='gray')
    plt.title("mask")
    plt.show()
    corr_mask=signal.correlate(im, mask, mode='same')
    max_corr_mask=np.max(corr_mask)
    min_corr_mask=np.min(corr_mask)
    y, x = np.histogram(corr_mask, bins=np.arange(min_corr_mask,max_corr_mask))
    fig, ax = plt.subplots()
    plt.plot(x[:-1], y)
    plt.show()
    im_corr=corr_mask>seuil
    plt.imshow(im_corr, cmap='gray')
    plt.show()
    return(im_corr)
    

# traitement fibre F1

fftc_highpass=highpass_filter(fibre_F1_inverted,Dc=5)
fft_highpass=np.fft.ifftshift(fftc_highpass)
invfft_highpass=np.real(np.fft.ifft2(fft_highpass))
plt.imshow(invfft_highpass, cmap='gray')
plt.show()

im_highpass=invfft_highpass 

#im_corr_I1=correlation_mask_I(fibre_F1_inverted,4,40, seuil=371, angle=45) 
#im_corr_I2=correlation_mask_I(fibre_F1_inverted,5,60, seuil=657, angle=135) 
#im_segmentation= (im_corr_I1+im_corr_I2)
#plt.imshow(im_segmentation, cmap='gray')
#plt.show()

im_corr_I1=correlation_mask_I(im_highpass,4,40, seuil=31, angle=45) 
im_corr_I2=correlation_mask_I(im_highpass,5,40, seuil=26, angle=135) #4,20, seuil=193, angle=135)

im_segmentation_F1= (im_corr_I1+im_corr_I2)
plt.imshow(im_segmentation_F1, cmap='gray')
plt.show()

#traitement fibre F5

fftc_highpass=highpass_filter(fibre_F5_inverted,Dc=5)
fft_highpass=np.fft.ifftshift(fftc_highpass)
invfft_highpass=np.real(np.fft.ifft2(fft_highpass))
plt.imshow(invfft_highpass, cmap='gray')
plt.show()

im_highpass=invfft_highpass 

im_corr_I1=correlation_mask_I(im_highpass,3,40, seuil=28, angle=45) 
im_corr_I2=correlation_mask_I(im_highpass,4,60, seuil=26, angle=135) 
im_segmentation_F5= (im_corr_I1+im_corr_I2)
plt.imshow(im_segmentation_F5, cmap='gray')
plt.show()

# traitement toute fibre
fftc_highpass=highpass_filter(zone_fibres_inverted,Dc=5)
fft_highpass=np.fft.ifftshift(fftc_highpass)
invfft_highpass=np.real(np.fft.ifft2(fft_highpass))
plt.imshow(invfft_highpass, cmap='gray')
plt.show()

im_highpass=invfft_highpass /10000
im_corr_I1=correlation_mask_I(im_highpass,3,40, seuil=5, angle=45) 

