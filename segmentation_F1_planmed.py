# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 22:01:38 2019


On parcourt les images planmed issue du dictionnaire d_planmed et on segmente la fibre 1, on obtient des résultats seulement pour les im de même taille

@author: Gauthier Frecon



à faire: 
    
-recadrer les zones en fonction de Lx et Ly
-fair une étude détaillée sur une iamge de ref (à moitié déja fait)
-calculer pr chaque image les valeurs moyenne de la glande et de la graisse (àmoitié ok)
-seuil en fonctio du tiret au dessus
-voir normalisation gamma


-mettre au propre redim




"""
import skimage

plt.close('all')

for i in range(12):#len(d_planmed['im'])):
    
    
    #im=d_planmed['im'][i]
    
    pipeline_segm_fibre(d_planmed['im'][i])
    
    '''
    
    fibre_F1=isoler(im, [0.11,0.22], [0.70,0.87])#partie de l'image correspondant à la fibre F1
    #plt.imshow(zone_fibres_F1, cmap='gray')
    
    
    fibre_F1_inverted=linear(fibre_F1, -1, 10000)
    fibre_F1_inverted=equalize_adapthist(fibre_F1_inverted) #fibre F1 avec pixels inversé (blanc=noir et inversement)    
    fftc_highpass=highpass_filter(fibre_F1_inverted,Dc=5)
    fft_highpass=np.fft.ifftshift(fftc_highpass)
    invfft_highpass=np.real(np.fft.ifft2(fft_highpass))
    im_highpass=invfft_highpass 
    im_corr_I1=correlation_mask_I(im_highpass,4,40, seuil=31, angle=45)  
    im_corr_I2=correlation_mask_I(im_highpass,5,40, seuil=26, angle=135) #4,20, seuil=193, angle=135)
    
    im_segmentation_F1= (im_corr_I1+im_corr_I2)
    plt.figure(i)
    plt.imshow(im_segmentation_F1, cmap='gray')
    plt.show()


'''