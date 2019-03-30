# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 11:12:27 2019

@author: villa


"""
import random
import numpy as np
import skimage
from skimage.transform import rotate
from skimage.exposure import equalize_adapthist
from skimage import filters
import math
import scipy
from scipy import signal
from scipy import ndimage
from skimage.restoration import denoise_bilateral
import matplotlib.pyplot as plt
import math
import skimage.filters



#Fonctions génerales

def gradlap():
    kx=np.zeros((3,3))
    kx[1][0],kx[1][2]=1,-1
    ky=np.transpose(kx)
    klap=np.ones((3,3))
    klap[1][1]=-8
    return(kx,ky,klap)
    
def gaussianKernel(hs,sig):
    kernel=np.zeros((2*hs+1 ,2*hs+1))
    for n in range(2*hs+1):
        for p in range(2*hs+1):
            kernel[n][p]=math.exp(-((n-hs)**2+(p-hs)**2)/2/sig**2)
    
    return kernel / np.sum(kernel)

def rotation(im): #prend une image en argument, et la ressort pivotée dans le bon sens
    [n,p]=np.shape(im)
    im2=ndimage.convolve(im,gradlap()[2])
    if n<p :
        return(rotation(rotate(im,90)))
    else:
        if im2[:,2].all()==np.zeros((n,1)).all() :
            return(rotate(im,180))
        else:
            return(im)
    
def isoler(im,mask,X,Y) : #Renvoie la zone de l'image correspondant aux pixels entre les valeurs spécifiées par X et Y
    [px1,px2]=X    #(en proportions), utile pour tester les filtres seulement sur les zones d'interret
    [py1,py2]=Y
    [n,p]=np.shape(im)
    [x1,x2]=[int(n*px1),int(n*px2)]
    [y1,y2]=[int(p*py1),int(p*py2)]
    return(im[x1:x2,y1:y2],mask[x1:x2,y1:y2])
    
def linear (source, a, b): #entrée: image 2D, sortie: image ou les pixels d'intensité x prennent la valeur ax+b
    taille=np.shape(source)
    I=np.zeros_like(source)
    for k in range(taille[0]):
        for l in range(taille[1]):
            I[k][l]=a*source[k][l]+b
#            if I[k][l]>10000:
#                I[k][l]=10000
#            elif I[k][l]<0:
#                I[k][l]=0
    return(I)

def rgb2gray(rgb):
    '''
entrée: image RGB
sortie: Image niveau de gris
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def highpass_filter(im, Dc, option=True): #filtre passe haut de fréaquence de coupure Dc, gaussiien si option=true
    
    dim=np.shape(im)
    n=dim[0]
    p=dim[1]
    fft=np.fft.fft2(im)
    fftcenter=np.fft.fftshift(fft)
    
    for k in range(n):
        for l in range(p):
            if option:
                D2=(k-n/2)**2+(l-p/2)**2
                fftcenter[k,l]=fftcenter[k,l]*(1-np.exp(-D2/(2*Dc**2)))
                
            else:
                if (k-n/2)**2+(l-p/2)**2 < Dc**2:
                    fftcenter[k,l]=0+0*1j
    return(fftcenter)

def lowpass_filter(im, Dc, option=True): #filtre passe bas de fréaquence de coupure Dc, gaussiien si option=true
    
    dim=np.shape(im)
    n=dim[0]
    p=dim[1]
    fft=np.fft.fft2(im)
    fftcenter=np.fft.fftshift(fft)
    
    for k in range(n):
        for l in range(p):
            if option:
                D2=(k-n/2)**2+(l-p/2)**2
                fftcenter[k,l]=fftcenter[k,l]*np.exp(-D2/(2*Dc**2))
            else:
                if (k-n/2)**2+(l-p/2)**2 > Dc**2:
                    fftcenter[k,l]=0+0*1j
                
    return(fftcenter)
    
def meanKernel(hs):
    kernel = 1/np.power((2*hs+1),2) * np.ones([2*hs+1,2*hs+1])
    return kernel
 
def correlation_mask_I(im, Lx, Ly, angle=45, option=True):
    '''
        correlation de l'image (array) avec un mask en I de taille 2*Lx * 2*Ly orienté avec un angle de valeur angle
        sortie: tableau de booléen issu de la correlation seuillé
        option = True : seuil de yen, option=False: seuil d'Otsu
        
    '''
    
    mask=np.zeros_like(im) 
    [n,p]=np.shape(mask)
    mask[n//2-Lx:n//2+Lx,p//2-Ly:p//2+Ly]=np.ones([2*Lx,2*Ly])
    mask=rotate(mask, angle) 
    mask[0,0]=np.max(mask)*np.ones([1,1])
#    plt.figure()
#    plt.imshow(mask, cmap='gray')
#    plt.title("mask")
#    plt.show()
#    
    corr_mask=signal.correlate(im, mask, mode='same')

#    max_corr_mask=np.max(corr_mask)
#    min_corr_mask=np.min(corr_mask)
#    y, x = np.histogram(corr_mask, bins=np.arange(min_corr_mask,max_corr_mask))
#    fig, ax = plt.subplots()
#    plt.plot(x[:-1], y)
#    plt.show()

    skimage.filters.try_all_threshold(corr_mask)
    if option:
        seuil=skimage.filters.threshold_yen(corr_mask)
    else:
        seuil=skimage.filters.threshold_otsu(corr_mask)

    plt.figure()
    plt.imshow(corr_mask, cmap='gray')
    plt.title("image corrélée")
    plt.show()
    im_corr=corr_mask>seuil
#    plt.figure()
#    plt.imshow(im_corr, cmap='gray')
#    plt.show()
    
    return(im_corr)
    
    
def redim_im(im, im_mask):
    '''
    entrée: image du phantom et mask du phantom (array 2D)
    traitment: détection de la position du sein selon x et y
    sortie: image du phantom et mask coupé (array 2D) pour ne garder que la partie de l'image ou se trouve le sein (on enleve les bandes noires)
    
    '''
    
    [n,p]=np.shape(im)
    #détection de la posistion du sein selon y (vertical)
    y_haut=0
    mini=np.min(im)
    while im[y_haut,p-1]<(mini+500) and im[n//2,1]<(mini+500): #on ne prend pas les images retourner car la fonction rotation change le tableau
        y_haut+=1
    y_bas=1
    while im[n-y_bas,p-1]<(mini+500) and im[n//2,1]<(mini+500): #on ne prend pas les images retourner car la fonction rotation change le tableau
        y_bas+=1
    pos_y=[y_haut, y_bas]  # le sein estv de la ligne y_haut à y_bas
    
    #détection de la position du sein selon x (horrizontal)
    xg=0
    while im[n//2,xg]<(mini+500) and im[n//2,1]<(mini+500): #on ne prend pas les images retourner car la fonction rotation change le tableau
        xg+=1
    pos_x=[xg, p] # le sein est de la colonne x à taille[1]         
   
    im_red=im[ y_haut:(n-y_bas-1), xg:p-1]
    im_mask=im_mask[ y_haut:(n-y_bas-1), xg:p-1]
    
    taille=np.shape(im_red) 
    return (im_red, im_mask)

def redim_im_bis(im,mask):
    
    ims=skimage.filters.sobel(im)
    #plt.imshow(im)
    [n,p]=np.shape(ims)

    #détection de la posistion du sein selon x (vertical)
    xh=0
    while ims[xh,p-10]<1e-4:
        xh+=1
        
    xb=n-1
    while ims[xb,p-10]<1e-4 : 
        xb-=1

    im=im[xh:xb,:]
    mask=mask[xh:xb,:]
    ims=ims[xh:xb,:]
    #détection de la position du sein selon y (horrizontal)
    yg=0
    while ims[(xb-xh)//2,yg]<1e-4 :
        yg+=1
    yd=p-1
    while ims[(xb-xh)//2,yd-1]<1e-4 :
        yd-=1
    return (im[:,yg:yd],mask[:,yg:yd])


def resultat(im_segmentation, im_mask):
    '''
    entrée: image_segmentée et mask initial  (array 2D)
    traitement: calcul des TP, FP, TN,...
    sortie: dictionnaire measure qui contient les mesures classiques (dice, accuracy...)
    
    '''
    #analyse
    eps = 1e-12
    measures=dict()
    
    #nb de pixels = à 1 dans le mask
    measures["N_mask"]=len(np.where(im_mask==1)[1])     
    #nb de pixels = à 1 dans la segmentation
    measures["N_segm"]=len(np.where(im_segmentation==1)[1])
    
    
    TP=len(np.where(im_mask+im_segmentation==2)[1])
    FP=len(np.where(im_segmentation-im_mask==1)[1])
    TN=len(np.where(im_segmentation+im_mask==0)[1])
    FN=len(np.where(im_segmentation-im_mask==-1)[1])

    # Accuracy
    measures['dice'] = 2*TP/(2*TP+FN+FP)
    measures['accuracy'] = (TP+TN)/(TP+TN+FP+FN+eps)    
    # Precision
    measures['precision'] = TP/(TP+FP+eps)        
    # Specificity
    measures['specificity']=TN/(TN+FP+eps)    
    # Recall
    measures['recall'] = TP/(TP+FN+eps)   
    # F-measure
    measures['f1'] = 2*TP/(2*TP+FP+FN+eps)    
    # Negative Predictive Value
    measures['npv'] = TN/(TN+FN+eps)  
    # False Predictive Value
    measures['fpr'] = FP/(FP+TN+eps)


    print('\n',
          'Dice', measures['dice'], '\n',
          '\n',
          'nbpixels mask', measures['N_mask'], '\n',
          'nbpixels segmentation', measures['N_segm'], '\n',
          'Accuracy ', measures['accuracy'], '\n',
          'Precision', measures['precision'], '\n',
          'Recall', measures['recall'], '\n',
          'Specificity ', measures['specificity'], '\n',
          'F-measure', measures['f1'], '\n',
          'NPV', measures['npv'],'\n',
          'FPV', measures['fpr'],'\n'
          ) 
    return(measures)
    
    
def pipeline_segm_fibre(im,  im_mask, zone_fibre_n=[0.12,0.22], zone_fibre_p=[0.70,0.85]):
    '''entée =image array 2D, image du mask correspondant, délimitation zone à segmenter (adapté à une seule fibre)

traitement: segmente la fibres issue de la zone délimitée par zone_fibre_n (lignes) et zone_fibre_p (colonne)
pipeline :

-inversion de la valeur des pixels si nécessaire (fibres en blance, fond en noir)
-redimension
-isolement des fibres
-equalize adapthist
-bilinéaire
-filtrage passe haut pour enlever le gradient
-non local mean
-corrélation des deux mask
-OU logique

sortie: array correspondant à l'image segmentée, array du mask correspondant, dictionnaire measures (voir fct resultats)

    '''

    #test inversion
    [n,p]=np.shape(im)
    if np.mean(im[n//2-10:n//2+10 , 0:20]) > np.mean(im[n//2-10:n//2+10 , p-21:p-1]) :
        im=-im+np.max(im)
        
    #redimension
    [im_red, im_mask]=redim_im(im, im_mask)
    [n,p]=np.shape(im_red)

    #isolement fibre     
    [fibre, im_mask]=isoler(im_red,im_mask, zone_fibre_n, zone_fibre_p)           
    plt.figure()
    plt.imshow(fibre, cmap='gray')
    plt.title("Image originale")
    plt.show()
  
    # egalisation histogramme
    fibre=equalize_adapthist(fibre)
#    plt.figure()
#    plt.imshow(fibre, cmap='gray')
#    plt.title("egalisation d'histogramme")
#    plt.show()
      
    #bilineaire
    zone_bruit, m=isoler(fibre, np.zeros_like(fibre),[0,0.1], [0,0.1])
    moy_bruit =np.mean(zone_bruit)
    denoised = denoise_bilateral(fibre,win_size=3, sigma_color=moy_bruit, sigma_spatial=4, multichannel=False)
#    plt.figure()
#    plt.imshow(fibre, cmap='gray')
#    plt.title("filtrage bilinéaire")
#    plt.show() 
    
    #passe-haut
    fftc_highpass=highpass_filter(denoised,Dc=3)
    fft_highpass=np.fft.ifftshift(fftc_highpass)
    invfft_highpass=np.real(np.fft.ifft2(fft_highpass))
    
    #non local mean
    im_filtree=skimage.restoration.denoise_nl_means(invfft_highpass)
    plt.figure()
    plt.imshow(im_filtree, cmap='gray')
    plt.title("fin de filtrage")
    plt.show()
    
    #corrélation
    im_corr_I1=correlation_mask_I(im_filtree,4,40, angle=45) 
    im_corr_I2=correlation_mask_I(im_filtree,5,40, angle=135) #4,20, seuil=193, angle=135)
    im_segmentation= (im_corr_I1+im_corr_I2)
    im_segmentation=im_segmentation.astype('float32')
    
    plt.figure()
    plt.imshow(im_segmentation, cmap='gray')
    plt.title("segmentation")
    plt.show()
    
    [n,p]=np.shape(im_segmentation)
    comparaison=np.ones((n,p,3))
    comparaison[:,:,0]=im_mask
    comparaison[:,:,1]=im_segmentation
    
    plt.figure()
    plt.imshow(rgb2gray(comparaison))
    plt.title("comparaison")
    plt.show()
    
    mesures=resultat(im_segmentation, im_mask)
    return(im_segmentation, im_mask, mesures)



def pipeline_toute_fibre(im,  im_mask, option=True, zone_fibre_n=[0.11,0.42], zone_fibre_p=[0.295,0.82]):
    '''entée =image array 2D, image du mask correspondant, délimitation zone à segmenter (adapté à la zone des fibres entières)

traitement: segmente la zone des fibres issue de la zone délimitée par zone_fibre_n (lignes) et zone_fibre_p (colonne)

pipeline :

-inversion si nécessaire
-redimension
-isolement des fibres
-on brouille les endroits hors zones avec du bruit issue de la zone des fibres 
-equalize adapthist
-bilinéaire
-filtrage passe haut pour enlever le gradient
-non local mean
-corrélation des deux mask
-OU logique

sortie: array correspondant à l'image segmentée, array du mask correspondant, dictionnaire measures (voir fct resultats)


    '''
    

    
    
    #test inversion
    [n,p]=np.shape(im)
    if np.mean(im[n//2-10:n//2+10 , 0:20]) > np.mean(im[n//2-10:n//2+10 , p-21:p-1]) :
        im=-im+np.max(im)
        
    #redimensionnement 
    [im_red, im_mask]=redim_im(im, im_mask)
    [n,p]=np.shape(im_red)
    
    #isolation zone fibres
    [zone_fibres,im_mask]=isoler(im_red, im_mask, zone_fibre_n, zone_fibre_p) 
    
    plt.figure()
    plt.imshow(equalize_adapthist(zone_fibres), cmap="gray")
    plt.title("zone fibres")
    plt.show()
    
    
    #on brouille les endroits hors zones avec du bruit issue de la zone des fibres 
    [zone_bruit,m]=isoler(zone_fibres, np.zeros_like(zone_fibres),[0.25,0.75], [0.2,0.8])
    moy_bruit=np.mean(zone_bruit)
    zone_fibres_ravel=np.ravel(zone_bruit)
    [n,p]=np.shape(zone_fibres)
    print(n,p)
    for i in range(n):
        for j in range(p):
            if (-i) > ( -0.48*n + 0.92*j ) or (-i) < ( - 1.4*n  + 0.92*j ) :   #équations des deux droites qui délimitent la zone des fibres après l'isoaltion de la zone
                zone_fibres[i,j]=random.choice(zone_fibres_ravel)
             
                
                
    #égalisation d'histogramme            
    zone_fibres=equalize_adapthist(zone_fibres)
    

    #filtrage bilinéaire 
    denoised = denoise_bilateral(zone_fibres,win_size=3, sigma_color=moy_bruit, sigma_spatial=4, multichannel=False)
    
    #filtrage passe haut
    fftc_highpass=highpass_filter(denoised,Dc=10)
    fft_highpass=np.fft.ifftshift(fftc_highpass)
    invfft_highpass=np.real(np.fft.ifft2(fft_highpass))
        
    #on rajoute du bruit car le traitement réhausse la séparation entre le bruit que l'on a rajouté et la zone des fibres
    #on "baisse" la droite du haute et on "monte" la droite du bas pour réduire la zone des fibres: règle le problème de bord 
    [zone_bruit,m]=isoler(invfft_highpass, np.zeros_like(invfft_highpass),[0.5,0.6], [0.5,0.6])
    moy_bruit=np.mean(zone_bruit)
    zone_bruit_ravel=np.ravel(zone_bruit)
    [n,p]=np.shape(zone_fibres)
    for i in range(n):
        for j in range(p):
            if (-i) > ( -0.54*n + 0.92*j ) or (-i) < ( - 1.3*n  + 0.92*j ) : 
                invfft_highpass[i,j]=random.choice(zone_bruit_ravel)

    #filtre nl mean
    im_filtree=skimage.restoration.denoise_nl_means(invfft_highpass, patch_distance=2)
    
    #corrélation

    im_corr_I1=correlation_mask_I(im_filtree,3,50, 45, option) 
    im_corr_I2=correlation_mask_I(im_filtree,3,50, 135, option) 
        
    #résultat
    im_segmentation= (im_corr_I1+im_corr_I2)
    im_segmentation=im_segmentation.astype('float32')
    
    plt.figure()
    plt.imshow(im_segmentation)
    plt.title("segmentation fibres")
    plt.show()
    
    [n,p]=np.shape(im_segmentation)
    comparaison=np.ones((n,p,3))
    comparaison[:,:,0]=im_mask
    comparaison[:,:,1]=im_segmentation
       
    plt.figure()
    plt.imshow(rgb2gray(comparaison))
    plt.title("comparaison")
    plt.show()
    
    mesures=resultat(im_segmentation, im_mask)
    return(im_segmentation, im_mask, mesures)
