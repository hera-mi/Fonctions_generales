# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 11:12:27 2019

@author: villa
"""
from scipy import ndimage
import numpy as np
from skimage.transform import rotate
import math
from skimage import filters


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
    
def isoler(im,X,Y) : #Renvoie la zone de l'image correspondant aux pixels entre les valeurs spécifiées par X et Y
    [px1,px2]=X    #(en proportions), utile pour tester les filtres seulement sur les zones d'interret
    [py1,py2]=Y
    [n,p]=np.shape(im)
    [x1,x2]=[int(n*px1),int(n*px2)]
    [y1,y2]=[int(p*py1),int(p*py2)]
    return(im[x1:x2,y1:y2])
    
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
 
def correlation_mask_I(im, Lx, Ly, seuil, angle=45):
    ''' correlation de l'image avec un mask en I de taille 2*Lx   * 2*Ly puis seuillage à seuil'''
    
    mask=np.zeros_like(im) #prendre la shape et enlever les 122
    [n,p]=np.shape(mask)
    mask[n//2-Lx:n//2+Lx,p//2-Ly:p//2+Ly]=np.ones([2*Lx,2*Ly])
    mask=rotate(mask, angle)
    mask[0,0]=np.max(mask)*np.ones([1,1])
#    plt.figure(1)
#    plt.imshow(mask, cmap='gray')
#    plt.title("mask")
#    plt.show()
    corr_mask=signal.correlate(im, mask, mode='same')
    max_corr_mask=np.max(corr_mask)
    min_corr_mask=np.min(corr_mask)
#    y, x = np.histogram(corr_mask, bins=np.arange(min_corr_mask,max_corr_mask))
#    fig, ax = plt.subplots()
#    plt.plot(x[:-1], y)
#    plt.show()
#    
    skimage.filters.try_all_threshold(corr_mask)
#    plt.figure(3)
#    plt.imshow(corr_mask, cmap='gray')
#    plt.show()
    im_corr=corr_mask>seuil
#    plt.figure(2)
#    plt.imshow(im_corr, cmap='gray')
#    plt.show()
    return(im_corr)
    
    
def redim_im(im):
    
    [n,p]=np.shape(im)
    #détection de la posistion du sein selon y (vertical)
    y_haut=0
    maxi=np.max(im)
    while im[y_haut,p-1]>(maxi-500) and im[n//2,1]>(maxi-500): #on ne prend pas les images retourner car la fonction rotation change le tableau
        y_haut+=1
    y_bas=1
    while im[n-y_bas,p-1]>(maxi-500) and im[n//2,1]>(maxi-500): #on ne prend pas les images retourner car la fonction rotation change le tableau
        y_bas+=1
    pos_y=[y_haut, y_bas]  # le sein estv de la ligne y_haut à y_bas
    
    #détection de la position du sein selon x (horrizontal)
    x=0
    while im[n//2,x]>(maxi-500) and im[n//2,1]>(maxi-500): #on ne prend pas les images retourner car la fonction rotation change le tableau
        x+=1
    pos_x=[x, p] # le sein est de la colonne x à taille[1]         
   
    im_red=im[ y_haut:(n-y_bas-1),x:p-1]
    
    
    taille=np.shape(im_red) 
    return (im_red)

def redim_im_bis(im):
    
    ims=skimage.filters.sobel(im)
    [n,p]=np.shape(ims)

    #détection de la posistion du sein selon x (vertical)
    xh=0
    while ims[xh,p-10]<1e-4:
        xh+=1
        
    xb=n-1
    while ims[xb,p-10]<1e-4 : 
        xb-=1

    im=im[xh:xb,:]
    ims=ims[xh:xb,:]
    #détection de la position du sein selon y (horrizontal)
    yg=0
    while ims[(xb-xh)//2,yg]<1e-4 :
        yg+=1
    yd=p-1
    while ims[(xb-xh)//2,yd-1]<1e-4 :
        yd-=1
    return (im[:,yg:yd])


def pipeline_segm_fibre(im, zone_fibre_n=[0.12,0.22], zone_fibre_p=[0.70,0.85], seuil1=28, seuil2=30):
    '''segmente la fibres de l'im 
     
    faire une correlation plus propre en prenant les moyennes et en gérant la variance
    '''

    #redim
    im_red=redim_im(im)
    [n,p]=np.shape(im_red)
    
    #isolement fibre f1
     
    fibre=isoler(im_red, zone_fibre_n, zone_fibre_p)           
    plt.figure()
    plt.imshow(fibre, cmap='gray')
    plt.show()
    
    # traitement fibre 
    
    fibre_inverted=linear(fibre, -1, 10000)
    fibre_inverted=equalize_adapthist(fibre_inverted)
    fftc_highpass=highpass_filter(fibre_inverted,Dc=5)
    fft_highpass=np.fft.ifftshift(fftc_highpass)
    invfft_highpass=np.real(np.fft.ifft2(fft_highpass))
    im_highpass=invfft_highpass 
    invfft_highpass=scipy.signal.medfilt(skimage.restoration.denoise_nl_means(invfft_highpass)) 

    #corrélation
    im_corr_I1=correlation_mask_I(im_highpass,4,40, seuil=seuil1, angle=45) 
    im_corr_I2=correlation_mask_I(im_highpass,5,40, seuil=seuil2, angle=135) #4,20, seuil=193, angle=135)
    im_segmentation= (im_corr_I1+im_corr_I2)
    plt.figure()
    plt.imshow(im_segmentation, cmap='gray')
    plt.show()
    
    return(im_segmentation)


