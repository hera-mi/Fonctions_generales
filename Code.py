# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:07:44 2018

@author: villa
"""

import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import os
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage.transform import resize
import random
import scipy.stats
import skimage.morphology.selem
from math import sqrt
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from sklearn.cluster import KMeans
from skimage import data
from skimage import filters
from skimage import exposure
from skimage.transform import rotate
from skimage.segmentation import active_contour
import math
from skimage.restoration import denoise_nl_means, estimate_sigma
#from Filtres import *
import cv2
from skimage import img_as_ubyte

from Fct_generales import *

import scipy.signal
from skimage.filters import threshold_otsu, threshold_adaptive






IMDIR="D:/Documents/Travail/DATASIm/Projet/datasim-prj-phantoms-dataset-201812061411/digital/2.16.840.1.113669.632.20.20110707.94917643.9.2/"
#filename = get_testdata_files(IMDIR+"2.16.840.1.113669.632.20.20110707.95031041.200083.50.dcm")[0]
ds = pydicom.dcmread(IMDIR+"2.16.840.1.113669.632.20.20110707.95031041.200083.50.dcm")
#plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
im= ds.pixel_array
#plt.imshow(im, cmap="hot")
#
#im2=ndimage.convolve(im,gradlap()[2])
#im3=rotate(im2,180)
#plt.imshow(im3)


    
masses=isoler(rotation(im),[0.35,0.43],[0.01,0.3])  
masses_l=ndimage.convolve(masses,gaussianKernel(10,7)) #Sinon utiliser scipy.ndimage.filters.gaussian_filter
masses_ll=ndimage.convolve(masses_l,gaussianKernel(10,7))
masses_c=ndimage.convolve(masses,gradlap()[2]) 
#masses_co=otsu(masses_c).astype(int)
#masses_lo=otsu(masses_l).astype(int)

    
s = np.linspace(0, 2*np.pi, 400)
x = 280 + 50*np.cos(s)
y = 500 + 50*np.sin(s)
init = np.array([x, y]).T
"""
snake = active_contour(massesbis_l,init, alpha=0.015, beta=10, gamma=0.001)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(massesbis_l, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, masses.shape[1], masses.shape[0], 0])
"""
from skimage import img_as_ubyte
cv_image = img_as_ubyte(masses)
#np.uint8(masses)
thresh=cv2.adaptiveThreshold(cv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1)
#https://pythonprogramming.net/thresholding-image-analysis-python-opencv-tutorial/

def trouver_masses(im):
    cv_im = img_as_ubyte(im)
    #plt.imshow(cv_im)
    #imgray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY) 
    im_gauss = cv2.GaussianBlur(cv_im, (5, 5), 0) 
    thresh = cv2.adaptiveThreshold(im_gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1)
# get contours 
    #Yplt.imshow(thresh)
    imb, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(imb, cmap=plt.cm.gray)
    for i in contours:
        if np.shape(i)[0]>100 : #on ne garde que les formes significatives
            ax.plot(i[:,0 ,0], i[:,0, 1], '-b', lw=3)
    return(imb,contours)
#https://stackoverrun.com/fr/q/11610670    
    
    
dirbis="D:/Documents/Travail/DATASIm/Projet/"
imbis=io.imread(dirbis+"2.16.840.1.113669.632.20.20120425.94850094.200083.40.dcm.jpg")
massesbis=isoler(rotation(imbis),[0.3,0.45],[0.01,0.36]) 

massesbis_l=ndimage.convolve(massesbis,gaussianKernel(10,7))
"""
s = np.linspace(0, 2*np.pi, 400)
xb = 180 + 50*np.cos(s)
yb = 230 + 50*np.sin(s)
initb = np.array([xb, yb]).T

snake = active_contour(skimage.util.invert(i),initb, alpha=0.015, beta=10, gamma=0.001)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(i, cmap=plt.cm.gray)
ax.plot(initb[:, 0], initb[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, i.shape[1], i.shape[0], 0])
"""
#plt.plot(c[501][:,0,0],c[501][:,0,1])
#skimage.measure.find_contours(massesbis,100,fully_connected='low', positive_orientation='high')
#
def moyenne(im,pos,pas):#effectue la moyenn autour d'une position (peut sûrement être remplacer par de la convolution locale)
    m=0
    [x,y]=pos
    for n in range(2*pas+1):
        for p in range(2*pas+1):
            m+=im[x+n-pas,y+p-pas]
    m=m/(2*pas+1)**2
    return(m)
    

def trouver_zone(im,pas,valeur,sig=1): #peut être faut-il differencier le pas de la taille du noyau
   # filtre=gaussianKernel(pas,sig)
    [n,p]=np.shape(im)
    print(n,p)
    x,y=pas+1,pas+1
    lieux_x,lieux_y=[],[]
    while x< (n-pas):
        y=0
        while y<(p-pas):
            if moyenne(im,[x,y],pas)>(valeur):#/256): #attention à vérifier si les valeurs vont de 0 à1 ou de 0 à 255
                lieux_x+=[x]
                lieux_y+=[y]
            y+=pas
        x+=pas
    fig, ax = plt.subplots()#figsize=(7, 7))
    ax.imshow(im, cmap=plt.cm.gray)
    ax.plot(lieux_y, lieux_x, 'bo')#, lw=3)#marche pas car lieux est une liste, pas un array
    return(lieux_x,lieux_y) #y correspond aux colonnes et x aux lignes
      
#trouver_zone(massesbis,10,90)      
            
""" Pour améliorer le résultat de "trouver_zone", essayer d'appliqer le filtrage adaptatif proposé
dans le pdf.
Peut être qu'un filtre médian marcherait bien, puisqu'on a l'impression que le bruit est poivre et sel.
Peut être qu'extraire le fond pourrait être efficace aussi.
Il faudra étudier une chaîne de traitements qui donnent de bons résultats une fois la fonction codée."""

def contrast_local(imtot,pos,N,F=lambda x:sqrt(x)):
    #Cette fonction calcule le niveau de gris d'un pixel situé en position pos de imtot en lui appliquant 
    #un réhaussement de contraste local
    
    #im est centrée autour de la position dont cette fonction calcule le contraste 
    #On aurait pu penser la fonction en donnant l'image totale et une position et une taille de fenetre
    #mais la on donne directement la petite partie de la taille de la fenetre autour de la position
    #Mettre une petite fenetre si on veut filtrer les  HF, et une grande fenetre si on veut filtrer les BF
    
    #F est la fonction de transformation du contraste
    T=imtot.std()
    [x,y]=pos
    [Tx,Ty]=np.shape(imtot)
    assert (x-N>=0 and y-N>=0 and x+N<Tx and y+N<Ty) is True, "la fenetre sort de l'image"
    #assert N%2==1 is True, "La taille de l'image doit être impaire"
    im=imtot[x-N:x+N+1,y-N:y+N+1]
    A=np.ones([2*N+1,2*N+1])
    for n in range(2*N+1):
        for p in range(2*N+1):
            A[n,p]=(abs(im[n,p]-im[N,N])<=T)
    d=2 #décrement de diametre
    c=2*N+1-d
    p=0 #proportion de 1
    n=0 #Nombre de cases parcourues
    for i in range(d-2,2*N+1-(d-2)): #On parcourt les 4 lignes
        p+=sum(A[d-2:d,i])+sum(A[2*N-d:2*N-(d-2),i])
        n+=6*(2*N+1-2*d+4)
    for i in range(d+1,2*N+1-d-1): #♦puis les 4 colonnes
        p+=sum(A[i,d-2:d])+sum(A[i,2*N-d:2*N-(d-2)])
        n+=6*(2*N+1-2*d)
    p=p/n
    while p>0.4 or d<N-4: #Proportion de 0<60% + condition d'arret si ça n'arrive jamais
        d+=1
        #c=2*N+1-2*d
        p=0 #proportion de 1
        n=0
        for i in range(d-2,2*N+1-(d-2)): #On parcourt les 4 lignes
            p+=sum(A[d-2:d,i])+sum(A[2*N-d:2*N-(d-2),i])
            n+=6*(2*N+1-2*d+4)
        for i in range(d+1,2*N+1-d-1): #♦puis les 4 colonnes
            p+=sum(A[i,d-2:d])+sum(A[i,2*N-d:2*N-(d-2)])
            n+=6*(2*N+1-2*d)
        p=p/n
    imbis=im[d-2:2*N-(d-2),d-2:2*N-(d-2)]
    Abis=A[d-2:2*N-(d-2),d-2:2*N-(d-2)]
    #♠print(Abis)
    Apadd=np.zeros([2*N-2*(d-2)+2,2*N-2*(d-2)+2]) #On 0-padd afin d'éviter les effets de bord lors du calcul des régions
    Apadd[1:2*N-2*(d-2)+1,1:2*N-2*(d-2)+1]=Abis
    #print(Apadd)
    Mc,Mb=0,0 #moyennes pour chaque zone
    cc,cb=0,0 #compteurs pour chaque zone
    for i in range(2*N-2*(d-2)):
        for j in range(2*N-2*(d-2)):
            if Apadd[i+1,j+1]==1:
                Mc+=imbis[i,j]
                cc+=1
            elif (Apadd[i,j]+Apadd[i+1,j]+Apadd[i+2,j]+Apadd[i+2,j+1]+Apadd[i+2,j+2]+Apadd[i+1,j+2]+Apadd[i,j+2]+Apadd[i+1,j])!=0:
                Mb+=imbis[i,j]
                cb+=1
    if Mc!=0:
        Mc=Mc/cc
    if Mb!=0:
        Mb=Mb/cb
    Cont=abs(Mb-Mc)/max(Mb,Mc)
    Cont=F(Cont)
    if Mb<Mc:
        if Mb!=0:
            E=Mb/(1-Cont)
        else:
            E=0
    else :
        E=Mb*(1-Cont)
    return(E)
    
    
def rehausse(im,N):
    [n,p]=np.shape(im)
    imbis=np.copy(im)
    
    #Soit on peut padder l'image, soit commencer pas tout à fait au bord, pour prendre en compte la fenetre
    #je choisis de ne pas commencer tout à fait au bord
    for i in range(N,n-N):
        for j in range(N,p-N):
            imbis[i,j]=contrast_local(im,[i,j],N)
    return(imbis)
                    
    
#Tenter non local means
   
patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)
sigma_est = np.mean(estimate_sigma(masses, multichannel=True))
denoise = denoise_nl_means(masses, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)

#denoise = denoise_nl_means(massesbis, h=1.15 * sigma_est, fast_mode=False,**patch_kw)
#masses_cont=skimage.exposure.rescale_intensity(denoise)
masses_cont=skimage.exposure.equalize_hist(masses)
denoiseb = denoise_nl_means(masses_cont, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
denoisemed=scipy.signal.medfilt(masses_cont,7)
d=scipy.signal.medfilt(masses,7)

block_size = 9
binary_adaptive = threshold_adaptive(d, block_size, offset=10)

global_thresh = threshold_otsu(d)
binary_global = d > global_thresh

masses_contloc=skimage.exposure.equalize_adapthist(masses)

def pipeline1(im,block_size = 8,sig=im.std()):
    #im=ndimage.convolve(masses,gradlap()[2])
    im_contloc=skimage.exposure.equalize_adapthist(im)
    im=ndimage.convolve(im_contloc,gaussianKernel(block_size,sig))
    #im=skimage.filters.gaussian(im)
    #im_contloc=skimage.exposure.equalize_adapthist(im)
    im_med=scipy.signal.medfilt(im,7)
    im_contloc=skimage.exposure.equalize_adapthist(im_med)
    im_eq=skimage.exposure.equalize_hist(im_contloc)
    #im=skimage.filters.sobel(im_eq)-im_eq  #Ca marche pas mal en ajoutant ca
    im_nl = denoise_nl_means(im_eq, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
    im_med=scipy.signal.medfilt(im_nl,7)
    im=skimage.filters.gaussian(im_med)
    im=scipy.signal.medfilt(im,7)
    #im_thresh = threshold_adaptive(im_nl, block_size, offset=0)
    #global_thresh = threshold_otsu(im_nl)
    #im_thresh = im_nl > global_thresh
    #plt.imshow(im_thresh)
    #plt.imshow(im>1-im.std()*3.1,cmap='gray')
    #◘trouver_zone(skimage.util.invert(i),10,1-i.std())
    t=skimage.filters.threshold_minimum(im)
    imt=(im>t).astype(int)
    
    return(imt)
    #plt.imshow(imt)
    #skimage.filters.try_all_threshold(im)
    #plt.show()

def pipeline2(im,block_size = 8,sig=8):
    #im=ndimage.convolve(masses,gradlap()[2])
    #im=im-ndimage.convolve(im,meanKernel(101))
    #im_contloc=skimage.exposure.equalize_adapthist(im)
    #im_med=scipy.signal.medfilt(im,7)
    im=skimage.filters.sobel(im)-im
    im=ndimage.convolve(im,gaussianKernel(block_size,sig))
    #im=skimage.filters.gaussian(im)
    #im_contloc=skimage.exposure.equalize_adapthist(im)
    #hp=skimage.filters.laplace(im,10)
    im_med=scipy.signal.medfilt(im,7)
    #im_med=ndimage.convolve(im_med,gaussianKernel(block_size,sig))
    #im_contloc=skimage.exposure.equalize_adapthist(im_med)  #ca marche pas dans ce sens
    im_eq=skimage.exposure.equalize_hist(im_med)
    #hp=skimage.filters.laplace(im_eq,10)
    im_nl = denoise_nl_means(im_eq, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
    im_med=scipy.signal.medfilt(im_nl,7)
    im=skimage.filters.gaussian(im_med)
    #im=skimage.exposure.equalize_hist(im)
    #hp=skimage.filters.laplace(im,10)
    
    #im_thresh = threshold_adaptive(im_nl, block_size, offset=0)
    #global_thresh = threshold_otsu(im_nl)
    #im_thresh = im_nl > global_thresh
    #plt.imshow(im_thresh)
    #plt.imshow(im>1-im.std()*3.1,cmap='gray')
    #trouver_zone(skimage.util.invert(i),10,1-i.std())

    t=skimage.filters.threshold_minimum(im)
    imt=(im>t).astype(int)
    #plt.imshow(imt)
    #skimage.filters.try_all_threshold(im)
    #plt.show()
    return(imt)
    
def pipeline3(im,block_size = 8,sig=im.std()):
    #im=ndimage.convolve(masses,gradlap()[2])
    im_contloc=skimage.exposure.equalize_adapthist(im)
    im=ndimage.convolve(im_contloc,gaussianKernel(block_size,sig))
    #im=skimage.filters.gaussian(im)
    #im_contloc=skimage.exposure.equalize_adapthist(im)
    im_med=scipy.signal.medfilt(im,7)
    im_contloc=skimage.exposure.equalize_adapthist(im_med)
    im_eq=skimage.exposure.equalize_hist(im_contloc)
    im=skimage.filters.sobel(im_eq)-im_eq  #Ca marche pas mal en ajoutant ca
    im_nl = denoise_nl_means(im, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
    im_med=scipy.signal.medfilt(im_nl,7)
    im=skimage.filters.gaussian(im_med)
    im=scipy.signal.medfilt(im,7)
    #im_thresh = threshold_adaptive(im_nl, block_size, offset=0)
    #global_thresh = threshold_otsu(im_nl)
    #im_thresh = im_nl > global_thresh
    #plt.imshow(im_thresh)
    #plt.imshow(im>1-im.std()*3.1,cmap='gray')
    #◘trouver_zone(skimage.util.invert(i),10,1-i.std())
    t=skimage.filters.threshold_minimum(im)
    imt=(im>t).astype(int)
    
    return(imt)
    
    
  

#Essayer de retirer certains filtres afin d'avoir moins de paramètres à regler, et d'être donc moins dépendant de l'image
    
DIR="D:/Documents/Travail/DATASIm/Projet/Fonctions_generales/Masses test/"
DIRBIS='./Masses test/'
def detection(DIR):
    d={}
    d[0]="2.16.840.1.113669.632.20.20140513.202406491.200064.424.dcm"
    d[1]="ge-0001-0000-00000000.dcm"
    d[2]="hologic-MG02.dcm"
    d[3]="1.2.392.200036.9125.4.0.2718896371.50333032.466243176.dcm"
    IM=[]
    for i in d:
        ds = pydicom.dcmread(DIR+d[i],force=True)
        IM+= [rotation(ds.pixel_array)]
    for i in IM:
        i=pipeline1(i)
    return(IM)
  
im1=pydicom.dcmread(DIR+"2.16.840.1.113669.632.20.20140513.202406491.200064.424.dcm")
#im1=rotation(im1.pixel_array)
#masses1=isoler(im1,[0.37,0.46],[0.05,0.29])    
masses1=isoler(redim_im_bis(im1.pixel_array),[0.603,0.722],[0.436,0.9])

im2=pydicom.dcmread(DIR+"hologic-MG02.dcm")
#im2=rotation(im2.pixel_array)
#masses2=isoler(im2,[0.355,0.445],[0.05,0.29])
masses2=isoler(redim_im_bis(im2.pixel_array),[0.603,0.722],[0.436,0.9])


im3=pydicom.dcmread(DIR+"ge-0001-0000-00000000.dcm")
#im3=rotation(im3.pixel_array)
#masses3=isoler(im3,[0.32,0.4021],[0.05,0.29])
masses3=isoler(redim_im_bis(im3.pixel_array),[0.603,0.722],[0.436,0.9])

im4=pydicom.dcmread(DIR+"1.2.392.200036.9125.4.0.2718896371.50333032.466243176.dcm")
#im4=rotation(im4.pixel_array)
#masses4=isoler(im4,[0.35,0.44],[0.05,0.29])
masses4=isoler(redim_im_bis(im4.pixel_array),[0.603,0.722],[0.436,0.9])

plt.figure(1)
plt.subplot(4,2,1)
plt.imshow(masses1)
plt.title("PlanMed")
plt.subplot(4,2,2)
plt.imshow(pipeline1(masses1))
plt.subplot(4,2,3)
plt.imshow(masses2)
plt.title("Hologic")
plt.subplot(4,2,4)
plt.imshow(pipeline1(masses2))
plt.subplot(4,2,5)
plt.imshow(masses3)
plt.title("GE")
plt.subplot(4,2,6)
plt.imshow(pipeline1(masses3))
plt.subplot(4,2,7)
plt.imshow(masses4)
plt.title("Fuji")
plt.subplot(4,2,8)
plt.imshow(pipeline1(masses4))

plt.figure(2)
plt.subplot(4,2,1)
plt.imshow(masses1)
plt.title("PlanMed")
plt.subplot(4,2,2)
plt.imshow(pipeline2(masses1))
plt.subplot(4,2,3)
plt.imshow(masses2)
plt.title("Hologic")
plt.subplot(4,2,4)
plt.imshow(pipeline2(masses2))
plt.subplot(4,2,5)
plt.imshow(masses3)
plt.title("GE")
plt.subplot(4,2,6)
plt.imshow(pipeline2(masses3))
plt.subplot(4,2,7)
plt.imshow(masses4)
plt.title("Fuji")
plt.subplot(4,2,8)
plt.imshow(pipeline2(masses4))

plt.figure(3)
plt.subplot(4,2,1)
plt.imshow(masses1)
plt.title("PlanMed")
plt.subplot(4,2,2)
plt.imshow(pipeline3(masses1))
plt.subplot(4,2,3)
plt.imshow(masses2)
plt.title("Hologic")
plt.subplot(4,2,4)
plt.imshow(pipeline3(masses2))
plt.subplot(4,2,5)
plt.imshow(masses3)
plt.title("GE")
plt.subplot(4,2,6)
plt.imshow(pipeline3(masses3))
plt.subplot(4,2,7)
plt.imshow(masses4)
plt.title("Fuji")
plt.subplot(4,2,8)
plt.imshow(pipeline3(masses4))


hp=skimage.filters.laplace(masses,10)