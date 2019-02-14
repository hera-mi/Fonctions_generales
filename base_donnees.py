# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 19:16:05 2019

script indépendant pour analyser la base de données

-crée un dictionnaire pour tous les constructeurs
-chaque dictionnaire, contient toutes les images de la base de données (pixel array), la date de création, et sa taille, et le taille du sein en x et y (en nombre de pixel)


@author: Gauthier Frecon
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
from scipy import ndimage
import math
IMDIR_fuji=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-fujifilm-dataset-201812130858\datasim-prj-phantoms-fuji-dataset-201812130858"
IMDIR_planmed=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-planmed-dataset-201812061411\datasim-prj-phantoms-dataset-201812061411\digital"
IMDIR_ge=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-ge-20190125-mg-proc"
IMDIR_hologic=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-hologic-20190125-mg-proc"


d_planmed={}
d_planmed['im']=[]
d_planmed['taille']=[] #taille du pixel array
d_planmed['date']=[] #date de création
d_planmed['Lx']=[] #taille du sein selon x
d_planmed['Ly']=[] #taille du sein selon y
for root, dirnames, filenames in os.walk(IMDIR_planmed):
    for filename in filenames:
        f = os.path.join(root, filename)
        #filter only image files with the following format
        if f.endswith(('.dcm')) and not root.endswith('__MACOSX'):  
            ds = pydicom.dcmread(f)
            im=ds.pixel_array
            [n,p]=np.shape(im)
            d_planmed['im'].append(im)
            taille=np.shape(ds.pixel_array)
            d_planmed['taille'].append(taille)
            d_planmed['date'].append(ds.ContentDate)
            #détection de la taile du sein selon x (vertical)
            x_haut=0
            while im[x_haut,taille[1]-1]==10000 and im[n//2,1]==10000: #on ne prend pas les images retourner car la fonction rotation change le tableau
                x_haut+=1
            x_bas=1
            while im[taille[0]-x_bas,taille[1]-1]==10000 and im[n//2,1]==1000: #on ne prend pas les images retourner car la fonction rotation change le tableau
                x_bas+=1
            Lx=taille[0]-x_bas-x_haut
            d_planmed['Lx'].append(Lx)
            
            #détection de la taile du sein selon x (vertical)
            y=0
            while im[n//2,y]==10000 and im[n//2,1]==10000: #on ne prend pas les images retourner car la fonction rotation change le tableau
                y+=1

            Ly=taille[1]-y
            d_planmed['Ly'].append(Ly)            
                
                



#d_fuji={}
#d_fuji['im']=[]
#d_fuji['taille']=[]
#d_fuji['date']=[]
#for root, dirnames, filenames in os.walk(IMDIR_fuji):
#    for filename in filenames:
#        f = os.path.join(root, filename)        
#        #filter only image files with the following format
#        if f.endswith(('.dcm')) and not root.endswith('__MACOSX'):         
#            ds = pydicom.dcmread(f)
#            d_fuji['im'].append(ds.pixel_array)
#            d_fuji['taille'].append(np.shape(ds.pixel_array))
#            d_fuji['date'].append(ds.ContentDate)
#d_hologic={}
#d_hologic['im']=[]
#d_hologic['taille']=[]
#d_hologic['date']=[]
#for root, dirnames, filenames in os.walk(IMDIR_hologic):
#
#    for filename in filenames:
#        f = os.path.join(root, filename)
#        #filter only image files with the following format
#        if f.endswith(('.dcm')) and not root.endswith('__MACOSX'):
#            ds = pydicom.dcmread(f)
#            d_hologic['im'].append(ds.pixel_array)
#            d_hologic['taille'].append(np.shape(ds.pixel_array))
#            d_hologic['date'].append(ds.ContentDate)
#          
#
#
#            
#d_ge={}
#d_ge['im']=[]
#d_ge['taille']=[]
#d_ge['date']=[]
#for root, dirnames, filenames in os.walk(IMDIR_ge):
#    for filename in filenames:
#        f = os.path.join(root, filename)
#        #filter only image files with the following format
#        if f.endswith(('.dcm')) and not root.endswith('__MACOSX'):     
#            ds = pydicom.dcmread(f)
#            d_ge['im'].append(ds.pixel_array)
#            d_ge['taille'].append(np.shape(ds.pixel_array))
#            d_ge['date'].append(ds.ContentDate)
#            
#                   
