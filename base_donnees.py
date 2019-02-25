# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 19:16:05 2019

script indépendant pour analyser la base de données

-crée un dictionnaire pour tous les constructeurs
-chaque dictionnaire, contient toutes les images de la base de données (pixel array), la date de création, et sa taille, et le taille du sein en x et y (en nombre de pixel)


@author: Gauthier Frecon
"""

plt.close ('all')
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
d_planmed['pos_y']=[] ##position de la position du sein selon y, le sein est entre les lignes de pos_y
d_planmed['pos_x']=[] #position de la position du sein selon x, le sein est de la colonne x à taille[1]  
d_planmed['moyenne_mat_blanc']=[] #moyenne des pixels graisse et glande
d_planmed['moyenne_mat_noir']=[]
for root, dirnames, filenames in os.walk(IMDIR_planmed):
    for filename in filenames:
        f = os.path.join(root, filename)
        #filter only image files with the following format
        if f.endswith(('.dcm')) and not root.endswith('__MACOSX'):  
            ds = pydicom.dcmread(f)
            im=ds.pixel_array
            im_red=redim_im(im) 
            d_planmed['im'].append( im_red)      
            [n,p]=np.shape(im_red)
            d_planmed['taille'].append([n,p])
            d_planmed['date'].append(ds.ContentDate)  

            
            #calcul valeurs moyennes glande et graisse
            #partie pour trouver les carrés blancs et noi, peut-etre à modifier en pondérant par les tailles des images. 
            
            #m=(pos_y[0]+n-pos_y[1])//2
#            x_bas=pos_x[0]+525
#            x_haut=pos_x[0]+575
#            y_matblanc_bas=m-30
#            y_matblanc_haut=m+30
#            y_matnoir_bas=m+125-30
#            y_matnoir_haut=m+125+30
            
            #a faire fonction redimension et calcul des zones par proportion
            
            m=n//2
            x_bas=525
            x_haut=575
            y_matblanc_bas=m-30
            y_matblanc_haut=m+30
            y_matnoir_bas=m+125-30
            y_matnoir_haut=m+125+30
            
            
            
            moyenne_matblanc=np.mean(im_red[y_matblanc_bas: y_matblanc_haut, x_bas:x_haut])
            moyenne_matnoir=np.mean(im_red[y_matnoir_bas: y_matnoir_haut, x_bas:x_haut])              
            d_planmed['moyenne_mat_blanc'].append(moyenne_matblanc)
            d_planmed['moyenne_mat_noir'].append(moyenne_matnoir)
            plt.imshow(im_red[y_matblanc_bas: y_matblanc_haut, x_bas:x_haut], cmap='gray')
            plt.figure()
            plt.imshow(im_red, cmap='gray')
            plt.plot([0,p], [m, m])
            plt.plot([x_bas, x_bas,x_haut, x_haut, x_bas], [y_matblanc_bas, y_matblanc_haut, y_matblanc_haut, y_matblanc_bas,y_matblanc_bas])
            plt.plot([x_bas, x_bas,x_haut, x_haut, x_bas], [y_matnoir_bas, y_matnoir_haut, y_matnoir_haut, y_matnoir_bas, y_matnoir_bas])
            plt.show()


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
