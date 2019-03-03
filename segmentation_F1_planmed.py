# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 22:01:38 2019


On parcourt les images planmed issue du dictionnaire d_planmed et on segmente la fibre 1, on obtient des résultats seulement pour les im de même taille

@author: Gauthier Frecon








"""


plt.close('all')

for i in range(3):#len(d_planmed['im'])):
    
    
 
    #plt.imshow(d_planmed['im'][i])
    pipeline_segm_fibre(d_planmed['im'][i])
    
   