# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:37:08 2019

@author: Gauthier Frecon

Méthode ICP (iterative closest point)  pour cadrer une image (model) sur une autre (data)

-compléter :IMDIR/chemin 1 et IMDIR/chemin2 correspondant au chemin d'accès de model et data les images au format jpg ou png à cadrer 
-sortie: les matrices de rotation R et de translation T à appliquer à apliquer à l'image "data" pour la cadrer comme "model"

nb1: j'ai du télécharger la police (font_file) FreeSerif.ttf pour pouvoir afficher les résultats de la méthode
nb2: la pipeline correspond à une ICP sur les edges (moins efficace sur les angles)
sources: https://engineering.purdue.edu/kak/distICP/ICP-2.1.1.html
"""

import ICP
import os


IMDIR= ### A completer ####
chemin1= ### A completer ####  #model
chemin2= ### A completer ####   #data

os.chdir(IMDIR) 

#### ICP #####

icp = ICP.ICP(
           
           binary_or_color = "colors",
           corners_or_edges = "edges",
           auto_select_model_and_data = 1,
           calculation_image_size = 200,
           max_num_of_pixels_used_for_icp = 300,
           pixel_correspondence_dist_threshold = 50,
           iterations = 10,
           model_image = chemin1,
           data_image = chemin2, #nom_imref_planmed, 
           
           #font_file = r"C:\Users\Gauthier Frecon\Downloads\freeserif\FreeSerif/FreeSerif.ttf"
           
           )


icp.extract_pixels_from_color_image("model")
icp.extract_pixels_from_color_image("data")
icp.icp()

####  matrices utiles ####

R=icp.R
T=icp.T

###

icp.display_images_used_for_edge_based_icp()
icp.display_results_as_movie()
icp.cleanup_directory()




#### exploitation des resultas ####      à travailler

#si les images sont de tailles [n,p], et que l'icp les redimensionne en 200*200 (option), on peut transformer T par  T[0]=T[0]*n/200 et T[1]=T[1]*p/200

#T[0]=T[0]*n/200 
#T[1]=T[1]*p/200
#num_rows, num_cols = im.shape
#translation_matrix = np.float32([ [1,0,T[0]], [0,1,T[1]] ])
#model=plt.imread(chemin1)
#data=plt.imread(chemin2)
#img_translation = cv2.warpAffine(data, translation_matrix, (num_cols, num_rows))
#


