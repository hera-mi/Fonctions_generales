# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:37:08 2019

@author: Gauthier Frecon

on ne voit pas les zones de repere à tester


tester d'abotd avec n=billes et carres
"""

import ICP
import os


os.chdir(r"C:\Users\Gauthier Frecon\Documents\GitHub\Fonctions_generales\MASK") 
nom_imref_ge=r"ge-0001-0000-00000000.dcm-w1.png"
nom_imref_hologic=r"hologic-MG02.dcm.png"
nom_imref_planmed=r"PM-2.16.840.1.113669.632.20.20130917.192819263.200064.384.dcm.png"  
nom_imref_fuji=r"1.2.392.200036.9125.4.0.2702151069.50333032.466243176.dcm.png"

chemin_ge=nom_imref_ge
chemin_hologic=IMDIR_hologic+nom_imref_hologic


icp = ICP.ICP(
           
           binary_or_color = "colors",
           corners_or_edges = "edges",
           auto_select_model_and_data = 1,
           calculation_image_size = 200,
           max_num_of_pixels_used_for_icp = 300,
           pixel_correspondence_dist_threshold = 50,
           iterations = 10,
           model_image = "test1.png", #nom_imref_ge
           data_image = "test2.png", #nom_imref_planmed, 
           font_file = r"C:\Users\Gauthier Frecon\Downloads\freeserif\FreeSerif/FreeSerif.ttf") #'/Library/Fonts/Arial.ttf',)#


#icp = ICP.ICP(
#           binary_or_color = "color",
#           corners_or_edges = "corners",
#           calculation_image_size = 200,
#           image_polarity = -1,
#           smoothing_low_medium_or_high = "medium",
#           corner_detection_threshold = 0.005,
#           pixel_correspondence_dist_threshold = 40,
#           auto_select_model_and_data = 1,
#           max_num_of_pixels_used_for_icp = 100,
#           iterations =4,
#           model_image =  nom_imref_ge,
#           data_image = nom_imref_hologic,
#           font_file = r"C:\Users\Gauthier Frecon\Downloads\freeserif\FreeSerif/FreeSerif.ttf"
#        )

icp.extract_pixels_from_color_image("model")
icp.extract_pixels_from_color_image("data")
icp.icp()
R=icp.R
T=icp.T
icp.display_images_used_for_edge_based_icp()
#icp.display_images_used_for_corner_based_icp()
icp.display_results_as_movie()
icp.cleanup_directory()


model=plt.imread("test1.png")
data=plt.imread("test2.png")

num_rows, num_cols = im.shape
translation_matrix = np.float32([ [1,0,-3], [0,1,203] ])
img_translation = cv2.warpAffine(data, translation_matrix, (num_cols, num_rows))


plt.figure(1)
plt.imshow(model)
plt.show()
plt.figure(2)
plt.imshow(data)
plt.show()

plt.figure(3)
plt.imshow(img_translation)
plt.show()

# met au meme e,droit quand on fait T[1]/200*n donne bon nb de pixels a decaler