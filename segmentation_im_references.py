# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:54:45 2019

@author: Gauthier Frecon


"""
plt.close('all')


IMDIR_fuji=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-fujifilm-dataset-201812130858\datasim-prj-phantoms-fuji-dataset-201812130858"
IMDIR_planmed=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-planmed-dataset-201812061411\datasim-prj-phantoms-dataset-201812061411\digital\2.16.840.1.113669.632.20.20140513.192554394.19.415"
IMDIR_hologic=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-hologic-20190125-mg-proc"
IMDIR_ge=r"D:\Documents\Projet Mammographie\datasim-prj-phantoms-ge-20190125-mg-proc"

nom_imref_ge="ge-0001-0000-00000000.dcm"
nom_imref_hologic="hologic-MG02.dcm"
nom_imref_planmed="2.16.840.1.113669.632.20.20140513.202406491.200064.424.dcm"
nom_imref_fuji=""


#planmed
chemin=IMDIR_planmed + "/" + nom_imref_planmed
ds = pydicom.dcmread(chemin)
imref_planmed=ds.pixel_array
pipeline_segm_fibre(imref_planmed)

#ge
chemin=IMDIR_ge + "/" + nom_imref_ge
ds = pydicom.dcmread(chemin)
imref_ge=ds.pixel_array
imref_ge=linear(imref_ge, -1, 10000)

pipeline_segm_fibre(imref_ge, zone_fibre_n=[0.9,0.20], zone_fibre_p=[0.68,0.85])

#hologic
chemin=IMDIR_hologic + "/" + nom_imref_hologic
ds = pydicom.dcmread(chemin)
imref_hologic=ds.pixel_array
imref_hologic=linear(imref_hologic, -1, 10000)
pipeline_segm_fibre(imref_hologic)
