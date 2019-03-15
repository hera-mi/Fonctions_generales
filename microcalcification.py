import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import os
from PIL import Image
from pylab import *
import skimage.io as io
from skimage.transform import resize
from scipy import ndimage
import numpy as np
import matplotlib.image as mpimg
import cv2 
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
import numpy as np
from skimage import measure, morphology, segmentation
from Fct_generales import *

pyplot.close('all')
#IMDIR= '/Users/reda/Downloads/EI2/PROJET/base de données hologic'
IMDIR= "/Users/reda/Downloads/EI2/PROJET/BASE DE DONNEES GE"
ds = pydicom.dcmread(IMDIR+"/1.dcm")
im= ds.pixel_array
mask = mpimg.imread("/Users/reda/Downloads/CODE_PY/GE_mask.png") #lecture du mask
plt.figure(10)
plt.imshow(mask, cmap=plt.cm.gray)
plt.axis('off')

"""
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
mask = rgb2gray(mask)
plt.figure(10)
plt.imshow(mask, cmap=plt.cm.gray)
plt.axis('off')

mask=isoler(mask, [0.72,0.90], [0.4,0.97])
plt.figure(10)
plt.imshow(mask, cmap=plt.cm.gray)
plt.axis('off')
mask = mask>0
"""
#redimensionnement

[n,p]=np.shape(im)
if np.mean(im[n//2-10:n//2+10 , 0:20]) > np.mean(im[n//2-10:n//2+10 , p-21:p-1]) :
     im=-im+np.max(im)

im_red=redim_im(im)
[n,p]=np.shape(im_red)

#sharpening+ filtering
f = im_red
#blurred_f = ndimage.gaussian_filter(f, sigma=1)
#filter_blurred_f = ndimage.gaussian_filter(blurred_f, sigma=8)
#alpha = 60
#sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
#sharpened=isoler(f, [0.78,0.90], [0.42,0.867])
#plt.figure(1)
#plt.imshow(sharpened, cmap=plt.cm.gray)
#plt.axis('off')

#filtrage 
#fftc_highpass=highpass_filter(im_red,Dc=200)
fftc_highpass=highpass_filter(f,Dc=200)
fft_highpass=np.fft.ifftshift(fftc_highpass)
invfft_highpass=np.real(np.fft.ifft2(fft_highpass))
invfft_highpass=isoler(invfft_highpass, [0.78,0.90], [0.4,0.87])
plt.figure(2)
plt.imshow(invfft_highpass, cmap=plt.cm.gray)
plt.axis('off')

#Correlation
pas = 9
kernel = np.zeros((2*pas+1,2*pas+1))
for i in range(2*pas+1):
    for j in range(2*pas+1):
        if (i-9)**2+(j-9)**2<1:
            kernel[i,j]=1;
           
#plt.imshow(kernel, cmap=plt.cm.gray)
corr_mask=signal.correlate(invfft_highpass, kernel, mode='same')
#plt.figure(2)
#plt.imshow(corr_mask, cmap=plt.cm.gray)
#plt.axis('off')
#plt.figure(7)
#plt.imshow(corr_mask, cmap=plt.cm.gray)
#plt.axis('off')
# Seuillage 
#Is= corr_mask > 246
Is= corr_mask > 100
plt.figure(3)
plt.imshow(Is, cmap=plt.cm.gray)
plt.axis('off')

#Labeling and counting
labeled_array, num_features =scipy.ndimage.measurements.label(Is)
print("le nombre de micro-calcification après traitement",num_features)
labeled_array, num_features =scipy.ndimage.measurements.label(mask)
print("le nombre de micro-calcification après traitement dans le mask", num_features-17)#17 le nombre d'elements hors microcalcification 





