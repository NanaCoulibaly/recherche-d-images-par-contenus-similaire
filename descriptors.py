from BiT import bio_taxo

#def bitdesc(data):
   # all_statistics = bio_taxo(data)
   # return all_statistics

from BiT import bio_taxo
from skimage.feature import graycomatrix, graycoprops # scikit-image
import mahotas.features as ft
import numpy as np
def bitdesc(data):
    all_statistics = bio_taxo(data)
    return all_statistics
def haralick(data):
    all_statistics = ft.haralick(data).mean(0)
    return all_statistics
def haralick_with_mean(data):
    all_statistics = ft.haralick(data).mean(0)
    return all_statistics

 

# Gray-Level Co-occurence Matrix
def glcm(data):
    glcm = graycomatrix(data, [2], [0], 256, symmetric=True, normed=True)
    diss = graycoprops(glcm, 'dissimilarity')[0,0]
    cont = graycoprops(glcm, 'contrast')[0,0]
    corr = graycoprops(glcm, 'correlation')[0,0]
    ener = graycoprops(glcm, 'energy')[0,0]
    homo = graycoprops(glcm, 'homogeneity')[0,0]    
    all_statistics = [diss, cont, corr, ener, homo]
    return all_statistics