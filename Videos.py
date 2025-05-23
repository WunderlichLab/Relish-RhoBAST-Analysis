#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 18:52:44 2025

@author: noshin
"""


import matplotlib.pyplot as plt
import tifffile as tf
import numpy as np
import matplotlib.patches as patches
import math
import os
import cv2
from scipy import ndimage
import skimage.io
from skimage import measure
from skimage.measure import label, regionprops
from scipy.ndimage import label
from scipy.ndimage import binary_dilation
import pandas as pd
import tifffile
from tifffile import imread
from tifffile import imshow
from skimage.segmentation import mark_boundaries
from skimage.feature.peak import peak_local_max
from skimage.feature import corner_fast, corner_peaks
import matplotlib.gridspec as gridspec
from PIL import Image
import pickle
import datetime
import time
from datetime import timedelta
import matplotlib.patches as patches
import skimage.io
import  microfilm.microplot
from microfilm import microplot
from microfilm import microanim
#from microfilm.microanim import panel
from microfilm.microplot import microshow, Micropanel
import seaborn as sns
import gc
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import copy
from scipy.signal import savgol_filter


datetime_str = time.strftime("%m%d%y_%H:%M")

mac = '/Volumes/rkc_wunderlichLab/'
PC = 'R:/'
anacomp = 'Z:/'
resolution=  3.4756 #pixels per micron
units_per_pix = 1/resolution

computer = anacomp

plt.rcParams['font.family'] = 'Arial'
labelfsize = 12
fsize = 10
tickfsize = 9


fig_output = 'G:/Shared drives/Wunderlich Lab/People/Noshin/Paper/Figures/'

#%%
#long time course
tif_long_path ='/Users/noshin/Desktop/07_Dpt_100X_1_maxZ_times_brightadj.tif'
tif_long = skimage.io.imread(tif_long_path)
data = np.transpose(tif_long, (3, 0, 1, 2))  # Transpose from (time, height, width, channels) → (channels, time, height, width)

tif_time = skimage.io.imread('/Users/noshin/Desktop/long_time.tif')
tif_time = np.expand_dims(tif_time, axis=1) # Adds channel at the end (shape: (2, 2, 1))
tif_time = np.transpose(tif_time, (1, 0,2,3))  # Transpose from (time, height, width, channels) → (channels, time, height, width)

#%% long video
scale = 3.4756 #pixels/micron

microanim1 = microanim.Microanim(
    data= tif_time[[0]],              # only the first channel → shape: (1, 32, 244, 244)
    cmaps= ['pure_blue'],
    flip_map=[False],
    label_text= 'Hoechst',
    label_color= 'blue',
    label_location='lower-left',
    label_font_size= 24,
    #rescale_type='limits', limits=minmax,
    dpi=700)

minmax = [45, 1818]
microanim2 = microanim.Microanim(
    data=data[[1]],              # channel 1
    cmaps=['pure_red'],
    flip_map=[False],
    label_text='Relish',
    label_color='red',
    label_location='lower-left',
    label_font_size= 24,
    rescale_type='limits', limits=minmax,
    dpi=700
)
minmax = [280, 3500]
microanim3 = microanim.Microanim(
    data=data[[2]],              # channel 2
    cmaps=['pure_yellow'],
    flip_map=[False],
    label_text='RhoBAST',
    label_color='yellow',
    label_location='lower-left',
    label_font_size= 24,
    rescale_type='limits', limits=minmax,

    unit='um', scalebar_unit_per_pix=1/scale, 
    scalebar_size_in_units=20, scalebar_thickness=0.03,
    scalebar_color='white', scalebar_font_size= 14,
    dpi=700
)

animpanel = microanim.Microanimpanel(rows=1, cols=3, figsize=[12,6])
animpanel.add_element(pos=[0,0], microanim=microanim1)
animpanel.add_element(pos=[0,1], microanim=microanim2)
animpanel.add_element(pos=[0,2], microanim=microanim3)




animpanel.save_movie('/Users/noshin/Desktop/long.mp4',fps=5)

#%% dense video
tif_dense_path='/Users/noshin/Desktop/Dpt_100X_T1_0306_timescalebar_brightadj.tif'
tif_dense = skimage.io.imread(tif_dense_path)
data = np.transpose(tif_dense, (3, 0, 1, 2))  # Transpose from (time, height, width, channels) → (channels, time, height, width)


tif_time = skimage.io.imread('/Users/noshin/Desktop/dense_time.tif')
tif_time = np.expand_dims(tif_time, axis=1) # Adds channel at the end (shape: (2, 2, 1))
tif_time = np.transpose(tif_time, (1, 0,2,3))  # Transpose from (time, height, width, channels) → (channels, time, height, width)


microanim1 = microanim.Microanim(
    data= tif_time[[0]],              # only the first channel → shape: (1, 32, 244, 244)
    cmaps= ['pure_blue'],
    flip_map=[False],
    label_text= 'Hoechst',
    label_color= 'blue',
    label_location='lower-left',
    label_font_size= 24,
    #rescale_type='limits', limits=minmax,
    dpi=700)

minmax = [35, 2688]
microanim2 = microanim.Microanim(
    data=data[[1]],              # channel 1
    cmaps=['pure_red'],
    flip_map=[False],
    label_text='Relish',
    label_color='red',
    label_location='lower-left',
    label_font_size= 24,
    rescale_type='limits', limits=minmax,
    dpi=700
)
minmax = [470, 4606]
microanim3 = microanim.Microanim(
    data=data[[2]],              # channel 2
    cmaps=['pure_yellow'],
    flip_map=[False],
    label_text='RhoBAST',
    label_color='yellow',
    label_location='lower-left',
    label_font_size= 24,
    rescale_type='limits', limits=minmax,

    unit='um', scalebar_unit_per_pix=1/scale, 
    scalebar_size_in_units=20, scalebar_thickness=0.03,
    scalebar_color='white', scalebar_font_size= 14,
    dpi=700
)

animpanel = microanim.Microanimpanel(rows=1, cols=3, figsize=[12,6])
animpanel.add_element(pos=[0,0], microanim=microanim1)
animpanel.add_element(pos=[0,1], microanim=microanim2)
animpanel.add_element(pos=[0,2], microanim=microanim3)




animpanel.save_movie('/Users/noshin/Desktop/dense.mp4',fps=20)
