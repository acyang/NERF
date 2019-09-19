#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:00:04 2019

@author: acyang
"""

import os
import time 
import numpy as np
import pandas as pd
import seaborn as sb
import h5py
import matplotlib.pyplot as plt

#filename="/991GB/tmp/190407_CA014/000/channel_001.bin"
#data = load_single_channel(filename, np.int16)

with h5py.File("output/7_sigma.h5", "r") as f:
    print(list(f.keys()))

    index=f["index"][:]
    sd=f["Spike_Data"][:]
    
#print(type(dset))
print(index.shape, sd.shape)

df = pd.DataFrame(sd) 
#print(df)

corrmat = df.T.corr()
mask=corrmat<0.9
print(corrmat)
print(mask) 
  
f, ax = plt.subplots(figsize =(9, 8)) 
sb.heatmap(corrmat, ax = ax, cmap ="YlGnBu", mask = mask, linewidths = 0.01)   