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

def corl(s1, s2, time_window):
    single1=pd.Series(s1)
    single2=pd.Series(s2)
    corrmat=np.zeros(shape=time_window, dtype=np.float64)
    
    for t in list(range(time_window)):
        corrmat[t]=single1.corr(single2.shift(periods=-t), method="pearson")
        print(t, corrmat[t])
        
    return corrmat

def corl_dumy(s1, s2, time_window):
    pass
    return

with h5py.File("output/7_sigma.h5", "r") as f:
    print(list(f.keys()))

    index=f["index"][:]
    sd=f["Spike_Data"][:]
    
#print(index.shape, index.dtype)
#print(sd.shape, sd.dtype)
#print(sd[0,:])

#nsingle=list(range(4))
nsingle=list(range(sd.shape[0]))
ntime=list(range(sd.shape[1]))
flags=np.zeros(shape=sd.shape[0], dtype=np.int)
#print(nsingle,ntime)


cal_corr=[corl, corl_dumy]

#for i in nsingle:
#    s1=pd.Series(sd[i,:])
    
#    for j in nsingle[i:]:
#        s2=pd.Series(sd[j,:])

#corrmat=np.zeros(shape=(sd.shape[0],sd.shape[0],sd.shape[1]), dtype=np.float64)
#for i in nsingle:
#    s1=pd.Series(sd[i,:])
#    #print(s1)
#    for j in nsingle[i:]:
#        s2=pd.Series(sd[j,:])
#        #print(i,j)
#        for tau in ntime:
#            #print(i, j, tau)
#            #print(tau, s2.shift(periods=-tau))
#            corrmat[i,j,tau]=s1.corr(s2.shift(periods=-tau), method="pearson")
#    
##print(corrmat[:,:,0])    
##df = pd.DataFrame(sd) 
##print(df)
#
##corrmat = df.T.corr()
#mask=corrmat<0.9
###print(corrmat)
###print(mask) 
#  
#f, ax = plt.subplots(figsize =(9, 8)) 
#sb.heatmap(corrmat[:,:,0], ax = ax, cmap ="YlGnBu", linewidths = 0.01)   