# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os 
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def watch_single_value(filename, dtype, idx):
    byte_size = np.dtype(dtype).itemsize
    offset=idx*byte_size
    with open(filename, "rb") as f:
        f.seek(offset, os.SEEK_SET)
        value=np.fromfile(f, dtype, 1)[0]
    return value

def write_to_bin(input, filename):
    np.array(input).tofile(filename)
    return 0

def load_single_channel(filename, dtype, nchannels, channel):
    byte_size = np.dtype(dtype).itemsize
    #print(byte_size)
    stride=nchannels*byte_size
    with open(filename, "rb") as f:
        nframe=0
        offset=(channel-1)*byte_size
        f.seek(offset, os.SEEK_SET)        

        data=[]
        while f.read(byte_size) != b"":
        #while np.fromfile(f, dtype, nchannels).size == nchannels :
            offset=(channel-1)*byte_size+nframe*stride
            f.seek(offset, os.SEEK_SET)
            data.append(np.fromfile(f, dtype, 1)[0])
            #print(f.tell(), np.fromfile(f, dtype, 1), nframe)            
            nframe += 1
            offset=(channel-1)*byte_size+nframe*stride
            f.seek(offset, os.SEEK_SET)

    return data

def find_by_deviration(input, dist):
    mean = np.mean(input)
    std = np.std(input)
    bound = dist*std
    #print(mean,bound)
    output = []
    for i in list(range(input.size)):
        if np.abs(input[i]-mean) > bound :
            output.append(i)

    return output

"""
#function test
def fake_binary(number, filename, dtype):
    np.arange(number, dtype=dtype).tofile(filename)
    return 0

fake_binary(3740, "fake.bin", np.int64)
data = load_single_channel("fake.bin", np.int64, 374, 1)
"""    

#data = load_single_channel("190407_CA014_session_000.dat", np.int16, 374, 1)
#print(len(data))
#np.array(data).tofile("channel_001.bin")

 
data=np.empty(1000, np.int16)
for i in list(range(data.size)):
    #print(i, watch_single_value("channel_001.bin", np.int16, i))
    data[i]=watch_single_value("channel_001.bin", np.int16, i)
"""
data = np.array(load_single_channel("channel_001.bin", np.int16, 1, 1))
""" 
df=pd.Series(data)
avg=np.full(df.size, df.mean())
std=np.full(df.size, df.std())
plt.figure()
plt.plot(df.index, df, 'k')
plt.axhline(y=df.mean(), color='b')
plt.fill_between(df.index, avg - 1.0 * std, avg + 1.0 * std, color='r', alpha=0.8)
plt.fill_between(df.index, avg - 2.0 * std, avg + 2.0 * std, color='r', alpha=0.4)
plt.fill_between(df.index, avg - 3.0 * std, avg + 3.0 * std, color='r', alpha=0.2)

one_sigma=find_by_deviration(data, 1.0)
two_sigma=find_by_deviration(data, 2.0)
thr_sigma=find_by_deviration(data, 3.0)

plt.scatter(one_sigma, data[one_sigma], c='c', alpha=0.2) 
plt.scatter(two_sigma, data[two_sigma], c='g', alpha=0.4) 
plt.scatter(thr_sigma, data[thr_sigma], c='m', alpha=0.8)   
