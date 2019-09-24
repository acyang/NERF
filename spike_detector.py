# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import time 
import numpy as np
import pandas as pd
import seaborn as sb
import h5py
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

def load_single_channel_v0(filename, dtype, nchannels, channel):
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

def split_single_channel(filename, dtype, nchannels, channel):
    file_size=os.stat(filename).st_size
    byte_size = np.dtype(dtype).itemsize
    stride=nchannels*byte_size
    if file_size%stride == 0 :
        pass
    else:
        print("the file may be corrupted!!")
        
    nframes=file_size//stride
    
    with open(filename, "rb") as f:
        data=np.empty(nframes, np.int16)
        
        for i in list(range(nframes)):
            offset=(channel-1)*byte_size+i*stride
            f.seek(offset, os.SEEK_SET)
            #data[i]=np.fromfile(f, dtype, 1)
            data[i]=int.from_bytes(f.read(byte_size), byteorder='little', signed=True)
            
    return data

def load_single_channel(filename, dtype):
    file_size=os.stat(filename).st_size
    byte_size = np.dtype(dtype).itemsize
    stride=byte_size
    if file_size%stride == 0 :
        pass
    else:
        print("the file may be corrupted!!")
        
    nframes=file_size//stride
    
    with open(filename, "rb") as f:
        data=np.empty(nframes, np.int16)
        f.seek(0, os.SEEK_SET)
        for i in list(range(nframes)):
            data[i]=int.from_bytes(f.read(byte_size), byteorder='little', signed=True)
            
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

def crop_data_0(input, idx, time_window):
    ubound=len(input)
    start=idx-time_window//2
    end=idx+time_window//2
    
    if start < 0 :
        output=input[:end]
        pad=np.abs(start)
        output=np.pad(output, (pad, 0), "constant")
    elif end >= ubound :
        output=input[start:]
        pad=end-ubound+1
        output=np.pad(output, (0, pad), "constant")
    else:
        output=input[start:end+1]
    
    return output

def crop_data(input, idx, time_window):
    padding=time_window//2
    input_padded=np.pad(input, (padding, padding), "constant")
    
    start=idx
    end=idx+time_window
    
    output=input_padded[start:end+1]
    
    return output

def write_to_hdf5(input, filename):    
    with h5py.File(filename, 'w') as f:
        f.create_dataset("index", data=input)
    
    return 0

#t = time.time()
#data = split_single_channel("190407_CA014_session_000.dat", np.int16, 374, 1)
#elapsed=time.time()-t
#print(elapsed)

"""
f = open("190407_CA014_session_000.dat", "rb")
t = time.time()
my_ch = np.zeros((35568720,),dtype='int16')
for i in range(35568720):
  my_byte=f.read(2)
  my_ch[i] = int.from_bytes(my_byte, byteorder='little', signed=True)
  f.seek(746,1)

f.close
elapsed=time.time()-t
print(elapsed)
"""
"""
#function test
def fake_binary(number, filename, dtype):
    np.arange(number, dtype=dtype).tofile(filename)
    return 0

fake_binary(3740, "fake.bin", np.int64)
data = load_single_channel("fake.bin", np.int64, 374, 1)
"""    
filename="/991GB/tmp/190407_CA014/000/channel_001.bin"
data = load_single_channel(filename, np.int16)

df=pd.Series(data)
avg=np.full(df.size, df.mean())
std=np.full(df.size, df.std())
plt.figure()
plt.plot(df.index, df, 'k')
#plt.axhline(y=df.mean(), color='b')
#plt.fill_between(df.index, avg - 3.0 * std, avg + 3.0 * std, color='r', alpha=0.8)
#plt.fill_between(df.index, avg - 5.0 * std, avg + 5.0 * std, color='r', alpha=0.4)
#plt.fill_between(df.index, avg - 7.0 * std, avg + 7.0 * std, color='r', alpha=0.2)

one_sigma=find_by_deviration(data, 3.0)
two_sigma=find_by_deviration(data, 5.0)
thr_sigma=find_by_deviration(data, 7.0)

#plt.scatter(one_sigma, data[one_sigma], c='c', alpha=0.2) 
#plt.scatter(two_sigma, data[two_sigma], c='g', alpha=0.4) 
#plt.scatter(thr_sigma, data[thr_sigma], c='m', alpha=0.8)   

#print(len(one_sigma),len(two_sigma),len(thr_sigma))

print(crop_data(data, 0, 10))
print(crop_data(data, 1, 10))
print(data[:10])
print(crop_data(data, 35568715, 10))
print(crop_data(data, 35568716, 10))
print(data[-10:])

#with h5py.File("output/7_sigma.h5", "w") as f:
#    d = f.create_dataset("index", data=thr_sigma)
#    d.attrs["Spikes_Number"] = len(thr_sigma)
#    d.attrs["Data_Type"] = "int"
#    d.attrs["Index_Type"] = "center"
#    #print(d.shape, d.dtype)
#    
#    sd = f.create_dataset("Spike_Data", shape=(len(thr_sigma), 11))
#    sd.attrs["Data_Type"] = "int"
#    sd.attrs["Time_window"] = 10
#    j=0
#    for i in thr_sigma:
#        dcrop=crop_data(data, i, 10)
#        #print(i)
#        #print(dcrop)
#        
#        sd[j,:]=dcrop
#        #print(j, sd[j,:])
#        j+=1
#        
#with h5py.File("output/5_sigma.h5", "w") as f:
#    d = f.create_dataset("index", data=two_sigma)
#    d.attrs["Spikes_Number"] = len(two_sigma)
#    d.attrs["Data_Type"] = "int"
#    d.attrs["Index_Type"] = "center"
#    #print(d.shape, d.dtype)
#    
#    sd = f.create_dataset("Spike_Data", shape=(len(two_sigma), 11))
#    sd.attrs["Data_Type"] = "int"
#    sd.attrs["Time_window"] = 10
#    j=0
#    for i in two_sigma:
#        dcrop=crop_data(data, i, 10)
#        #print(i)
#        #print(dcrop)
#        
#        sd[j,:]=dcrop
#        #print(j, sd[j,:])
#        j+=1
#        
#with h5py.File("output/3_sigma.h5", "w") as f:
#    d = f.create_dataset("index", data=one_sigma)
#    d.attrs["Spikes_Number"] = len(one_sigma)
#    d.attrs["Data_Type"] = "int"
#    d.attrs["Index_Type"] = "center"
#    #print(d.shape, d.dtype)
#    
#    sd = f.create_dataset("Spike_Data", shape=(len(one_sigma), 11))
#    sd.attrs["Data_Type"] = "int"
#    sd.attrs["Time_window"] = 10
#    j=0
#    for i in one_sigma:
#        dcrop=crop_data(data, i, 10)
#        #print(i)
#        #print(dcrop)
#        
#        sd[j,:]=dcrop
#        #print(j, sd[j,:])
#        j+=1