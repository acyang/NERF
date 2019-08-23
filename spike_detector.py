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

    #print(nframe)
    #print(data)
    return data

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

data=[]
for i in list(range(1,1000)):
    #print(i, watch_single_value("channel_001.bin", np.int16, i))
    data.append(watch_single_value("channel_001.bin", np.int16, i))

df=pd.Series(data)
print(df.size)
df.plot()
plt.show()
#data = load_single_channel("channel_001.bin", np.int16, 1, 1)
#print(len(data))
#df=pd.Series(data)

