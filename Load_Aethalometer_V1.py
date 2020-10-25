# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:20:03 2020

@author: Ivo Beck, ivo.beck@psi.ch

PURPOSE:

CHANGELOG:

AETHALOMETER 
"""


import csv
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO
import time

start = time.time()


#%%START=====================================================================
#Close all existing figures:
plt.close("all") #this is the same as: .atplotlib.pyplot.close("all")
plt.style.use('classic')
#==========================================================================


#%% LOAD Aethalometer
def load_Aethalometer(file):
    df=pd.read_csv(file, skiprows=[0,1,2,3,4,6,7],sep = ' ',usecols=[0,1,55], parse_dates = [[0,1]], infer_datetime_format = True, index_col=False)
    # usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    # cols 0,1,38-58
    return(df)

#%% LOAD DATA
datapath = r'F:\All_Data\Aethalometer/'
files = sorted(glob.glob(datapath + 'AE33_AE33*.dat')) #Open pointer to files

aethadata=pd.DataFrame()
for j in files:
    rawdata = load_Aethalometer(j)
    rawdata.columns=pd.Series(rawdata.columns).str.strip()
    aethadata=aethadata.append(rawdata)  

    
# Sort by datetime index
aethadata.index=pd.to_datetime(aethadata['Date(yyyy/MM/dd);_Time(hh:mm:ss);']) #Replace Index of df by time
aethadata.sort_index(inplace=True) #sort index by datetime 

aethadata=aethadata.resample('60s').mean()

aethadata.to_hdf(r'F:\All_Data\Aethalometer\Aethalometer_190928-200225.hdf', key='aetha', format= 'table', append = True)
#aethadata.to_csv(r'F:\All_Data\Aethalometer\Aethalometer_test_190928-200225.csv', mode='a')

#sep_data.to_hdf(os.path.join(outpath, outfile_hdf), key='ccnc',format = 'table', append = True)


#%%
end = time.time()
print(end - start)