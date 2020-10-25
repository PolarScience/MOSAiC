# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:26:43 2020

@author: beck_i
Load all CPC csv data files and put them into a dataframe 
replace index column by datetime
save dataframe to csv and hdf file

Load CPC data:
    This scrips is supposed to load several CPC datafiles and put them together into one file. 
    The intention is to skip the time consuming step of loading data every time I want to 
    plot CPC data. Epecially for long datasets it is useful. 
"""


import csv
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO
import time
import os #os.path: This module implements some useful functions on pathnames.


start = time.time()

#%% LOAD Function
def Load_CSV(datapath):
    fdir = Path(datapath)
    files = fdir.rglob('CPC_DMA_Log*.csv' ) #For Help use: help(Path.glob)
    df=pd.DataFrame() # initialize dataframe
    
    for j in files:
        df_temp=pd.read_csv(j,usecols=[0,1,17]) #
        df=df.append(df_temp)

    # Sort by datetime index
    df.index=pd.to_datetime(df.TimeEnd, dayfirst=True) #Replace Index of df by time
    df.sort_index(inplace=True) #sort index by datetime
    return(df)
    
#%% Define paths
datapath =r'D:\All_Data\CPC3776/'
outpath = Path(r'D:\All_Data\CPC3776')
outfile_csv = 'CPC3776_tot.csv'
outfile_hdf = 'CPC3776_tot_reduced.hdf'

#%% LOAD DATA
data = Load_CSV(datapath)

# data.to_csv(os.path.join(outpath, outfile_csv), mode = 'a')
data.to_hdf(os.path.join(outpath, outfile_hdf), key='cpc3776',format = 'table', append = True)

end = time.time()
print(end - start)