# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:42:31 2020

@author: ivo

Load weather files

"""

import csv
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from numpy import nan as NA
from io import StringIO
import seaborn as sns
import matplotlib.ticker as ticker
from scipy import stats
from sklearn import preprocessing
import time
import os
#import ruptures as rpt
#import changefinder as cf
sns.set()
#%%
start = time.time()

#%% 
    #LOAD WEATHER DATA
path = r'D:/All_Data/Weather/weather_all.dat'
# file = 
weatherfiles=sorted(glob.glob(r'D:/All_Data/Weather/weather_all.dat'))
df= pd.read_csv(path,skiprows=[1,2], infer_datetime_format = True, sep='\s+', parse_dates = [[0,1]], usecols=([0,1,2,11,12])) #
df.index=pd.to_datetime(df['date_time'])
df.sort_index(inplace=True) #sort index by datetime
df=df.replace(9,NA) #Replace all '9' with nan, since 9 is the value for missing data
    #Export weather data
outpath = r'D:/All_Data/Weather'
outfile_hdf = 'weather_EAC.hdf'
# df.to_hdf(os.path.join(outpath, outfile_hdf), key='weather_EAC',format = 'table')



end = time.time()
print(end - start)