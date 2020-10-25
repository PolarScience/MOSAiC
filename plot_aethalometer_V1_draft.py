# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:02:59 2020

@author: beck_i
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
import time
#import ruptures as rpt
#import changefinder as cf
#sns.set()



start = time.time()

#%%START=====================================================================
#Close all existing figures:
plt.close("all") #this is the same as: .atplotlib.pyplot.close("all")
plt.style.use('seaborn-white')
#==========================================================================
    
#%% LOAD DATA
    #Define files 
file =r'F:\All_Data\Aethalometer/Aethalometer_200210-200219.csv'
hdffile =r'F:\All_Data\Aethalometer/Aethalometer_190928-200225.hdf'
aethadat = pd.read_hdf(hdffile)
#aethadat = pd.read_csv(file, index_col=1)
#aethadat.index=pd.to_datetime(aethadat.index,infer_datetime_format=True) #Replace Index of df with the time column


#%% Preprocess data
BC1 = aethadat['BC1;']
BC2 = aethadat['BC2;']
BC3 = aethadat['BC3;']
BC4 = aethadat['BC4;']
BC5 = aethadat['BC5;']
BC6 = aethadat['BC6;'] #BC6 is best for ship pollution
BC7 = aethadat['BC7;']

meanBC1=BC1.resample('300s').mean()
meanBC2=BC2.resample('300s').mean()
meanBC3=BC3.resample('300s').mean()
meanBC4=BC4.resample('300s').mean()
meanBC5=BC5.resample('300s').mean()
meanBC6=BC6.resample('300s').mean()
meanBC7=BC7.resample('300s').mean()


    #Choose time period of interest
startdate = pd.to_datetime('2019-12-30 00:00:00')
enddate = pd.to_datetime('2020-01-02 00:00:00')

#%% Plot with Subplots
# First create a grid of plots
# ax will be an array of two Axes objects

#fig, ax = plt.subplots()
# Call plot() method on the appropriate object
#ax.plot(meanBC1, label='BC1')
#ax[1].plot();
#ax.plot(meanBC2, color = 'green', label='BC2')
#ax.plot(BC3, color = 'red', label='BC3')
#ax.plot(BC4, color = 'grey', label='BC4')
#ax.plot(BC6, color = 'purple', label='BC6')

end = time.time()
print(end - start)