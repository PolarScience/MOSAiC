# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:07:54 2020

@author: ivo


Load CPC data from HDF file
extract the counts
average them
choose time period of interest 
plot it

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
import scipy.stats as ss

#import ruptures as rpt
#import changefinder as cf
sns.set()

#%%
start = time.time()

#%%START=====================================================================
#Close all existing figures:
plt.close("all") #this is the same as: .atplotlib.pyplot.close("all")
# plt.style.use('seaborn')
#==========================================================================

#%% Load all data,  and set start and enddate for all. 

    #Set start and enddate for data so that both dataframes match each other
datastart = pd.to_datetime('2020-04-08 00:00')
dataend = pd.to_datetime('2020-04-10 23:59')


    # LOAD CPC FILES
cpcfile =r'D:\All_Data\CPC3025\MOSAiC_CPC3025int_all.hdf'
cpcraw = pd.read_hdf(cpcfile)

#     # Choose the time period of interest
# cpcraw= cpcraw.loc[datastart:dataend]

    # Calculate mean temperature
cpc_1min=cpcraw.resample('60s').mean() 

counts = cpc_1min['CPCconc']




end = time.time()
print(end - start)

