# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:28:53 2020

@author: ivo
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


    #Set start and enddate for data so that both dataframes match each other
datastart = pd.to_datetime('2019-09-27 00:00')
dataend = pd.to_datetime('2020-05-11 00:00')





df_zeros = pd.DataFrame()
liste = []
for idx,value in enumerate(cpc3776_cal_start):    
    df_temp=pd.date_range(cpc3776_cal_start[idx],cpc3776_cal_end[idx],freq='1min').to_frame()
    df_zeros=df_zeros.append(df_temp)

   # LOAD CPC FILES
cpcfile =r'D:\All_Data\CPC3776\CPC3776_tot_EAC.hdf'
cpcraw = pd.read_hdf(cpcfile)
cpcraw= cpcraw.loc[datastart:dataend]
cpc_1min=cpcraw.resample('60s').mean()

cpc_1min['calib']=df_zeros.copy()