# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:05:42 2020

@author: ivo


Load SMPS Datafile (csv) that is produced by the IGOR SMPS Toolkit
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
    
#%% 

    #Set start and enddate for data so that both dataframes match each other
datastart = pd.to_datetime('2019-09-27 00:00')
dataend = pd.to_datetime('2020-05-11 07:58')

    # LOAD SMPS DATA
smpsfile =r'D:\All_Data\SMPS\SMPS_totalcounts_geomMeanDp_Leg1-3.csv'
smpsdat = pd.read_csv(smpsfile)
 # Sort by datetime index
smpsdat.index=pd.to_datetime(smpsdat.timecentre1, dayfirst=True) #Replace Index of df by time
smpsdat=smpsdat.resample('180s').mean()


  # DEFINE OUTPUT FILES
outpath=Path(r'D:\All_Data\SMPS')
outfile_csv='SMPS_resampled.csv'
outfile_hdf='SMPS_resampled.hdf'
# smpsdat.to_csv(..)
# smpsdat.to_hdf(os.path.join(outpath, outfile_hdf), key='smps',format = 'table')



end = time.time()
print(end - start)