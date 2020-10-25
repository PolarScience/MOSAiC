# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 08:04:23 2020

@author: ivo

Test reading csv files etc. 
Test for performance, if it works etc...



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
#import ruptures as rpt
#import changefinder as cf
sns.set()

#%%
start = time.time()
   
def load_cpc(datafile):
    """
    Load cpc data (.csv)
    """
    df=pd.read_csv(datafile, index_col = 0, usecols=['TimeEnd', 'CPCconc']) # #sep='\s+' for variable amount of white space
    df.index=pd.to_datetime(df.index, infer_datetime_format=True) #Replace Index of df with the time column
    df.sort_index(inplace=True) #sort index by datetime 
    
    return(df)
    
    
#%% 
    #Define files 
cpcfiles=sorted(glob.glob(r'D:/All_Data/CPC3776/CPC3776_190921-200226_2.csv'))

#%% Load CPC data
for j in cpcfiles:
    cpc_raw = load_cpc(j)
    
    

    