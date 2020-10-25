# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:59:53 2020

@author: ivo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:01:01 2020

@author: becki

Load CPC data and 
        
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

    #LOAD WEATHER DATA
weatherfile = r'D:/All_Data/Weather/weather_EAC.hdf'
df_weather = pd.read_hdf(weatherfile) #Its 1min averaged weather data
df_weather = df_weather.rename(columns = {'WEATHER.PAWIBWW.air_pressure':'pressure','WEATHER.PAWIBWW.rel_wind_velocity':'windvel', 'WEATHER.PAWIBWW.rel_wind_direction':'winddir'})
df_weather = df_weather.loc[datastart:'2020-05-26 23:59']
    
    # LOAD CPC FILES
cpcfile =r'D:\All_Data\CPC3776\CPC3776_tot_EAC.hdf'
cpcraw = pd.read_hdf(cpcfile)
cpcraw= cpcraw.loc[datastart:dataend]
cpc_1min=cpcraw.resample('60s').mean()

    # LOAD CCNC DATA
ccncfile =r'D:\All_Data\CCNC\MOSAiC_CCNC_all_new_2.hdf'
CCNCdat = pd.read_hdf(ccncfile)
ccncdata=CCNCdat.loc[datastart:dataend] #contains CCNC Data for time period: Time, SS0.15, SS0.2, SS0.3, SS0.5, SS1, and NumberConc 

    # LOAD SMPS DATA
smpsfile =r'D:\All_Data\SMPS\SMPS_resampled.hdf'
df_smps = pd.read_hdf(smpsfile)
df_smps= df_smps.loc[datastart:dataend]



    # DEFINE OUTPUT FILES
outpath=Path(r'D:\MOSAiC Pollution Mask')
outfile_csv='good_dates_leg1-3.csv'
outfile_hdf='good_dates_leg1-3.csv'

#%%##############################################################################

dirmin = 75 #wind window min
dirmax = 270 #wind window max

#%%##############################################################################
    #Resample all CPC measurements to 10s interval (there is some 1s measurements in between)
cpcraw=cpcraw.resample('10s').mean()
cpcraw['gradient']=np.gradient(cpcraw['CPCconc']) #add gradients to raw data
cpcraw['gradient']=cpcraw['gradient'].abs() #Take only absolute changes, positve or negative

#%%##############################################################################

    #Create a dataframe for the defined time window
    #Choose time period of interest
startdate = pd.to_datetime('2019-12-01 00:00')
enddate = pd.to_datetime('2019-12-31 00:00')

df_window=pd.DataFrame()
df_window['counts']=cpcraw['CPCconc'].loc[startdate:enddate].resample('60s').mean()
df_window['windvel']=df_weather['windvel'].loc[startdate:enddate]
df_window['winddir']=df_weather['winddir'].loc[startdate:enddate]

m = 0.64 #slope m, get slope from: m=log(y2/y1)/log(x2/x1)
a=0.4 # intercept a corresponds to: (y1/x1**m)

df_1min=pd.DataFrame()
df_1min['counts']=cpcraw['CPCconc'].resample('60s').mean()
df_1min['log_counts'] = np.log(df_1min['counts'])
df_1min['gradient']=cpcraw['gradient'].resample('60s').mean()
df_1min['line']=a*(df_1min['counts']**m)
df_1min['normalized']=df_1min['gradient']/df_1min['line'] 
df_1min['good']=df_1min['counts'][df_1min['normalized']<1]
df_1min['bad']=df_1min['counts'][df_1min['normalized']>1]
df_1min['goodwind'] = df_1min['counts'][(df_weather['winddir'] >= 270) | (df_weather['winddir'] <= 75)]
df_1min['badwind']=df_1min['counts'][df_1min['goodwind'].isna()]


ccncdata=ccncdata.resample('60s').mean()
ccncdata['SS1']=ccncdata['CCN Number Conc'][ccncdata['SS1'].notna()]
ccncdata['SS0.5']=ccncdata['CCN Number Conc'][ccncdata['SS0.5'].notna()]
ccncdata['SS0.3']=ccncdata['CCN Number Conc'][ccncdata['SS0.3'].notna()]
ccncdata['SS0.2']=ccncdata['CCN Number Conc'][ccncdata['SS0.2'].notna()]
ccncdata['SS0.15']=ccncdata['CCN Number Conc'][ccncdata['SS0.15'].notna()]
ccncdata['good']=df_1min['good']
ccncdata=ccncdata[ccncdata['good'].notna()] #Keep only good data

# ss1 = ccncdata['SS1']!=0 #Old way, contains zero values
ss1=ccncdata['SS1'].loc[(ccncdata['SS1']>=1)]
# ss03 = ccncdata['SS0.3']
ss03=ccncdata['SS0.3'].loc[(ccncdata['SS0.3']>=1)]