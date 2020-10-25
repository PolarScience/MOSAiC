# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:07:50 2020

@author: ivo

Script to load data from HDF file and prepare data: 
    Delete zero measurements, clean data etc. 
    Output should be cleaned, useful dataframe, ready to plot.  
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
datastart = pd.to_datetime('2019-09-27 00:00')
dataend = pd.to_datetime('2020-05-11 07:58')

    
    # LOAD CPC FILES
cpcfile =r'D:\All_Data\CPC3776\CPC3776_tot_EAC.hdf'
cpcraw = pd.read_hdf(cpcfile)
cpcraw= cpcraw.loc[datastart:dataend]
    #Resample all CPC measurements to 10s interval (there is some 1s measurements in between)
cpcraw=cpcraw.resample('10s').mean()
cpcraw['gradient']=np.gradient(cpcraw['CPCconc']) #add centered gradients over 3 values to raw data
cpcraw['gradient']=cpcraw['gradient'].abs() #Take only absolute changes, positve or negative




#%% Define Unnusable data range (from logbook, manually written in here)

df_zeros = pd.DataFrame()
liste = []
for idx,value in enumerate(cpc3776_cal_start):    
    df_temp=pd.date_range(cpc3776_cal_start[idx],cpc3776_cal_end[idx],freq='1min').to_frame() #create 1min values between start and end time of each calibration
    df_zeros=df_zeros.append(df_temp) #this is the series of zeros 

cpc_1min=cpcraw.resample('60s').mean()
cpc_1min['calib']=df_zeros.copy() #all calibration data (from logbook)
    #clean CPC data from calibration sessions
cpc_1min['clean_cpc'] = cpc_1min['CPCconc'][cpc_1min['calib'].isna()]


"""
cpc_3776_data = pd.DataFrame()


# This dataframe finally contains: 
#     CPC 60s averaged data, cleaned from calibrations and pollution

cpc_3776_data['counts']=cpcraw['CPCconc'].resample('60s').mean() #60s averaged cpc data

df_1min['log_counts'] = np.log(df_1min['counts'])
df_1min['gradient']=cpcraw['gradient'].resample('60s').mean()
df_1min['line']=a*(df_1min['counts']**m)
df_1min['normalized']=df_1min['gradient']/df_1min['line'] 
df_1min['good']=df_1min['counts'][df_1min['normalized']<1]
df_1min['bad']=df_1min['counts'][df_1min['normalized']>1]
df_1min['goodwind'] = df_1min['counts'][(df_weather['winddir'] >= 270) | (df_weather['winddir'] <= 75)]
df_1min['badwind']=df_1min['counts'][df_1min['goodwind'].isna()]

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


    Write good data to CSV file
df_1min[['good','bad']].to_csv(os.path.join(outpath, outfile_csv))

"""
