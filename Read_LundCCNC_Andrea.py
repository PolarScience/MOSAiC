# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:18:13 2019

@author: baccarini_a
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path

def Group_getindex(data,threshold=10):
    '''This function identify groups of data in a timeseries.
    It was written for the CCN timeseries but eventually it can
    be adapted to other dataset.
    Input:
        -Timeseries, should be a single column DataFrame
        -Threshold in minutes between groups
    Output:
        -Dataframe containing start and end time of each group'''

    ddf=data.dropna().reset_index()
    threshold_t = pd.Timedelta(threshold, 'm')
    starting = ddf['Time'].loc[ddf['Time'].diff() > threshold_t]
    ending = ddf['Time'].loc[starting.index-1]

    #adding first row of ddf to starting, and last row of ddf to ending
    starting = pd.Series(ddf['Time'].iloc[0]).append(starting)
    ending = ending.append(pd.Series(ddf['Time'].iloc[-1]))

    #make a dataframe, each row contains starting and ending times of a group
    groups = pd.DataFrame({'start':starting.reset_index(drop=True), 'end':ending.reset_index(drop=True)})
    return(groups)

#%% Inputs
Date = '2018-08-31'
fdir=Path('D:/2018_MOCCHA/Analysis/Dcrit_analysis/CCNCLund/')
# Set the time interval to skip data at the beginning of each SS step
# this is nedded to let the instrument equilibrate.
skipmin = 5
#%% Read files
Date = '2018-08-31'
fdir=Path('D:/2018_MOCCHA/Analysis/Dcrit_analysis/CCNCLund/')
files = fdir.glob('*.csv')
LundCCNC=pd.DataFrame()
for i in files:
    temp=pd.read_csv(i,sep=',',skiprows=[0,1,2,3,5])
    LundCCNC=LundCCNC.append(temp)

#remove whitespaces from column
LundCCNC.columns=pd.Series(LundCCNC.columns).str.strip()
#Create datetime index column
LundCCNC['Time']=pd.to_datetime(Date+' '+LundCCNC.Time)
#Set current SS value as index
LundCCNC.set_index('Current SS',inplace=True)

LundCCNCstat=pd.DataFrame()
for j in LundCCNC.index.unique()[:-1]:
    LundCCNsel=pd.Series(LundCCNC.loc[j]['CCN Number Conc'].reset_index(drop=True))
    LundCCNsel.index=LundCCNC.loc[j]['Time'].reset_index(drop=True)

    CCNselgroups=Group_getindex(LundCCNsel)

    mean_ts=pd.to_datetime((CCNselgroups.start.apply(lambda x: x.value)+\
                            CCNselgroups.end.apply(lambda x: x.value))/2)+pd.Timedelta(skipmin/2, 'm')
    stat_temp=pd.DataFrame({'Central_time':mean_ts})

    for k in CCNselgroups.start:
        LundCCNsel[k:k+pd.Timedelta(skipmin, 'm')]=np.nan

    LundCCNsel.dropna(inplace=True)
    idx_range = pd.IntervalIndex.from_arrays(CCNselgroups.start,CCNselgroups.end,closed='both')

    stat_temp['Mean']=LundCCNsel.groupby(pd.cut(LundCCNsel.index,idx_range)).mean().values
    stat_temp['Std']=LundCCNsel.groupby(pd.cut(LundCCNsel.index,idx_range)).std().values

    LundCCNCstat=pd.concat([LundCCNCstat,stat_temp],axis=1)

colon=pd.MultiIndex.from_product([['CCN01','CCN1','CCN05','CCN03','CCN02'],['Central_time','Mean','Std']])
LundCCNCstat.columns=colon

#%% Save data
LundCCNCstat.to_hdf('LundCCNC31Aug.hdf','w')

#%% Figure
plt.figure()
plt.plot(LundCCNC['Time'],LundCCNC['CCN Number Conc'],'.')
lines = plt.plot(LundCCNCstat.loc(axis=1)[:,['Central_time']],
                 LundCCNCstat.loc(axis=1)[:,['Mean']],'o')
plt.legend(lines,LundCCNCstat.columns.levels[0])
