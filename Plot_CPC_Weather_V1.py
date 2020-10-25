# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:01:01 2020

@author: becki

Load weather from .dat file. 
Plot rel. wind direction and wind velocity
Identify bad wind sector, mark it in plot (polluted area)
Plot CPC number concentrations paralel to the weather data
Plot: 
    1) Weather and wind direction
    2) CPC data
    
MODIFICATIONS: 
    May20: Finished script 
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
#import ruptures as rpt
#import changefinder as cf
#sns.set()


#%%START=====================================================================
#Close all existing figures:
plt.close("all") #this is the same as: .atplotlib.pyplot.close("all")
plt.style.use('seaborn-white')
#==========================================================================

#%% Some Functions
def load_weather(datafile):
    """
    Load weather data (.csv)
    Return weather dataframe with datetime in index, sorted by time   
    """
    df=pd.read_csv(datafile, index_col = 0, usecols=['date time','WEATHER.PAWIBWW.rel_wind_velocity','WEATHER.PAWIBWW.rel_wind_direction','WEATHER.PAWIBWW.visibility']) # #sep='\s+' for variable amount of white space
    df.index=pd.to_datetime(df.index,infer_datetime_format=True) #Replace Index of df with the time column
    df.sort_index(inplace=True) #sort index by datetime 
    
    return(df)
    
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
weatherfiles=sorted(glob.glob(r'D:/All_Data/Weather/weather*.csv'))
cpcfiles=sorted(glob.glob(r'D:/All_Data/CPC3776/CPC3776*.csv'))

#%% 
    #Choose time period of interest
startdate = pd.to_datetime('2019-12-30 00:00')
enddate = pd.to_datetime('2020-01-02 00:00')
    #Loop through all files
for j in weatherfiles:
    weather = load_weather(j)

wthrwin=weather.loc[startdate:enddate]
    #Loop through all files
for j in cpcfiles:
    cpc = load_cpc(j)
    cpcav=cpc.resample('10s').mean()
    cpcwin = cpc.loc[startdate:enddate]


#hist = cpc_hist.hist(bins=50)

#%% Prepare weather dataframe
wind=wthrwin['WEATHER.PAWIBWW.rel_wind_velocity']
meanwind = wind.resample('180s').mean() #Choose time period for mean values
reldir=wthrwin['WEATHER.PAWIBWW.rel_wind_direction']
meanreldir=reldir.resample('300s').mean()
visibility = wthrwin['WEATHER.PAWIBWW.visibility']
meanvis = visibility.resample('180s').mean()

dirmin = 100
dirmax = 260

#%% Prepare CPC data
dy_cpc = pd.Series(cpcwin['CPCconc'].diff())
dt_cps = pd.Series(cpcwin.index).diff().dt.seconds.values

cpc_hist = cpcwin.resample('180s').mean()

#%% Plot Histogram

ax = cpcav.plot.hist(bins=80, alpha=0.5, logy=True) #Histogram


#%% Calculate running mean of CPC values
rolling_mean = cpcav.rolling(6).mean() # 6 = 60s rolling mean

#slope = pd.Series(np.gradient(tmp.values), tmp.index, name='slope')
    #%%
    #Object-oriented interface
    # First create a grid of plots
    # ax will be an array of three Axes objects
    # Initialise a figure. subplots() with no args gives one plot.
    
fig, ax = plt.subplots(2, sharex=True)

# Rename the axes for ease of use
ax1 = ax[0]
ax2 = ax[1]

## Set plot line width
line_width = 1.0
color = 'tab:red'

# Call plot() method on the appropriate object
ax1.plot(meanreldir,'.',label='rel. wind direction', lw=line_width)
ax1.fill_between(meanreldir.index, dirmin, dirmax, alpha=0.2, label='polluted sector')
ax1.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'))
ax1.set_ylabel('wind direction [deg]')
ax1.set_ylim((0, 400))   # set the ylim to bottom, top
#ax1.set(ylabel='wind speed [m/s]', title = 'Pollution example')
#ax.xaxis.set_major_formatter(plt.NullFormatter())
ax1.legend(loc=2)

ax12 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax12.set_ylabel('wind speed [m/s]', color=color)
ax12.plot(meanwind, label = 'wind speed', color= color, lw=line_width)

#ax12.set(ylabel= 'Wind direction')  # we already handled the x-label with ax1
ax12.tick_params(axis='y', labelcolor=color)
ax12.legend(loc=1)

#ax12.set_yticks(np.linspace(ax12.get_yticks()[0], ax12.get_yticks()[-1], len(ax1.get_yticks())))

ax2.plot(cpcwin,'.',markersize=1, label='CPC3776')
ax2.set_ylabel('Particle concentration [#/cc]')
ax2.set_yscale('log')
ax2.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.autofmt_xdate()

# We'll write a custom formatter
N = len(meanreldir.index)
ind = np.arange(N)  # the evenly spaced plot indices


plt.show()

##%% New plot
#fig, ax = plt.subplots()
#num_bins = 50
#ax.hist(cpcav, bins=50, label=('cpc'))
#ax.legend()
#ax.set_title('Frequencies of counts')
#ax.yaxis.tick_right()

fig, ax = plt.subplots()

## Set plot line width
line_width = 1.0
color = 'tab:red'

# Call plot() method on the appropriate object
ax.plot(rolling_mean,'.',label='rolling mean', lw=line_width)
ax.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'))
ax.set_ylabel('rolling mean (1/cc)')
ax.set_yscale('log')
ax.set_ylim((0.1, 100000))   # set the ylim to bottom, top
#ax1.set(ylabel='wind speed [m/s]', title = 'Pollution example')
#ax.xaxis.set_major_formatter(plt.NullFormatter())
ax1.legend(loc=2)

fig.autofmt_xdate()

# We'll write a custom formatter
N = len(meanreldir.index)
ind = np.arange(N)  # the evenly spaced plot indices


plt.show()