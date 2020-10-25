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
    16.7.20: Increase fond size on plots
    16.7. Mark those points that will be filtered out in Julia's method 
        
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
sns.set()


#%%START=====================================================================
#Close all existing figures:
plt.close("all") #this is the same as: .atplotlib.pyplot.close("all")
# plt.style.use('seaborn-white')
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
weatherfiles=sorted(glob.glob(r'F:/All_Data/Weather/weather*.csv'))
cpcfiles=sorted(glob.glob(r'F:/All_Data/CPC3776/CPC3776*.csv'))

#%%Load weather data
for j in weatherfiles:
    weather = load_weather(j)

#%% Load CPC data
for j in cpcfiles:
    cpc_raw = load_cpc(j)
    
#%%##############################################################################
    #Choose time period of interest
startdate = pd.to_datetime('2019-12-27 06:00')
enddate = pd.to_datetime('2019-12-28 12:00')

t_rolmean = 18 # rolling mean periods
t_rolmedian = 18 #rolling median periods

dirmin = 100 #wind window min
dirmax = 260 #wind window max

#%%##############################################################################
wframe = pd.DataFrame()
wthrwin=weather.loc[startdate:enddate] #weather timewindow
wframe['wind']=wthrwin['WEATHER.PAWIBWW.rel_wind_velocity']
wframe['meanwind'] = wframe['wind'].resample('180s').mean() #Choose time period for mean values
wframe['reldir']=wthrwin['WEATHER.PAWIBWW.rel_wind_direction']
wframe['meanreldir']=wframe['reldir'].resample('300s').mean()
wframe['visibility']= wthrwin['WEATHER.PAWIBWW.visibility']
wframe['meanvis']=wframe['visibility'].resample('180s').mean()


# Prepare weather dataframe
wind=wthrwin['WEATHER.PAWIBWW.rel_wind_velocity']
meanwind = wind.resample('180s').mean() #Choose time period for mean values
reldir=wthrwin['WEATHER.PAWIBWW.rel_wind_direction']
meanreldir=reldir.resample('20s').mean()
visibility = wthrwin['WEATHER.PAWIBWW.visibility']
meanvis = visibility.resample('180s').mean()


#%%##############################################################################
cpcf=pd.DataFrame()

cpcav=cpc_raw.resample('10s').mean() #averages of cpc raw
cpcwin = cpcav.loc[startdate:enddate]
dy_cpc = pd.Series(cpcwin['CPCconc'].diff())
dt_cps = pd.Series(cpcwin.index).diff().dt.seconds.values
mean_cpc_win = cpcwin.resample('180s').mean()

#%% Calculations
 # Calculate running mean and median of CPC values
rolling_mean_1min = cpcav.rolling(6).mean() # 6 = 60s rolling mean
rolling_mean = cpcav.rolling(t_rolmean).mean() # 18 = 180s rolling mean
rolling_mean_win_1min = rolling_mean.loc[startdate:enddate]
rolling_mean_win = rolling_mean.loc[startdate:enddate]
rolling_median_1min = cpcav.rolling(6).median() # 6 = 60s rolling median
rolling_median = cpcav.rolling(t_rolmedian).median() # 6 = 60s rolling median
rolling_median_win = rolling_median.loc[startdate:enddate]

quotient = cpcwin/rolling_mean_win
quotient_median = cpcwin/rolling_median_win
threshold = 3

flag=pd.DataFrame()
flag['cpc']=cpcwin['CPCconc'].copy()
flag['mean_values'] = rolling_mean_win['CPCconc'].copy()
flag['mean_quotient']=quotient['CPCconc']
flag['mean_flagged'] = np.where(flag['mean_quotient'] <= threshold, 1, 0)
# flag['change_colors']=flag['mean_flagged'].map({1:'Blue', 0:'Orange'})
flag['cpc_good']=flag['cpc'][flag['mean_flagged']==1]
flag['cpc_bad']=flag['cpc'][flag['mean_flagged']==0]

flag['median_values']= rolling_median_win['CPCconc'].copy()
flag['median_quotient']=quotient_median['CPCconc']
flag['median_flagged'] = np.where(flag['median_quotient'] <= threshold, 1, 0)
flag['cpc_good_median']=flag['cpc'][flag['median_flagged']==1]
flag['cpc_bad_median']=flag['cpc'][flag['median_flagged']==0]

#CPC Stats Dataframe
dt = pd.DataFrame()
dt['cpc']=cpcwin['CPCconc'].copy()
dt['dt_cpc']=dt['cpc'].
dy_cpc = pd.Series(cpcwin['CPCconc'].diff())
dt_cps = pd.Series(cpcwin.index).diff().dt.seconds.values



#%%##############################################################################
#PLOTS
#%%##########################################################################
## Set plot specs
line_width = 0.5
rcol = 'tab:green'
lcol= 'tab:blue'
fcol = 'tab:red'
zcol = 'tab:orange'
fs = 18 #fontsize for plots
ts = 16 #tichsize for plots
ms = 4 #markersize
#%% 
# Plot Histogram of cpc 
start_nov = pd.to_datetime('2019-11-01 00:00')
end_nov = pd.to_datetime('2019-12-01 00:00')
hist_nov = cpcav.loc[start_nov:end_nov]

start_hist = pd.to_datetime('2019-12-01 00:00')
end_hist = pd.to_datetime('2020-01-01 00:00')
hist_dez = cpcav.loc[start_hist:end_hist]

start_hist2 = pd.to_datetime('2020-01-01 00:00')
end_hist2 = pd.to_datetime('2020-02-01 00:00')
hist_jan = cpcav.loc[start_hist2:end_hist2]
fig, ax = plt.subplots(3,1)
ax1=ax[0]
ax2=ax[1]
ax3=ax[2]

ax1.hist(hist_nov.values, bins=100, log = True, range = (0,1000) )
ax1.set_title('Start: '+start_nov.strftime('%Y/%m/%d %H:%Mh')+'  End: '+end_nov.strftime('%Y/%m/%d %H:%Mh'))
ax1.set_ylabel('Frequency')
# ax1.set_xlabel('Concentrations')

ax2.hist(hist_dez.values, bins=100, log = True, range = (0,1000) )
ax2.set_title('Start: '+start_hist.strftime('%Y/%m/%d %H:%Mh')+'  End: '+end_hist.strftime('%Y/%m/%d %H:%Mh'))
ax2.set_ylabel('Frequency')
# ax2.set_xlabel('Concentrations')

ax3.hist(hist_jan.values, bins=100, log = True, range = (0,1000) )
ax3.set_title('Start: '+start_hist2.strftime('%Y/%m/%d %H:%Mh')+'  End: '+end_hist2.strftime('%Y/%m/%d %H:%Mh'))
ax3.set_ylabel('Frequency')
ax3.set_xlabel('CPC Concentrations')

#slope = pd.Series(np.gradient(tmp.values), tmp.index, name='slope')

#%%

    #Object-oriented interface
    # First create a grid of plots
    # ax will be an array of three Axes objects
    # Initialise a figure. subplots() with no args gives one plot.
    
# Plot cpc and wind
fig, ax = plt.subplots(2, sharex=True)

# Rename the axes for ease of use
ax1 = ax[0] #first plot axis
ax2 = ax[1]#second plot axis


# Call plot() method on the appropriate object
ax1.plot(reldir,'.',label='rel. wind direction', lw=line_width)
ax1.fill_between(reldir.index, dirmin, dirmax, alpha=0.2, label='polluted sector')
ax1.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'))
ax1.set_ylabel('Rel. wind direction [deg]')
ax1.set_ylim((0, 400))   # set the ylim to bottom, top
#ax1.set(ylabel='wind speed [m/s]', title = 'Pollution example')
#ax.xaxis.set_major_formatter(plt.NullFormatter())
ax1.legend(loc=2)

ax12 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax12.set_ylabel('wind speed [m/s]', color=rcol)
ax12.plot(wind, label = 'wind speed', color= rcol, lw=line_width)
#ax12.set(ylabel= 'Wind direction')  # we already handled the x-label with ax1
ax12.tick_params(axis='y', labelcolor=rcol)
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

"""
#%% New plot
fig, ax = plt.subplots()

# Call plot() method on the appropriate object
ax.plot(rolling_mean,'.',label='rolling mean', lw=line_width)
ax.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'))
ax.set_ylabel('rolling mean (1/cc)')
ax.set_yscale('log')
ax.set_ylim((0.1, 100000))   # set the ylim to bottom, top
#ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.legend(loc=2)

fig.autofmt_xdate()

plt.show()
"""
#%% New plot: CPC raw and median minus mean
fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize = ts)
ax.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'), fontsize = fs)
ax.set_xlabel('Time', color = lcol, fontsize=fs)


# Call plot() method on the appropriate object
ax.plot(cpcwin, '.' ,label='Raw CPC', lw=line_width, markersize = ms)
ax.set_ylabel('Concentration (1/cc)', color = lcol, fontsize=fs)
ax.tick_params(axis='y', labelcolor=lcol, labelsize = ts)
ax.set_yscale('log')
ax.set_ylim((1, 100000))   # set the ylim to bottom, top

ax12 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax12.plot(rolling_mean_win-rolling_median_win, label = '3min mean-median', color= rcol, lw=line_width)
ax12.set_ylabel('Concentration (1/cc)', color=rcol, fontsize=fs)
ax12.tick_params(axis='y', labelcolor=rcol, labelsize = ts)
ax12.legend(loc=1, fontsize = fs)
ax12.set_ylim((-200,200))

#ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.legend(loc=2, fontsize=fs)

fig.autofmt_xdate()

plt.show()

#%% New plot: CPC raw 
fig, ax = plt.subplots()


ax.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'))
ax.set_ylabel('raw cpc concentrations')
ax.set_xlabel('3min median')
# ax.set_yscale('log')
ax.set_ylim((1, 1000))   # set the ylim to bottom, top
ax.set_xlim((1, 1000))   # set the ylim to bottom, top
ax.legend(loc=2)

# Call plot() method on the appropriate object
ax.scatter(cpcwin,rolling_median_win, label='bs', lw=line_width)


fig.autofmt_xdate()

plt.show()


#%% Plot Histogram

fig, ax = plt.subplots(3,1)
ax1=ax[0]
ax2=ax[1]
ax3=ax[2]

ax1.hist(cpcwin.values, bins=100, log = True, range = (0,1000) )
ax1.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'))
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Raw CPC')
ax1.legend()

ax2.hist(rolling_mean_win.values, bins=100, log = True, range = (0,1000) )
ax2.set_ylabel('Frequency')
ax2.set_xlabel('3min Mean')
ax2.legend()

ax3.hist(rolling_median_win.values, bins=100, log = True, range = (0,1000) )
ax3.set_ylabel('Frequency')
ax3.set_xlabel('3min Median')
ax3.legend()
#%% New plot: CPC raw vs median vs mean
fig, ax = plt.subplots()
ax.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'), fontsize = fs)
ax.set_xlabel('Time', fontsize=fs)
ax.tick_params(axis='x', labelsize = ts)
ax.tick_params(axis='y', labelcolor= lcol, labelsize =ts)


# Call plot() method on the appropriate object
ax.plot(cpcwin ,'.',label='Raw CPC', lw=line_width, markersize = ms)
ax.plot(rolling_mean_win, label = '3min rolling mean', lw= line_width, markersize = ms)
ax.plot(rolling_median_win, label = '3min rolling median', lw= line_width, markersize = ms)
ax.set_ylabel('cpc concentration (1/cc)', color=lcol, fontsize=fs)
# ax.set_yscale('log')
ax.set_ylim((1, 10000))   # set the ylim to bottom, top
ax.legend(fontsize=fs)

#%% New plot: CPC raw and quotient

fig, ax = plt.subplots()
ax.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'), fontsize = fs)
ax.set_xlabel('Time', color = lcol, fontsize=fs)
ax.tick_params(axis='x', labelsize = fs)


# Call plot() method on the appropriate object
ax.plot(cpcwin ,'.',label='Raw CPC', lw=line_width, markersize = ms)
ax.set_ylabel('Concentration (1/cc)', color = lcol, fontsize = fs)
ax.set_ylim((1, 10000))   # set the ylim to bottom, top
ax.legend(loc=1, fontsize = fs)
ax.tick_params(axis='y', labelcolor= lcol, labelsize = fs)


ax12 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax12.set_ylabel('Ratio', color=rcol, fontsize = fs)
ax12.tick_params(axis='y', labelcolor=rcol, labelsize = ts)
# ax12.set_ylim(())
ax12.plot(quotient, label = 'Ratio raw CPC/3min rolling mean', color=rcol,lw= line_width, markersize = ms)
ax12.plot(quotient_median, label = 'Ratio raw CPC/3min rolling median', color=fcol, lw= line_width, markersize = ms)
ax12.legend(loc=2, fontsize = fs)

#%% New plot: rolling mean vs rolling median

fig, ax = plt.subplots()
ax.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'), fontsize = fs)
ax.tick_params(axis='x', labelsize = fs)
ax.tick_params(axis='y', labelsize = fs)

ax.scatter(rolling_mean_win,rolling_median_win, label='mean vs median', lw=line_width)
ax.set_xlabel('mean', fontsize=fs)
ax.set_ylabel('median', fontsize=fs)
ax.set_ylim((1,20000))
ax.set_xlim((1,20000))

#%% New plot: CPC raw with marker and quotient

fig, ax = plt.subplots(2,1)
ax1=ax[0]
ax2=ax[1]

ax1.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'), fontsize = fs)
ax1.tick_params(axis='x', labelsize = fs)

ax2.set_xlabel('Time', color = lcol, fontsize=fs)
ax2.tick_params(axis='x', labelsize = fs)


# Call plot() method on the appropriate object
ax1.plot(flag['cpc_good'] ,'.',label='cpc_clean', lw=line_width, markersize = ms)
ax1.plot(flag['cpc_bad'] ,'.',color='orange',label='cpc_polluted', lw=line_width, markersize = ms)
ax1.set_ylabel('Concentration (1/cc)', color = lcol, fontsize = fs)
ax1.set_ylim((1, 10000))   # set the ylim to bottom, top
ax1.legend(loc=1, fontsize = fs)
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor= lcol, labelsize = fs)

ax12 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax12.set_ylabel('Ratio', color=rcol, fontsize = fs)
ax12.set_ylim((0, 50))   # set the ylim to bottom, top
ax12.tick_params(axis='y', labelcolor=rcol, labelsize = ts)
# ax12.set_ylim(())
ax12.plot(flag['mean_quotient'], label = 'Ratio CPCraw/3min rolling mean', color=rcol,lw= line_width, markersize = ms)
# ax12.plot(quotient_median, label = 'Ratio raw CPC/3min rolling median', color=fcol, lw= line_width, markersize = ms)
ax12.legend(loc=2, fontsize = fs)

# Call plot() method on the appropriate object
ax2.plot(flag['cpc_good_median'] ,'.',label='cpc_clean', lw=line_width, markersize = ms)
ax2.plot(flag['cpc_bad_median'] ,'.',color='orange',label='cpc_polluted', lw=line_width, markersize = ms)
ax2.set_ylabel('Concentration (1/cc)', color = lcol, fontsize = fs)
ax2.set_ylim((1, 10000))   # set the ylim to bottom, top
ax2.legend(loc=1, fontsize = fs)
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor= lcol, labelsize = fs)

ax22 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
ax22.set_ylabel('Ratio', color=rcol, fontsize = fs)
ax22.set_ylim((0, 50))   # set the ylim to bottom, top
ax22.tick_params(axis='y', labelcolor=rcol, labelsize = ts)
# ax12.set_ylim(())
ax22.plot(flag['median_quotient'], label = 'Ratio CPC_raw/3min rolling median', color=rcol,lw= line_width, markersize = ms)
# ax12.plot(quotient_median, label = 'Ratio raw CPC/3min rolling median', color=fcol, lw= line_width, markersize = ms)
ax22.legend(loc=2, fontsize = fs)


#%% Plot of Filtered data and unfiltered data

fig, ax = plt.subplots(2)
ax1=ax[0]
ax2=ax[1]

ax1.hist(cpcwin.values, bins=100, log = True, range = (0,5000), alpha = 0.5, label = 'CPC_raw')
ax1.hist(flag['cpc_good'].values, bins=100, log = True, range = (0,5000), color='orange', alpha = 0.3, label='CPC_cleaned_mean' )
# ax1.hist(flag['cpc_good_median'].values, bins=100, log = True, range = (0,5000), color='yellow', alpha = 0.3, label='CPC_cleaned_median' )
ax1.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'), fontsize = fs)
ax1.set_ylabel('Frequency', fontsize = fs)
# ax1.set_xlabel('Concentrations', fontsize = fs)
ax1.tick_params(axis='x', labelsize = ts)
ax1.tick_params(axis='y', labelsize = ts)
ax1.legend(fontsize = ts)

ax2.hist(cpcwin.values, bins=100, log = True, range = (0,5000), alpha = 0.5, label = 'CPC_raw')
ax2.hist(flag['cpc_good_median'].values, bins=100, log = True, range = (0,5000), color='orange', alpha = 0.3, label='CPC_cleaned_median' )
# ax2.set_title('Start: '+startdate.strftime('%Y/%m/%d %H:%Mh')+'  End: '+enddate.strftime('%Y/%m/%d %H:%Mh'))
ax2.set_ylabel('Frequency', fontsize = fs)
ax2.set_xlabel('Concentrations', fontsize = fs)
ax2.tick_params(axis='x', labelsize = ts)
ax2.tick_params(axis='y', labelsize = ts)
ax2.legend(fontsize = ts)




