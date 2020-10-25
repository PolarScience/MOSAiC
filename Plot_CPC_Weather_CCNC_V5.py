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
    22.8. Adapted this script to HDF files. Plots for EAC. Works fine. 
    Exports good/bad dates. 
    - Pollution detection based on gradient. Found exponential function for line in log-log space to separate data. 
    - 
        
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


ccncdata=CCNCdat.loc[datastart:dataend] #contains CCNC Data for time period: Time, SS0.15, SS0.2, SS0.3, SS0.5, SS1, and NumberConc 
ccncdata=ccncdata.resample('60s').mean()
ccncdata['SS1']=ccncdata['CCN Number Conc'][ccncdata['SS1'].notna()]
ccncdata['SS0.5']=ccncdata['CCN Number Conc'][ccncdata['SS0.5'].notna()]
ccncdata['SS0.3']=ccncdata['CCN Number Conc'][ccncdata['SS0.3'].notna()]
ccncdata['SS0.2']=ccncdata['CCN Number Conc'][ccncdata['SS0.2'].notna()]
ccncdata['SS0.15']=ccncdata['CCN Number Conc'][ccncdata['SS0.15'].notna()]
ccncdata['good']=df_1min['good']
ccncdata=ccncdata[ccncdata['good'].notna()] #Keep only good data

ss1 = ccncdata['SS1']
ss03 = ccncdata['SS0.3']


    #Write good data to CSV file
# df_1min[['good','bad']].to_csv(os.path.join(outpath, outfile_csv))


#%% PLOTS

## Set plot specs
dirmin = 75 #wind window min
dirmax = 270 #wind window max

line_width = 0.8
rcol = 'tab:green'
lcol= 'tab:blue'
fcol = 'tab:red'
zcol = 'tab:red'
fs = 18 #fontsize for plots Title and X and Y Label
ts = 16 #tichsize for plots Legends, and ticks
ms = 3 #markersize
s = 1

##======================================
## Plot Wind direction vs particle number concentration with colored dots for wind speed
# fig, ax = plt.subplots()
# im=ax.scatter(df_weather['winddir'],cpc_1min['CPCconc'], s=s, c=df_weather['windvel'], cmap='jet')
# # Add a colorbar
# cbar=fig.colorbar(im, ax=ax)
# cbar.set_label('Wind speed [m/s]', fontsize =fs)
# ax.set_title('Particles, wind direction and speed '+datastart.strftime('%Y/%m/%d')+' - '+dataend.strftime('%Y/%m/%d'), fontsize = fs)
# ax.set_xlabel('Rel. wind direction [°]', fontsize=fs)
# ax.tick_params(axis='x', labelsize = ts)
# ax.tick_params(axis='y', labelsize =ts)
# ax.set_ylabel('Particle concentration [1/cc]', fontsize=fs)
# ax.set_yscale('log')
# # ax.set_xlim(0,360)

#============================================================
# Plot Wind direction , velocity and Particle concentrations
# fig, ax = plt.subplots(2, sharex=True)
# # window_start = 
# # window_end =
# winddir= df_weather['winddir'].loc['2019-12-28 00:00':'2019-12-28 23:59']
# windvel=df_weather['windvel'].loc['2019-12-28 00:00':'2019-12-28 23:59']
# concentration = cpc_1min['CPCconc'].loc['2019-12-28 00:00':'2019-12-28 23:59']

# # Rename the axes for ease of use
# ax1 = ax[0] #first plot axis
# ax2 = ax[1]#second plot axis

# # Call plot() method on the appropriate object
# ax1.plot(winddir,'.',label='relative wind direction',markersize = 1,color = lcol, lw=line_width)
# ax1.fill_between(winddir.index, dirmin, dirmax, alpha=0.2, label='polluted sector')
# ax1.set_title('Pollution from Ships exhaust '+datastart.strftime('%Y/%m/%d')+'  End: '+dataend.strftime('%Y/%m/%d'), fontsize=fs)
# ax1.set_ylabel('Rel. wind direction [°]', color = lcol,  fontsize=fs)
# ax1.tick_params(axis='y', labelsize =ts, labelcolor = lcol)
# ax1.set_ylim((0, 400))   # set the ylim to bottom, top
# #ax1.set(ylabel='wind speed [m/s]', title = 'Pollution example')
# #ax.xaxis.set_major_formatter(plt.NullFormatter())
# ax1.legend(loc=2)

# ax12 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax12.set_ylabel('wind speed [m/s]', color=rcol,fontsize=fs)
# ax12.plot(windvel, label = 'rel. wind speed', color= rcol, lw=line_width)
# #ax12.set(ylabel= 'Wind direction')  # we already handled the x-label with ax1
# ax1.tick_params(axis='x', labelsize = ts)
# ax12.tick_params(axis='y', labelsize =ts, labelcolor = rcol)
# ax12.legend(loc=1)

# ax2.plot(concentration,'.',markersize=1, label='CPC3776')
# ax2.set_ylabel('Particle concentration [1/cc]', color=lcol, fontsize=fs)
# ax2.set_yscale('log')
# ax2.tick_params(axis='y', labelsize =ts, labelcolor = lcol)
# ax2.tick_params(axis='x', labelsize =ts)
# ax2.set_xlabel('Time', fontsize=fs)
# ax2.legend()

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# fig.autofmt_xdate()
# plt.show()


##=========================================================================
    #Plot Gradient of Number concentration vs. Number concentration with straight line
    # scatter plot of mean gradient (dN/dt) vs Number concentration N 

# fig, ax = plt.subplots()
# im=ax.scatter(df_1min['counts'],df_1min['gradient'],s=s,c=df_weather['winddir'], cmap='jet')
#     ## Create line in log-loc scale: y = a*x**m
# x= np.linspace(1,10000,100)
# y=a*(x**m)
# ax.plot(x,y, color='red', lw = 2)
# ax.set_title('Gradient of N vs N', fontsize = fs)
# cbar=fig.colorbar(im, ax=ax)
# cbar.set_label('Rel. Wind direction [°]', fontsize =fs)
# ax.set_xlabel('N [1/cc]', fontsize=fs)
# ax.tick_params(axis='x', labelsize = ts)
# ax.tick_params(axis='y', labelsize =ts)
# ax.set_ylabel('gradN', fontsize=fs)
# ax.set_yscale('log')
# ax.set_xscale('log')


##================================================================================== 
   #Plot NORMALIZED Gradient of Number concentration vs. Number concentration
    #Plot scatter plot of mean gradient (dN/dt) vs Number concentration N 
# fig, ax = plt.subplots()
# im=ax.scatter(df_1min['counts'],df_1min['normalized'],s=s,c=df_weather['winddir'], cmap='jet')
# ax.set_title('Normalized gradient of N vs N', fontsize = fs)
# cbar=fig.colorbar(im, ax=ax)
# cbar.set_label('Rel. Wind direction [°]', fontsize =fs)
# ax.set_xlabel('N [1/cc]', fontsize=fs)
# ax.tick_params(axis='x', labelsize = ts)
# ax.tick_params(axis='y', labelsize =ts)
# ax.set_ylabel('gradN', fontsize=fs)
# ax.set_yscale('log')
# ax.set_xscale('log')


##======================================================================
    ## TEST PLOT OF GOOD OR BAD DATA IN THE SCATTER PLOT ==> check which data is excluded
# fig, ax = plt.subplots()
# im=ax.scatter(df_1min['badwind'],df_1min['gradient'],s=s,c=df_weather['winddir'], cmap='jet')
# ax.set_title('Normalized gradient of N vs N', fontsize = fs)
# cbar=fig.colorbar(im, ax=ax)
# cbar.set_label('Rel. Wind direction [°]', fontsize =fs)
# ax.set_xlabel('N [1/cc]', fontsize=fs)
# ax.tick_params(axis='x', labelsize = ts)
# ax.tick_params(axis='y', labelsize =ts)
# ax.set_ylabel('gradN', fontsize=fs)
# ax.set_yscale('log')
# ax.set_xscale('log')
##===========================================================================
#%% Histogram of gradients in log scale
# fig, ax = plt.subplots()
# ax.hist(np.log(df_1min['gradient'][df_1min['gradient']>0]), bins = 100, alpha = 0.3, label = 'gradient of particle concentrations')
# # ax.hist(dt['cpc_good'].values, bins=100, log = True, range = (0,5000), color='orange', alpha = 0.3, label='CPC_cleaned' )
# # ax1.hist(flag['cpc_good_median'].values, bins=100, log = True, range = (0,5000), color='yellow', alpha = 0.3, label='CPC_cleaned_median' )
# ax.set_title('Distribution of log(gradN)', fontsize = fs)
# ax.set_ylabel('Frequency', fontsize = fs)
# ax.set_xlabel('log(grad N)', fontsize = fs)
# ax.tick_params(axis='x', labelsize = ts)
# ax.tick_params(axis='y', labelsize = ts)
# ax.legend(fontsize = ts)

##=================================================================================
    # Plot Histogram of all Concentrations together with filtered concentrations
    # Plot of CPC Filtered data and unfiltered data (all vs clean)

fig, ax = plt.subplots()
ax1=ax
ax1.hist(df_1min['counts'], log = True, bins=np.logspace(0,4,num=100), alpha = 0.5, label = 'CPC_raw')# bins=100, range = (0,50000)
ax1.hist(df_1min['good'], log = True, bins=np.logspace(0,4,num=100), color='orange', alpha = 0.3, label='CPC_cleaned' ) #bins=np.logspace(0,4,num=100)
# ax1.hist(flag['cpc_good_median'].values, bins=100, log = True, range = (0,5000), color='yellow', alpha = 0.3, label='CPC_cleaned_median' )
ax1.set_title('Distribution of raw and cleaned data', fontsize = fs)
ax1.set_ylabel('Frequency', fontsize = fs)
# ax1.set_xlabel('Concentrations', fontsize = fs)
ax1.tick_params(axis='x', labelsize = ts)
ax1.tick_params(axis='y', labelsize = ts)
ax1.set_xscale('log')
ax1.legend(fontsize = ts)
ax1.set_xlabel('Number concentration N [1/cc]', fontsize = fs)

#=================================================================================
    ## PLOT TIME SERIES OF ALL DATA WITH FILTERED DATA MARKED IN RED
    ##Plot Good and bad CPC Data over Time 

# fig, ax1 = plt.subplots()
# ax1.plot(df_1min['bad'], '.',label = 'Bad data', lw= line_width,color=zcol, markersize = ms)
# ax1.plot(df_1min['good'], '.',label = 'Good data', lw= line_width, markersize = ms)
# ax1.set_title('Time Series of clean and polluted data', fontsize=fs)
# ax1.set_xlabel('Time', fontsize=fs)
# ax1.tick_params(axis='x', labelsize = ts)
# ax1.tick_params(axis='y', labelcolor= lcol, labelsize =ts)
# ax1.set_ylabel('Concentration N [1/cc]', color=lcol, fontsize=fs)
# ax1.set_yscale('log')
# ax1.set_ylim((1, 100000))   # set the ylim to bottom, top
# ax1.legend(loc=2, fontsize=fs)
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# fig.autofmt_xdate()

#=================================================================================
    ## PLOT TIME SERIES OF GOOD DATA ONLY

# fig, ax1 = plt.subplots()
# ax1.plot(df_1min['good'], '.',label = 'cleaned CPC data', lw= line_width, markersize = ms)
# ax1.set_title('Time Series of Particle concentrations', fontsize=fs)
# ax1.set_xlabel('Time', fontsize=fs)
# ax1.tick_params(axis='x', labelsize = ts)
# ax1.tick_params(axis='y', labelcolor= lcol, labelsize =ts)
# ax1.set_ylabel('Concentration N [1/cc]', color=lcol, fontsize=fs)
# ax1.set_yscale('log')
# ax1.set_ylim((1, 100000))   # set the ylim to bottom, top
# ax1.legend(loc=2, fontsize=fs)
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# fig.autofmt_xdate()

#=====================================================================================
    ## PLOT CCNC DATA AS A TIME SERIES
# # Initialise a figure. subplots() with no args gives one plot.
# fig, ax = plt.subplots()
# ax.plot(ss1,'b.', label='CCNC SS1') #,linewidth=1
# ax.plot(ss03,'g.', label='CCNC SS0.3') #,linewidth=1
# ax.plot(df_1min['good'],'r.')
# ax.set_yscale('log')
# ax.set_xlabel('Time', fontsize=15)
# ax.tick_params(axis='x', rotation=0, labelsize=12)
# ax.set_ylabel('Concentration', fontsize=15) #, color='tab:blue'
# ax.tick_params(axis='y', rotation=0  ) #,labelcolor='tab:blue'
# ax.grid(alpha=.4)
# leg = ax.legend()


#=========================================================================================
    ## PLOT CCNC VS CPC FOR ONE WEEK OR OTHER DEFINED PERIOD
ccncstart = pd.to_datetime('2019-09-28 00:00')
ccncend = pd.to_datetime('2019-12-10 00:00')

x=ss1.loc[ccncstart:ccncend] #.loc[ccncstart:ccncend]
y=ss03.loc[ccncstart:ccncend] #.loc[ccncstart:ccncend]
x=x.dropna()
z=df_1min['good'].loc[ccncstart:ccncend] #.loc[ccncstart:ccncend]

#=========================================================================================
    ##Time series of SS1, SS0.3 and CPC
fig, ax = plt.subplots()
ax.set_title('Time Series of Total Particle concentrations and CCN', fontsize=fs)
ax.plot(x, 'g.', label= 'SS1', markersize = ms)
ax.plot(y,'b.', label = 'SS0.3', markersize = ms)
ax.plot(z, 'y.', label = 'Total concentration (CPC)', markersize = ms)
ax.set_ylabel('Concentration [1/cc]', fontsize=fs) #, color='tab:blue'
ax.set_xlabel('Time', fontsize=fs)
ax.tick_params(axis='x', labelsize = ts)
ax.tick_params(axis='y', labelsize =ts)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax.set_yscale('log')
ax.legend()

#=============================================================================
    ## TODO: Time series of Ratio CCNC/CPC over time and in a scatter plot with CPC data 

ratio1 = ss1/df_1min['good']
ratio03 = ss03/df_1min['good']

    ## Initialise a figure. subplots() with no args gives one plot.    
fig, ax = plt.subplots()
ax.plot(ratio1,'.', label='Activation Ratio SS1', markersize=ms) #,linewidth=1
ax.plot(ratio03,'.', label='Activation Ratio SS0.3', markersize=ms) #,linewidth=1
# ax.set_yscale('log')
ax.set_ylim(0,10)
ax.set_xlabel('Time', fontsize=15)
ax.tick_params(axis='x', rotation=45, labelsize=12)
ax.set_ylabel('Ratio', fontsize=15) #, color='tab:blue'
ax.tick_params(axis='y', rotation=0  ) #,labelcolor='tab:blue'
ax.grid(alpha=.4)
leg = ax.legend()

#===========================================================================
    ## Scatter Plot of Activation ratio vs CPC Concentration

ratio = pd.DataFrame()
ratio['rss1']=ccncdata['SS1']/df_1min['good']
# ratio['rss03']=ratio1.resample('600s').mean()

ratio1_10min = ratio1.resample('600s').mean()
ratio03_10min = ratio03.resample('600s').mean()
cpc_10min = df_1min['good'].resample('600s').mean()

fig, ax = plt.subplots()
im=ax.scatter(cpc_10min,ratio1_10min,s=s, alpha = 0.9, label='SS1')
im2=ax.scatter(cpc_10min,ratio03_10min, s=s, alpha = 0.7, label = 'SS0.3')
# plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})      #Size of the plot
ax.set_title('Activation ratio vs Number Concentration', fontsize = fs)
# cbar=fig.colorbar(im, ax=ax)
# cbar.set_label('Rel. Wind direction [°]', fontsize =fs)
ax.set_xlabel('N [1/cc]', fontsize=fs)
ax.tick_params(axis='x', labelsize = ts, rotation = 0)
ax.tick_params(axis='y', labelsize =ts, rotation = 0)
ax.set_ylabel('ratio', fontsize=fs)
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_xlim(1,10000)
ax.set_ylim(0.1,1.1)
leg = ax.legend()

#===========================================================================
    ## Timeseries of activation ratio > 0.8




#=============================================================================
    ## Histogram of ratio of all points <10p, <100p, <500p , 1000p
    
# ratio_10 = df_1min['good'][df_1min['good'] < 10]/


#=====================================================================
    ##Histogram of SS and CPC with density function
# fig, ax = plt.subplots()
# ax.set_title('Hist and KDE of Total Particle concentrations and CCN', fontsize=fs)
# sns.distplot(x,kde=True, hist= True, bins=100, color="g", ax=ax, label='SS1')
# sns.distplot(y,kde=True, hist= True, bins=100, color="b", ax=ax, label = 'SS0.3')
# sns.distplot(z,kde=True, hist= True, bins=100, color="y", ax=ax, label = 'CPC')
# ax.legend()
# ax.set_xlim(1,1000)
# ax.set_xscale('log')

# #================================================================================
    ## Histogram Plot of CCNC and CPC
# fig, ax = plt.subplots()
# ax.hist(x, log = False, bins = 100,color="g", label='SS1')
# ax.hist(y, log = True, bins = 1000,color="b",label='SS0.3')
# # ax.hist(y, log = True, bins = 1000,color="y",label='CPC')
# ax.legend()
# ax.set_xlim(1,1000)

# #================================================================================
# fig, ax = plt.subplots()
# ax = sns.distplot(x,kde=True, bins=100, color='skyblue', hist_kws={"linewidth": 15,'alpha':1})
# ax = sns.distplot(z,kde=True, bins=100, color='skyblue', hist_kws={"linewidth": 15,'alpha':1})

# ax.set(xlabel='Gamma Distribution', ylabel='Frequency')
# ax.set_xscale('log')

#==========================================================
    ## New try of histogram
    
# Usual histogram plot
fig = plt.figure()
ax1 = fig.add_subplot(121)
nx, binsx, patchesx = ax1.hist(x, bins=np.logspace(0,3,num=20), alpha = 0.5, density=True)  # output is two arrays
ny, binsy, patchesy = ax1.hist(y, bins=np.logspace(0,3, num=20),alpha = 0.5, density=True)  # output is two arrays
nz, binsz, patchesz = ax1.hist(z, bins=np.logspace(0,3, num=20),alpha = 0.5, density=True)  # output is two arrays

# Scatter plot
# Now we find the center of each bin from the bin edges
bins_mean = [0.5 * (binsx[i] + binsx[i+1]) for i in range(len(nx))]
ax2 = fig.add_subplot(122)
ax2.scatter(bins_mean, nx, label='SS1')
ax2.scatter(bins_mean, ny, label='SS0.3')
ax2.scatter(bins_mean, nz, label='CPC')
ax2.set_xscale('log')
ax2.legend()
ax2.set_xlim(1,1000)
ax2.set_yscale('log')

#==========================================================
    ## New try of histogram
    ## Usual histogram plot
fig, ax = plt.subplots()
# Scatter plot
ax.scatter(bins_mean, nx, label='SS1')
ax.scatter(bins_mean, ny, label='SS0.3')
ax.scatter(bins_mean, nz, label='CPC')
ax.set_xscale('log')
ax.legend()
ax.set_xlim(1,1000)
ax.set_yscale('log')

#==========================================================
    ## Same Histogram using numpy
    
# # Usual histogram plot
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# nx, binsx = np.histogram(x, bins=100, range=(1,1000) )  # output is two arrays
# ny, binsy = np.histogram(y, bins=100, range =(1,1000))  # output is two arrays
# nz, binsz= np.histogram(z, bins=100, range = (1,1000))  # output is two arrays
# ax1.hist(x,100)

# # Scatter plot
# # Now we find the center of each bin from the bin edges
# bins_mean = [0.5 * (binsx[i] + binsx[i+1]) for i in range(len(nx))]
# ax2 = fig.add_subplot(122)
# ax2.scatter(bins_mean, nx, label='SS1')
# ax2.scatter(bins_mean, ny, label='SS0.3')
# ax2.scatter(bins_mean, nz, label='CPC')
# ax2.set_xscale('log')
# ax2.legend()
# ax2.set_xlim(1,1000)
# ax2.set_yscale('log')


end = time.time()
print(end - start)
