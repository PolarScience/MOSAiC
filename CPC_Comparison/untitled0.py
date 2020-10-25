# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:04:17 2020

@author: beck_i
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 08:51:57 2020

@author: beck_i


CPC comparison 200202
"""


import csv
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO


#%%START=====================================================================
#Close all existing figures:
plt.close("all") #this is the same as: .atplotlib.pyplot.close("all")
plt.style.use('classic')
#==========================================================================

#%% LOAD Function
def Load_CPC(datapath):
    fdir = Path(datapath)
    files = fdir.rglob('CPC_DMA_Log2002*.csv' ) #help(Path.glob)
    df=pd.DataFrame() # initialize dataframe
    
    for j in files:
        df_temp=pd.read_csv(j, usecols=[0,1,17]) #
        df=df.append(df_temp)

    # Sort by datetime index
    df.index=pd.to_datetime(df.TimeEnd, dayfirst=True) #Replace Index of df by time
    df.sort_index(inplace=True) #sort index by datetime
    return(df)


#%% LOAD DATA
    
dir3776 =r'F:\Data_Analysis\CPC_Comparison\CPC_3776_Calibrations/'
data3776 = Load_CPC(dir3776)

dir3025 = r'F:\Data_Analysis\CPC_Comparison\CPC_3025_calibrations/'
data3025 = Load_CPC(dir3025)

dir3010 = r'F:\Data_Analysis\CPC_Comparison\cpc_3010_calibration/'
data3010 = Load_CPC(dir3010)


#%% Preprocess data

conc76 = data3776.CPCconc
mean76 = conc76.resample('30s').mean() 

conc25 = data3025.CPCconc
mean25 = conc25.resample('30s').mean()

conc10 = data3010.CPCconc
mean10 = conc10.resample('30s').mean()

ratio2576=mean25/mean76


# Set Time limits
startdate = min(mean25.index)
enddate = max(mean25.index)
stdate = min(mean25.index).strftime('%Y-%m-%d')
endate = max(mean25.index).strftime('%Y-%m-%d')
#ticks = np.linspace(startdate,enddate,50)


#%% PLOT
#========================================================================================
fig=plt.figure()
ax = plt.axes()
# First create a grid of plots
# ax will be an array of two Axes objects
#fig, ax = plt.subplots(2)
#Call plot() method on the appropriate object
ax.plot(mean76, label='CPC3776')
ax.plot(mean25, label='CPC3025')
ax.plot(mean10, label='CPC3010')

ax.set(ylim=(0, 20000), xlim=['2020-02-02', '2020-02-03'],xlabel='Time', ylabel='Concentration p/cc', title='CPC Copmarison 20200202') #xlim=['2020-01-20', '2020-01-24']
xticks = ax.get_xticks()

#ax[1].plot();
leg = ax.legend()

fig=plt.figure()
ax=plt.axes()
ax.plot(ratio2576, label = 'CPC3025/CPC3776')
ax.set( title='CPC comparison ratio3025/3776')
ax.legend()

"""
#%% Another plot
#Object-oriented interface
# First create a grid of plots
# ax will be an array of three Axes objects
#fig=plt.figure()
fig, ax = plt.subplots(4, sharex=True)

# Call plot() method on the appropriate object
ax[0].plot(mean76, label='CPC3776')
ax[0].plot(mean25, label='CPC3025')
ax[0].plot(mean10, label='CPC3010')
ax[0].set(ylabel='Concentrations [p/cc]', ylim = (0,20000),title='CPC Comparison Data 200202-200204') #xlim=(,), ylim=(,)
#ax.xaxis.set_major_formatter(plt.NullFormatter())
ax[0].legend();
ax[1].plot(ratio2576,label='Ratio 3776/3025');
#ax[1].set(ylabel='Relative Humidity [%]')
#print(dir(ax[1].set)); #help(mpl.axes.Axes.set)
ax[1].legend()
ax[2].plot(label='CPC3025 Interstitial - Total inlet ');
ax[2].set(ylabel='Concentrations [p/cc]', xlabel='Time')
ax[2].legend()
#ax[0].xaxis.set_major_locator(plt.MaxNLocator(3))
#myx2 = ax[2].get_xaxis()
#myx2.set_transform() #help(mpl.axis.Axis.set_transform)
#ax[0].get_xaxis().set_visible(False) # Hide Ticks
"""

#%% Next
dir3025 =r'F:\Data_Analysis\CPC_Comparison\CPC3025_containerair/'
contdata3025 = Load_CPC(dir3025)

dir3776 = r'F:\Data_Analysis\CPC_Comparison\CPC3776_containerair/'
contdata3776 = Load_CPC(dir3776)

#%% Preprocess data

contconc76 = contdata3776.CPCconc
contmean76 = contconc76.resample('30s').mean() 

contconc25 = contdata3025.CPCconc
contmean25 = contconc25.resample('30s').mean()

contratio2576=contmean25/contmean76

#%% PLOT
#========================================================================================
fig=plt.figure()
ax = plt.axes()
# First create a grid of plots
# ax will be an array of two Axes objects
#fig, ax = plt.subplots(2)
#Call plot() method on the appropriate object
ax.plot(contmean76, label='CPC3776')
ax.plot(contmean25, label='CPC3025')

ax.set(ylim=(0, 1000),xlabel='Time', ylabel='Concentration p/cc', title='CPC Container 200214') #xlim=['2020-01-20', '2020-01-24']
xticks = ax.get_xticks()

#ax[1].plot();
leg = ax.legend()

fig=plt.figure()
ax=plt.axes()
ax.plot(contratio2576, label = 'CPC3025/CPC3776')
ax.set( title='CPC container air ratio3025/3776')
ax.legend()



