# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:59:27 2020

@author: beck_i

Purpose: 
Read the CCNC hdf file
Plot CCNC Number concentrations for a certain SS

Changelog: 

"""

import csv
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO
import time

start = time.time()

#%%START=====================================================================
#Close all existing figures:
plt.close("all") #this is the same as: .matplotlib.pyplot.close("all")
plt.style.use('classic')
#==========================================================================


#%% LOAD DATA
file =r'D:\All_Data\CCNC\MOSAiC_CCNC_all_new_test.hdf'
CCNCdat = pd.read_hdf(file)
## Set intey of dataframe to time strings
#CCNCdat.set_index('Time', inplace=True)
#CCNCdat.index=pd.to_datetime(CCNCdat.index,infer_datetime_format=True)

ss1 = CCNCdat['CCN Number Conc'][CCNCdat['SS1']==1].resample('60s').mean()

#%% Figure
plt.style.use('seaborn-whitegrid')

# Initialise a figure. subplots() with no args gives one plot.
fig, ax = plt.subplots()
ax.plot(ss1,'b.', label='CCNC SS1') #,linewidth=1


## Plot Line2 (Right Y Axis)
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#ax2.plot(x, y2, color='tab:blue')

# Decorations
# ax1 (left Y axis)
ax.set_yscale('log')
ax.set_xlabel('Time', fontsize=15)
ax.tick_params(axis='x', rotation=0, labelsize=12)
ax.set_ylabel('Concentration', fontsize=15) #, color='tab:blue'
ax.tick_params(axis='y', rotation=0  ) #,labelcolor='tab:blue'
ax.grid(alpha=.4)
leg = ax.legend()

#
#fig =plt.figure()
#ax = plt.axes()
#
#ax.plot(ss1, '.', label='CCNC 1%')
#ax.set(xlabel='Time', ylabel='concentration []', title='CCNC ') #xlim=(,), ylim=(,), xlim=('2020-02-11','2020-02-13')
#ax.legend()

#plt.plot(LundCCNC['Time'],LundCCNC['CCN Number Conc'],'.')
#lines = plt.plot(LundCCNCstat.loc(axis=1)[:,['Central_time']],
#                 LundCCNCstat.loc(axis=1)[:,['Mean']],'o')
#plt.legend(lines,LundCCNCstat.columns.levels[0])

end = time.time()
print(end - start)