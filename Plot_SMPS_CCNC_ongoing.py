# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:10:39 2020

@author: beck_i
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
plt.close("all") #this is the same as: .matplotlib.pyplot.close("all")
plt.style.use('classic')
#==========================================================================


#%% LOAD DATA
#%% LOAD SMPS DATA
file = r'F:\All_Data\SMPS\190919-200226_SMPS_totalcounts.csv'
SMPSdata = pd.read_csv(file)

SMPSdata.set_index('timecentre1', inplace=True)

#%% Figure
plt.style.use('seaborn-whitegrid')

# Initialise a figure. subplots() with no args gives one plot.
fig, ax = plt.subplots(2)
ax.plot(SMPSdata,'b.', label='SMPS Total') #,linewidth=1


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