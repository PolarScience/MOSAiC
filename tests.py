# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:07:04 2020

@author: ivo
"""

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
import pylab as pl
#import ruptures as rpt
#import changefinder as cf
sns.set()


#%%START=====================================================================
#Close all existing figures:
plt.close("all") #this is the same as: .atplotlib.pyplot.close("all")
# plt.style.use('seaborn-white')
#==========================================================================


data = np.random.normal(size=10000)

fig, ax = plt.subplots()
pl.hist(data, bins=np.logspace(np.log10(0.1),np.log10(1.0), 50))
pl.gca().set_xscale("log")
pl.show()

fig, ax = plt.subplots()
pl.hist(data, 50)