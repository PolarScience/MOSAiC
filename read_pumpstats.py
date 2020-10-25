# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:15:52 2020

@author: ivo
"""


import csv
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


#%%START=====================================================================
#Close all existing figures:
plt.close("all") #this is the same as: .atplotlib.pyplot.close("all")
plt.style.use('seaborn-white')
#==========================================================================
path = r'D:\MOSAiC_Expedition\Tuija AMS Data\PumpData_pump2_fixing.txt'
pumps = pd.read_csv(path, sep = ' ', skiprows= 0)