# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:32:03 2019

@author: beck_i
"""


 
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime


#LOAD AND PREPARE CPC DATA
#connect paths to CPC data
CPCpath = Path(r'C:\Users\beck_i\Documents\Python files\analysiscpcccncsmps\CPCtot')
cpc76file1 = CPCpath / 'CPC_DMA_Log190918.csv'
cpc76file2 = CPCpath / 'CPC_DMA_Log190919.csv'

#LOAD AND PREPARE CCNC DATA
#Connect path to CCNC data
CCNCpath = Path(r'C:\Users\beck_i\Documents\Python files\analysiscpcccncsmps')
CCNCfile = CCNCpath / 'CCNC_20190918.csv'

#LOAD AND PREPARE SMPS DATA
#Connect path to SMPS data
SMPSpath = Path(r'C:\Users\beck_i\Documents\Python files\analysiscpcccncsmps')
SMPSfile = SMPSpath / 'DistrTotCts_SMPS_2190918.csv'
SMPSall = SMPSpath / 'AllCounts_SMPS_20190918.csv'


#LOAD AND PREPARE PICARRO DATA
#Connect path to Picarro data
PICpath = Path(r'C:\Users\beck_i\Documents\Python files\analysiscpcccncsmps')
PICfile1 = PICpath / 'CFKADS2024-20190918-101443Z-DataLog_User.dat'
PICfile2 = PICpath / 'CFKADS2024-20190919-000012Z-DataLog_User.dat'
PICfile3 = PICpath / 'CFKADS2024-20190919-120315Z-DataLog_User.dat'

#***********************************************************************************

#create dataframe for CPC
cpc76data1 = pd.read_csv(cpc76file1, infer_datetime_format=True)
cpc76data2 = pd.read_csv(cpc76file2, infer_datetime_format=True)
#Concatenate dataframes
allcpcdata = pd.concat([cpc76data1, cpc76data2])

#Convert TimeEnd Strings from CPC DataFrame into datetime and put it into index column
allcpcdata.index = pd.to_datetime(allcpcdata.TimeEnd)
cpcconc=allcpcdata.CPCconc

#Create CCNC dataframe
ccncData = pd.read_csv(CCNCfile, infer_datetime_format=True) #load CCNC Data, exported from IGOR
#convert Time column from string to datetime
ccncData.TimeStart1 = pd.to_datetime(ccncData.TimeStart1)
#set index column to datetime 
ccncData.index = pd.to_datetime(ccncData.TimeStart1)

#Extract all 1%SS data from CCNC dataframe
CCNC = ccncData[(ccncData.CurrentSS1==1)]['CCNNumberConc1'].copy()
#ccncData_sort=ccncData.sort_values(by=['CurrentSS1'])

#Create SMPS dataframe
SMPSdata = pd.read_csv(SMPSfile, infer_datetime_format=True) #load SMPS Data, exported from IGOR
#convert Time column from string to datetime
SMPSdata.timestart = pd.to_datetime(SMPSdata.timestart)
#set index column to datetime 
SMPSdata.index = pd.to_datetime(SMPSdata.timestart)
SMPS=SMPSdata.DistrTotCts

#Load All SMPS Spectrum Data
SMPallSpec = pd.read_csv(SMPSall, infer_datetime_format=True) #load SMPS Data, exported from IGOR

#SMPS: Integrate over a row, select which diameters you want to take into account. 
#SMPallSpec.apply(lambda x: np.trapz(x[dP>40],np.log10(Dp[dp>40])),axis=1) #This command might be with some errors, still need to check it, got it from Andrea

#Create Picarro dataframe
PICdata1 = pd.read_csv(PICfile1, sep = '\s+', parse_dates=[['DATE','TIME']], infer_datetime_format=True) #load Picarro Data
PICdata2 = pd.read_csv(PICfile2, sep = '\s+', parse_dates=[['DATE','TIME']], infer_datetime_format=True) #load Picarro Data
PICdata3 = pd.read_csv(PICfile3, sep = '\s+', parse_dates=[['DATE','TIME']], infer_datetime_format=True) #load Picarro Data
#Concatenate picarro dataframes
allPICdata = pd.concat([PICdata1, PICdata2, PICdata3]) #merge several dataframes
allPICdata.index=pd.to_datetime(allPICdata.DATE_TIME)
#Extract CO2 data from Picarro data

CO2all = allPICdata['CO2']

#Pick Date Window
#CCNC[((CCNC.index>=)&())] #use pd.to_datetime(args)
#**********************************************************************************************

#PLOT DATA
#print(cpc76data1.head())
fig=plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2, sharex = ax1)
#fig,(ax1, ax2) = subplots(2, sharex=True)
ax1.plot(CCNC,"x-" ) 
ax1.plot(cpcconc) #plot cpc data into same fiugre
ax1.plot(SMPS)
#plt.yscale('log')
ax1.set_yscale('log')
##plt.xticks(np.arange(min(timeall), max(timeall), 10))

#Another possibility to plot: 
#pivot() #check it out in tutorial. 

plt.xlabel('Time')
#plt.ylabel('Concentration')
ax1.set_ylabel('Concentration')
ax1.legend()
ax1.set_ylim(10,20000)

ax2.plot(CO2all, "-")
ax2.legend()
ax2.set_ylabel('ppm')
ax2.set_ylim(350, 700)
plt.show()

