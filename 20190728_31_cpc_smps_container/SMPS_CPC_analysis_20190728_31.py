# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:14:16 2019

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
CPCpath = Path(r'C:\Users\beck_i\Documents\Python files\20190728_31_cpc_smps_container\CPC3776')
cpc76file1 = CPCpath / 'CPC_DMA_Log190729.csv'
cpc76file2 = CPCpath / 'CPC_DMA_Log190730.csv'
cpc76file3 = CPCpath / 'CPC_DMA_Log190731.csv'



#LOAD AND PREPARE SMPS DATA
#Connect path to SMPS data
SMPSpath = Path(r'C:\Users\beck_i\Documents\Python files\20190728_31_cpc_smps_container')
SMPSfile = SMPSpath / 'DistrTotCts_SMPS_2190918.csv'
SMPSall = SMPSpath / 'AllCounts_SMPS_20190918.csv'



#***********************************************************************************

#create dataframe for CPC
cpc76data1 = pd.read_csv(cpc76file1, infer_datetime_format=True)
cpc76data2 = pd.read_csv(cpc76file2, infer_datetime_format=True)
cpc76data3 = pd.read_csv(cpc76file3, infer_datetime_format=True)

#Concatenate dataframes
allcpcdata = pd.concat([cpc76data1, cpc76data2, cpc76data3])

#Convert TimeEnd Strings from CPC DataFrame into datetime and put it into index column
allcpcdata.index = pd.to_datetime(allcpcdata.TimeEnd)
cpcconc=allcpcdata.CPCconc


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


#Pick Date Window
#CCNC[((CCNC.index>=)&())] #use pd.to_datetime(args)
#**********************************************************************************************

#PLOT DATA
#print(cpc76data1.head())
fig=plt.figure()
ax1 = fig.add_subplot(2,1,1)
#fig,(ax1, ax2) = subplots(2, sharex=True)
ax1.plot(SMPS,"x-" ) 
ax1.plot(cpcconc) #plot cpc data into same fiugre
#plt.yscale('log')
ax1.set_yscale('log')
##plt.xticks(np.arange(min(timeall), max(timeall), 10))

plt.xlabel('Time')
#plt.ylabel('Concentration')
ax1.set_ylabel('Concentration')
ax1.legend()
ax1.set_ylim(10,20000)

plt.show()

