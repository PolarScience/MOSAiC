# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:22:50 2019

@author: beck_i
"""

 
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

#START=====================================================================
#Close all existing figures:
plt.close("all") #this is the same as: .atplotlib.pyplot.close("all")
#==========================================================================

#__________________________________________________________________________
#connect paths to data
datapath = Path('C:/Users/beck_i/Documents/Python files/Inlet Characterization/Calibrations')
cpc76_file1 = datapath / 'InletCharakterization_3776/CPC_DMA_Log190819.csv'
cpc76_file2 = datapath / 'InletCharakterization_3776/CPC_DMA_Log190819_2.csv'

#cpc25_file1 = datapath / 'InletCharacterization_Inter_3025_2/CPC_DMA_Log190816.csv'
cpc25_file2 = datapath / 'InletCharacterization_Inter_3025_2/CPC_DMA_Log190819.csv'
cpc25_file3 = datapath / 'InletCharacterization_Inter_3025_2/CPC_DMA_Log190820.csv'


#Create Dataframe____________________________________________________________
#create dataframe1, load data 
cpc76data1 = pd.read_csv(cpc76_file1, infer_datetime_format=True) #CPC76 data from total inlet
#print(cpc76data1.head())
#time1=cpc76data1.TimeEnd
#conc1=cpc76data1.CPCconc

#Convert TimeEnd Strings from CPC DataFrame into datetime and put it into index column
#cpc76data1.index = pd.to_datetime(cpc76data1.TimeEnd)
#cpc76conc1=cpc76data1.CPCconc

#create dataframe, load data
cpc76data2 = pd.read_csv(cpc76_file2, infer_datetime_format=True) #total inlet CPC76
#print(cpc76data1.head())
#time2=cpc76data2.TimeEnd
#conc2=cpc76data2.CPCconc

#**********************************************************************************************
#create dataframe, load data of CPC 3025
#--------------------------------------
#cpc25data1 = pd.read_csv(cpc25_file1, infer_datetime_format=True) #CPC3025 data 1
cpc25data2 = pd.read_csv(cpc25_file2, infer_datetime_format=True) #CPC3025 data 2
cpc25data3 = pd.read_csv(cpc25_file3, infer_datetime_format=True) #CPC3025 data 3

#Convert TimeEnd Strings from CPC DataFrame into datetime and put it into index column
#cpc76data2.index = pd.to_datetime(cpc76data2.TimeEnd)
#cpc76conc2=cpc76data2.CPCconc


#Concatenate dataframes
allcpc76 = pd.concat([cpc76data1, cpc76data2])
#cpc76conc=pd.concat([cpc76conc1, cpc76conc2 ])
#cpc76=pd.concat([cpc76data1, cpc76data2 ])

#merge data
allcpc25 = pd.concat([cpc25data2, cpc25data3])
cpc25conc = allcpc25.CPCconc

#Convert TimeEnd Strings from CPC DataFrame into datetime and put it into index column
#cpc76conc = cpc76conc.sort_values(["Index"], axis=0, ascending=True) #sort all data after time


#Convert TimeEnd Strings from CPC DataFrame into datetime and put it into index column
allcpc76 = allcpc76.sort_values(["TimeEnd"], axis=0, ascending=True) #sort all data after time
allcpc76.index = pd.to_datetime(allcpc76.TimeEnd) #replace index by time

cpc76conc = allcpc76.CPCconc

#Create 10s averages out of concentration data
#---------------------------------------------
#df_T = pd.DataFrame(df.iloc[:,-2]) #separate a column from dataframe
#meanconc = cpc76conc.mean(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)
test_mean = cpc76conc.resample('10S').mean()

#Pick Date Window
#CCNC[((CCNC.index>=)&())] #use pd.to_datetime(args)
#xc = allcpc25.iloc[:,0:2] #choose column 0, 1, 2 of dataframe
#xc=pd.to_datetime(allcpc25['TimeEnd']) #convert to datetime?

#===========Calculate moving mean============================
#mylist = [1, 2, 3, 4, 5, 6, 7]
#N = 3
#cumsum, moving_aves = [0], []
#
#for i, x in enumerate(mylist, 1):
#    cumsum.append(cumsum[i-1] + x)
#    if i>=N:
#        moving_ave = (cumsum[i] - cumsum[i-N])/N
#        #can do stuff with moving_ave here
#        moving_aves.append(moving_ave)
#=============End of Example============================
# Test: df['Data'].resample('5Min', how='mean')

#__________________________________________________________________________
#Plot raw data
fig=plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(cpc76conc, '.', label = 'CPC76 Total Inlet') 

#plt.yscale('log')
#ax1.set_yscale('log')
#plt.xticks(np.arange(min(timeall), max(timeall), 10))

plt.xlabel('Time')
#plt.ylabel('Concentration')
ax1.set_ylabel('Concentration CPC76')
ax1.legend()
#ax2 = fig.add_subplot(2,1,2)
#ax2.plot(cpc76conc2)
#ax2.legend()
#ax2.set_ylabel('Concentration')



#_________________________________________________________________________

#CPC concentration measurements on each instruments position
A= np.array([['Instr', '30nm_76', 'stdev30_76', '30nm_25', 'stdev30_25' , '45nm_76', 'stdev45_76', '45nm_25', 'stdev45_25', '150nm_76', 'stdev150_76', '150nm_25', 'stdev150_25', '350nm_76', 'stdev350_76', '350nm_25', 'stdev350_25', '500nm_76', 'stdev500_76', '500nm_25', 'stdev500_25'],
            ['CCNC', 238.8, 14.8, 234.8, 9.1, 292.8, 17.4, 353.6, 12.4, 113, 10.7, 131.7, 5.6, 117, 5.1, 95.8, 5.2, 207.5, 37.6, 128.6, 24.4], 
            ['SMPS', 256.2, 16.8, 280.9, 12.5, 350.4, 18.7, 430.8, 10.1, 123.1, 10.9, 146.2, 7.1, 116.4, 4.8, 88.9, 5.4, 395.5, 27.9, 332, 25.8],
            ['AMS', 279.4, 15.5, 276.1, 9.6, 357.7, 17.8, 397.7, 9.6, 109.3, 10.3, 125.1, 5.8, 116.3, 7.1, 102.4, 7.8, 318.7, 17.7, 286.9, 18.1],
            ['Aeth', 315, 18.8, 403.3, 12.2, 337.7, 17.4, 416, 11.1, 104.6, 10, 123.3, 4.8, 113, 6, 113.7, 6.5, 426.5, 28.7, 414.7, 26.8]])



#Array with CPC concentration averages on each diameter (quotient of cpc76/cpc25), plus stdev on each average with error propagation
Avg = np.array([['q30',	'stdev30',	'q45', 'stdev45',	'q150',	'stdev150',	'q350',	'stdev350',	'q500',	'stdev500'],
[0.98, 0.07, 1.21, 0.08, 1.17, 0.12, 0.82, 0.06, 0.62, 0.16],
[1.10, 0.09, 1.23, 0.07, 1.19, 0.12, 0.76, 0.06, 0.84, 0.09],
[0.99, 0.06, 1.11, 0.06, 1.14, 0.12, 0.88, 0.09, 0.90, 0.08],
[1.28, 0.09, 1.24, 0.07, 1.18, 0.12, 1.01, 0.08, 0.97, 0.09]])

x = [30, 45, 150, 350, 500]
CCNC = Avg[1, ::2]
SMPS = Avg[2, ::2]
AMS = Avg[3, ::2]
Aeth = Avg[4, ::2]

errCCNC=Avg[1, 1::2]
errSMPS=Avg[2, 1::2]
errAMS=Avg[3, 1::2]
errAeth=Avg[4, 1::2]

fig=plt.figure()

#fig, axs = plt.subplots(4)
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)
#ax1.plot(x, CCNC, "ob", c='b', label='CCNC')

ax1.errorbar(x, np.array(CCNC, dtype=float), yerr=np.array(errCCNC, dtype = float), fmt='o', label ='CCNC')
ax2.errorbar(x, np.array(SMPS, dtype=float), yerr=np.array(errSMPS, dtype = float), fmt='o', label ='SMPS')
ax3.errorbar(x, np.array(AMS, dtype=float), yerr=np.array(errAMS, dtype = float), fmt='o', label ='AMS')
ax4.errorbar(x, np.array(Aeth, dtype=float), yerr=np.array(errAeth, dtype = float), fmt='o', label ='Aethalometer')

#
#ax2.errorbar(x, SMPS, "ob", c='g', label='SMPS')
#ax3.plot(x, AMS, "ob", c='r', label='AMS')
#ax4.plot(x, Aeth, "ob", c='c', label='CCNC')

#ax1.set_xlabel('Diameter [nm]')
ax1.set_ylabel('Concentration ratio')
ax1.set_title('Concentration ratio Instrument/Inlet ')
ax1.set_xticks([30, 45, 150, 350, 500] )
ax1.legend()


#ax2.set_xlabel('Diameter [nm]')
ax2.set_ylabel('Concentration ratio')
ax2.set_xticks([30, 45, 150, 350, 500] )
#ax2.set_title('second data set')
ax2.legend()

#ax3.set_xlabel('Diameter [nm]')
ax3.set_ylabel('Concentration ratio')
ax3.set_xticks([30, 45, 150, 350, 500] )
#ax3.set_title('second data set')
ax3.legend()

ax4.set_xlabel('Diameter [nm]')
ax4.set_ylabel('Concentration ratio')
ax4.set_xticks([30, 45, 150, 350, 500] )
#ax4.set_title('second data set')
ax4.legend()

plt.show()

#axs[0].plot(x,CCNC,"ob", c='r', label = 'CCNC')
#plt.legend()
#
#axs[1].plot(x,SMPS,"ob", c='b', label = 'SMPS')
#plt.legend()
#axs[2].plot(x,AMS,"ob", c='g', label = 'AMS')
#axs[3].plot(x,Aeth, "ob", c='c', label = 'Aethalometer')

#plt.plot(x,CCNC,"ob" )
#plt.plot(x,SMPS,"ob" )
#plt.plot(x,AMS,"ob" )
#plt.plot(x,Aeth,"ob" )

#plt.plot(x,CCNC, "ob", x, SMPS, "ob", x, AMS, "ob", x, Aeth, "ob")
#plt.scatter(x,CCNC, c='r', label = 'CCNC')
#plt.plot(x,SMPS, "ob", label = 'SMPS')
#plt.scatter(x,AMS, label = 'AMS')
#plt.scatter(x,Aeth, label = 'Aethalometer')
#plt.ylim(-2, 10)
plt.xlabel('Diameter [nm]')
#plt.ylabel('Particle Concentration [p/cc]')
#plt.title('Concentration ratio Instrument/Inlet')
plt.legend()
#plt.show()



#ToDo: Plot in different colors, let all use the same y axis, implement errorbars.


#[0.98, 1.21, 1.16, 0.81, 0.62], 
#            [1.09, 1.23, 1.19, 0.76, 0.83],
#            [0.99, 1.11, 1.15, 0.88, 0.90],
#            [1.28, 1.23, 1.18, 1, 0.97]])

##Plot with mathplotlib
#x = []
#y = []
#
#with open (cpc76_file1, 'r') as csvfile:
#    plots = csv.reader(csvfile, delimiter=',')
#    for row in plots:
#        x.append(((row[0])))
#        y.append(((row[1])))
#            
#plt.plot(x,y, label='Loaded from file!')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Interesting Graph\nCheck it out')
#plt.legend()
#plt.show()