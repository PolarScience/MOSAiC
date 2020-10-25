#############################################################
#############################################################
## Python script used to analyze the ozone measurement for ##
## calibration of the O2B monitor in December18 and June19 ##
#############################################################
#############################################################
#Author: A. Baccarini
#Date: 27 June 2019


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib

#%%matplotlib parameters
matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams['font.size'] = 18

#%%
######################
#Calibration Dec 2018#
######################
datadir='C:\\Users\\baccarini_a\\cernbox\\MOCCHA\\Ozone calibration\\2018-12-Calib-O3\\'

fnam='20181212_Ozon_Arctic.txt'

ozone=pd.read_csv(datadir+fnam,header=None,delimiter=',',index_col=0,parse_dates=True,infer_datetime_format=True)

ozone.index=pd.to_datetime(ozone.index,errors='coerce')
ozoneval=pd.to_numeric(ozone[1],errors='coerce')


O3_real=np.array([100, 50, 75, 150, 30, 0])
start_time=pd.DatetimeIndex(['2018-12-12 10:18', '2018-12-12 10:51', '2018-12-12 11:30', '2018-12-12 13:30', '2018-12-12 14:30', '2018-12-12 15:38'])
end_time=pd.DatetimeIndex(['2018-12-12 10:33', '2018-12-12 11:11', '2018-12-12 13:00', '2018-12-12 14:05', '2018-12-12 15:30', '2018-12-12 15:50'])

O3_measured=np.zeros(len(O3_real))
O3_std=np.zeros(len(O3_real))
for j in range (len(O3_real)):
    O3_measured[j]=ozoneval[(ozoneval.index>start_time[j])&(ozoneval.index<end_time[j])].mean()  
    O3_std[j]=ozoneval[(ozoneval.index>start_time[j])&(ozoneval.index<end_time[j])].std()/np.sqrt(len(ozoneval[(ozoneval.index>start_time[j])&(ozoneval.index<end_time[j])]))

slope, intercept, r_value, p_value, std_err = stats.linregress(O3_real[:-1],O3_measured[:-1])

#plt.close('all')
plt.figure()
ax=plt.subplot(111)
ax.errorbar(O3_real,O3_measured, yerr=O3_std, fmt='o',markersize=8)
ax.plot((0,150),(0+intercept,150*slope+intercept),label='fit: y= %.3f x  %.3f' %(slope, intercept))
ax.plot((0,150),(0,150),label='1:1 line')

ax.set_xlabel('O3 calibrator concentration [ppb]')
ax.set_ylabel('O3 measured concentration [ppb]')
ax.legend()

#%%
######################
#Calibration Jun 2019#
######################

datadir='C:\\Users\\baccarini_a\\cernbox\\MOCCHA\\Ozone calibration\\O3Calib_Jun19\\'

#2B monitor
fnam='20190626_Ozon_Arctic.txt'

ozone2B=pd.read_csv(datadir+fnam,header=None,delimiter=',',index_col=0,parse_dates=True,infer_datetime_format=True)

ozone2B.index=pd.to_datetime(ozone2B.index,errors='coerce')
ozoneval2B=pd.to_numeric(ozone2B[1],errors='coerce')
ozoneval2B=ozoneval2B['2019-06-26']
#Serinus
fnam='20190625HONOgen.txt'

ozoneSer=pd.read_csv(datadir+fnam,delimiter='\t',index_col=0,parse_dates=True,infer_datetime_format=True)
ozonesigSer=ozoneSer['2019-06-26 11:00':]['NO ']

#Voltage conversion
#The serinus signal is in Volts I need to convert into a concentration
ozonevalSer=(ozonesigSer-0.25)/0.01


start_time=pd.DatetimeIndex(['2019-06-26 11:42', '2019-06-26 12:04','2019-06-26  12:20','2019-06-26 12:39', '2019-06-26 13:02','2019-06-26 13:20','2019-06-26 13:40','2019-06-26 13:53'])
end_time=pd.DatetimeIndex(['2019-06-26 11:58', '2019-06-26 12:15','2019-06-26 12:34', '2019-06-26 12:56','2019-06-26 13:14','2019-06-26 13:31','2019-06-26 13:48','2019-06-26 14:30'])

O3_2Bavg=np.zeros(len(start_time))
O3_Seravg=np.zeros(len(start_time))

O3_2Bstd=np.zeros(len(start_time))
O3_Serstd=np.zeros(len(start_time))

for j in range (len(start_time)):
    O3_2Bavg[j]=ozoneval2B[(ozoneval2B.index>start_time[j])&(ozoneval2B.index<end_time[j])].mean()  
    O3_2Bstd[j]=ozoneval2B[(ozoneval2B.index>start_time[j])&(ozoneval2B.index<end_time[j])].std()#/np.sqrt(len(ozoneval2B[(ozoneval2B.index>start_time[j])&(ozoneval2B.index<end_time[j])]))
    
    O3_Seravg[j]=ozonevalSer[(ozonevalSer.index>start_time[j])&(ozonevalSer.index<end_time[j])].mean()  
    O3_Serstd[j]=ozonevalSer[(ozonevalSer.index>start_time[j])&(ozonevalSer.index<end_time[j])].std()#/np.sqrt(len(ozonevalSer[(ozonevalSer.index>start_time[j])&(ozonevalSer.index<end_time[j])]))

O3_Seravg[-1]=0 #I artificially force the last point to be 0 as we know it was O3 free
slope, intercept, r_value, p_value, std_err = stats.linregress(O3_Seravg,O3_2Bavg)
slope2, intercept2, r_value, p_value, std_err = stats.linregress(O3_Seravg[-4:],O3_2Bavg[-4:])

#plt.close('all')
plt.figure(figsize=(15,13))
ax=plt.subplot(111)
ax.errorbar(O3_Seravg,O3_2Bavg, yerr=O3_2Bstd,xerr=O3_Serstd, fmt='o',markersize=8)
ax.plot((0,300),(0+intercept,300*slope+intercept),label='fit: y= %.3f x  %.3f' %(slope, intercept))
ax.plot((0,60),(0+intercept2,60*slope2+intercept),label='fit (limit to 60 ppb): y= %.3f x  %.3f' %(slope2, intercept2))
ax.plot((0,300),(0,300),label='1:1 line')

ax.set_xlabel('O3 reference concentration [ppb]')
ax.set_ylabel('O3 2B measured concentration [ppb]')
ax.legend()
plt.tight_layout()
plt.savefig('Calib_line.pdf')

plt.figure(figsize=(15,13))
plt.plot(ozoneval2B['2019-06-26 11:41':],'r', label='2B monitor')
plt.plot(ozonevalSer['2019-06-26 11:41':],'b',label='Serinus monitor')
plt.legend()
plt.tight_layout()
plt.savefig('Calib_timeseries.pdf')

plt.show()

####################################
# check Serinus voltage conversion #
####################################

#voltage calibration
SerVOutp=np.array([ozonesigSer['2019-06-26 11:39:32':'2019-06-26 11:39:57'].mean(),ozonesigSer['2019-06-26 11:40:05':'2019-06-26 11:40:50'].mean()])
SerVth=np.array([5,0.5])

RealConc=np.array([270, 187.1, 130.1, 85.4, 84.6, 30, 29.7, 12, 53.7, 54.7, 0.5])
RealConc2=np.array([270, 187.1, 130.1, 84.6, 29.7, 12, 54.7, 0.5])
Vreading=np.array([2.918, 2.102, 1.562, 1.113, 1.108, 0.547, 0.545, 0.371, 0.791, 0.801, 0.254]) #this is what I wrote in the logbook
Vmeasure=np.array([2.919,2.104,1.561,1.114,1.108,0.548,0.544,0.372, 0.794, 0.8,0.255]) #this is the average of the real signal over the minute I note down on the logbook

slope, intercept, r_value, p_value, std_err = stats.linregress(RealConc,Vreading)

plt.figure(figsize=(18,14))
ax=plt.subplot(211)
ax.plot(RealConc,((Vmeasure-0.25)*100)/RealConc,'o')
ax.set_ylabel('Measured/Displayed  concentration [ppb]')
ax.set_xlabel('Displayed concentration [ppb]')

ax2=plt.subplot(212)
ax2.plot(RealConc,(Vmeasure-0.25)*100-RealConc,'o')
ax2.set_ylabel('Measured - Displayed \n concentration [ppb]')
ax2.set_xlabel('Displayed concentration [ppb]')
#plt.plot((0,3),(intercept,3*slope+intercept))

plt.figure()

#%% Baseline drift
time=pd.to_datetime(['2017-07-01','2018-05-01','2018-12-01','2019-07-01'])
offset=np.array([2,0.7,-1.7,-4])

Timeofoperation=np.array([0,30,100,160])

plt.figure()
plt.plot(time,offset,'o')
plt.ylabel('2B offset [ppb]')

plt.figure()
plt.plot(Timeofoperation,offset,'o',markersize=8)
plt.ylabel('2B offset [ppb]')
plt.xlabel('Estimated nr of days of operations')
plt.title('2B monitor drift')