import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from scipy import stats
import glob

def read_Eco(filename):
    Eco=pd.read_csv(filename,delimiter='\t',skiprows=8,usecols=(0,8))
    fil = open(filename, 'r')
    f=fil.read()
    d=np.int(f[:2])
    m=np.int(f[3:5])
    y=np.int(f[6:10])
    h=np.int(f[11:13])
    M=np.int(f[14:16])
    time=pd.Index(pd.datetime(y,m,d,h,M) + np.array(1000*Eco['Time'],dtype=int) * pd.offsets.Milli())
    Eco.index=time
    fil.close()
    Eco=Eco.resample('1s').mean()
    return Eco

path='K:\\Data\\Pre-Campaign-Tests\\O3calibration\\'
file1='20180425_Ozon_Arctic.txt'
import glob

#%% Loading files
ozone2B=pd.read_csv(path+file1,header=None,delimiter=',',index_col=0,parse_dates=True,infer_datetime_format=True)

O3Calib=np.array([251,151,100,30,15,50])#-29.9
#O3Calib=np.array([228,,
        
Start=np.array(['13:24:00','13:37:00','13:47:00','14:09:00','14:22:00','14:50:00'])
End=np.array(['13:31:00','13:42:00','14:01:00','14:17:00','14:45:00','15:13:00'])
date='2018-04-25 '
O32Bavg=np.zeros(len(Start))
for j in range (len(Start)):
    O32Bavg[j]=ozone2B[1][(ozone2B.index>date+Start[j])&(ozone2B.index<date+End[j])].mean()
    
#reading Ecotech
FilesEco=glob.glob(path+'*.dat')
ozoneEco=pd.DataFrame()
for j in range (len(FilesEco)-1):
    ECOpar=read_Eco(FilesEco[j])
    ozoneEco=ozoneEco.append(ECOpar)

Start2=np.array(['15:25','15:35','16:05','16:39','17:09','18:01'])
End2=np.array(['15:27','15:50','16:28','17:04','17:55','18:55'])

O3Ecoavg=np.zeros(len(Start2))
for j in range (len(Start2)):
    O3Ecoavg[j]=ozoneEco['O3'][(ozoneEco.index>date+Start2[j])&(ozoneEco.index<date+End2[j])].mean()
#%% add zero
O3Calib=np.append(O3Calib,0)
O32Bavg=np.append(O32Bavg,0.7)
O3Ecoavg=np.append(O3Ecoavg,0)

#%%fit
slope2B,intercept2B,rvalue,pvalue,stderr=stats.linregress(O3Calib,O32Bavg)
slope2B_2,intercept2B_2,rvalue,pvalue,stderr=stats.linregress(O3Ecoavg,O32Bavg)

#%% Correct gas calibrator
Setpoint=np.array([0.05,0.25,0.5])*1000
Realvalue=np.array([0.03,0.228,0.521])*1000
slopeEco,interceptEco,rvalue,pvalue,stderr=stats.linregress(Realvalue,Setpoint)

plt.figure()
plt.plot(Realvalue,Setpoint,'o',label='Calibrator')
plt.plot((0,500),(interceptEco,interceptEco+slopeEco*500),label='slope=')
plt.xlabel('Real value [ppb]')
plt.ylabel('Set point [ppb]')
#%% PLOT
plt.close('all')
plt.figure()

plt.plot((0,250),(0,250),label='1:1 line')
plt.plot((0,250),(intercept2B,intercept2B+slope2B*250),label='fit: y=0.86x+8.1')
plt.plot((0,250),(intercept2B_2,intercept2B_2+slope2B_2*250),label='fit: y=0.9x-4.2')

plt.plot(O3Ecoavg,O32Bavg,'gs',label='2B vs Ecotech',markersize=8)
plt.plot(O3Calib,O32Bavg,'o',label='2B vs calibrator',markersize=7)


plt.xlabel('O3 concentration [ppb], calibrator')
plt.ylabel('Measured O3 concentration [ppb]')
plt.legend()
plt.title('O3 monitor calibration, 25/04/2018')
plt.show()

plt.figure()
plt.plot(O3Ecoavg,O32Bavg,'o',label='2B monitor')
plt.plot((0,250),(0,250),label='1:1 line')