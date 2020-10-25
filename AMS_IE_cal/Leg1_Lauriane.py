# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:32:34 2019

@author: beck_i
"""

#import csv
import matplotlib.pyplot as plt
from pathlib import Path
#import pandas as pd
import numpy as np
#import time
#from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit


#*********************************************
#DATA From 20190927
#create data list from Excel file:
AMS_NO3_1 = [9.3, 3.62, 5.34, 21.14, 11.46] 
AMS_NH4_1 = [1.98, 0.78, 1.15, 4.4, 2.41]

CPC_NO3_1 = [12.35, 5.58, 8.45, 31.9, 16.34]
CPC_NH4_1 = [5.38, 1.62, 2.45, 9.26, 4.74]

IE20190920 = 2.55E-8 #Ionization Efficency from AMS data

AB20190927 = 8.62E+04

#*******************************************
#DATA From 20191002
#create data list from Excel file:
AMS_NO3_2 = [2.02, 4.83, 2.94, 7.9, 13.78] 
AMS_NH4_2 = [0.45, 1.09, 0.66, 1.69, 2.94]

CPC_NO3_2 = [3.18, 7.56, 4.79, 11.8, 18.35]
CPC_NH4_2 = [0.92, 2.2, 1.39, 3.42, 5.33]

AB20191002 = 8.66E4 #Airbeam from AMS data (Igor)


#*******************************************
#DATA From 20191010
#create data list from Excel file:
AMS_NO3_3 = [3.58, 1.39, 8.8, 12.25, 6.49] 
AMS_NH4_3 = [0.76, 0.29, 1.87, 2.57, 1.35]

CPC_NO3_3 = [3.97, 1.58, 9.95, 13.54, 7.18]
CPC_NH4_3 = [1.15, 0.46, 2.89, 3.93, 2.08]

AB20191010 = 9.1E4 #Airbeam from AMS data (Igor)


#*******************************************
#DATA From 20191016
#create data list from Excel file:
AMS_NO3_4 = [0.64, 2.1, 5.8, 11.17] 
AMS_NH4_4 = [0.17, 0.43, 1.22, 2.4]

CPC_NO3_4 = [0.96, 3.18, 7.49, 12.27]
CPC_NH4_4 = [0.28, 0.92, 2.17, 3.56]

AB20191016 = 8.75E4  #Airbeam from AMS data (Igor)



##*******************************************
#DATA From 20191022
#create data list from Excel file:
AMS_NO3_5 = [4.42, 1.9, 13.48, 10.19, 27.76] 
AMS_NH4_5 = [.96, 0.41, 2.88, 2.13, 5.65]

CPC_NO3_5 = [5.98, 2.8, 16.7, 12.34, 33.8]
CPC_NH4_5 = [1.73, 0.8, 4.8, 3.58, 9.8]

AB20191022 = 9.14E4



#*******************************************
#DATA From 20191029
#create data list from Excel file:
AMS_NO3_6 = [1.66, 6.65, 8.66, 17.64, 1.91] 
AMS_NH4_6 = [0.36, 1.38, 1.79, 3.67, 0.42]

CPC_NO3_6 = [2.39, 8.37, 10.76, 21.91, 2.46]
CPC_NH4_6 = [0.69, 2.43, 3.12, 6.36, 0.71]

AB20191029 = 9.34E4

#*******************************************************
#Plot graph
#Plot 20190927
fig=plt.figure(1)

p = np.polyfit(CPC_NO3_1, AMS_NO3_1, 1) # Last argument is degree of polynomial. OUtput are the coefficients, highest firs (1D: ax +b, first a, then b)
f = np.poly1d(p) # So we can call f(x)
plt.plot(CPC_NO3_1, AMS_NO3_1,'x')
plt.plot(CPC_NO3_1,f(CPC_NO3_1), 'b-',label="Polyfit")
plt.xlabel('CPC NO3 mass (ug/m3)')
plt.ylabel('AMS NO3 mass (ug/m3)')
plt.title('IEcal from 2019/09/27')
slope20190927 = p[0] #Slope of the fit 
plt.legend(['data','fit slope: %.2f' %slope20190927])

plt.show()

#**********************************************************
#Plot 20191002
fig=plt.figure(2)

p = np.polyfit(CPC_NO3_2, AMS_NO3_2, 1) # Last argument is degree of polynomial. OUtput are the coefficients, highest firs (1D: ax +b, first a, then b)
f = np.poly1d(p) # So we can call f(x)
plt.plot(CPC_NO3_2, AMS_NO3_2,'x')
plt.plot(CPC_NO3_2,f(CPC_NO3_2), 'b-',label="Polyfit")
plt.xlabel('CPC NO3 mass (ug/m3)')
plt.ylabel('AMS NO3 mass (ug/m3)')
plt.title('IEcal from 2019/10/02')
slope20191002 = p[0] #Slope of the fit 
plt.legend(['data','fit slope: %.2f' %slope20191002])

plt.show()


#**********************************************************
#Plot 20191010
fig=plt.figure(3)

p = np.polyfit(CPC_NO3_3, AMS_NO3_3, 1) # Last argument is degree of polynomial. OUtput are the coefficients, highest firs (1D: ax +b, first a, then b)
f = np.poly1d(p) # So we can call f(x)
plt.plot(CPC_NO3_3, AMS_NO3_3,'x')
plt.plot(CPC_NO3_3,f(CPC_NO3_3), 'b-',label="Polyfit")
plt.xlabel('CPC NO3 mass (ug/m3)')
plt.ylabel('AMS NO3 mass (ug/m3)')
plt.title('IEcal from 2019/10/10')
slope20191010 = p[0] #Slope of the fit 
plt.legend(['data','fit slope: %.2f' %slope20191010])

plt.show()

#**********************************************************
#Plot 20191016
fig=plt.figure(4)

p = np.polyfit(CPC_NO3_4, AMS_NO3_4, 1) # Last argument is degree of polynomial. OUtput are the coefficients, highest firs (1D: ax +b, first a, then b)
f = np.poly1d(p) # So we can call f(x)
plt.plot(CPC_NO3_4, AMS_NO3_4,'x')
plt.plot(CPC_NO3_4,f(CPC_NO3_4), 'b-',label="Polyfit")
plt.xlabel('CPC NO3 mass (ug/m3)')
plt.ylabel('AMS NO3 mass (ug/m3)')
plt.title('IEcal from 2019/10/16')
slope20191016 = p[0] #Slope of the fit 
plt.legend(['data','fit slope: %.2f' %slope20191016])

plt.show()

#**********************************************************
#Plot 20191022
fig=plt.figure(5)

p = np.polyfit(CPC_NO3_5, AMS_NO3_5, 1) # Last argument is degree of polynomial. OUtput are the coefficients, highest firs (1D: ax +b, first a, then b)
f = np.poly1d(p) # So we can call f(x)
plt.plot(CPC_NO3_5, AMS_NO3_5,'x')
plt.plot(CPC_NO3_5,f(CPC_NO3_5), 'b-',label="Polyfit")
plt.xlabel('CPC NO3 mass (ug/m3)')
plt.ylabel('AMS NO3 mass (ug/m3)')
plt.title('IEcal from 2019/10/22')
slope20191022 = p[0] #Slope of the fit 
plt.legend(['data','fit slope: %.2f' %slope20191022])

plt.show()

#**********************************************************
#Plot 20191029
fig=plt.figure(6)

p = np.polyfit(CPC_NO3_6, AMS_NO3_6, 1) # Last argument is degree of polynomial. OUtput are the coefficients, highest firs (1D: ax +b, first a, then b)
f = np.poly1d(p) # So we can call f(x)
plt.plot(CPC_NO3_6, AMS_NO3_6,'x')
plt.plot(CPC_NO3_6,f(CPC_NO3_6), 'b-',label="Polyfit")
plt.xlabel('CPC NO3 mass (ug/m3)')
plt.ylabel('AMS NO3 mass (ug/m3)')
plt.title('IEcal from 2019/10/29')
slope20191029 = p[0] #Slope of the fit 
plt.legend(['data','fit slope: %.2f' %slope20191029])

plt.show()

#===============================================================
#Plot RIE of NH4 (AMS NO3/NH4 plot and slope)

fig=plt.figure(10)

p = np.polyfit(AMS_NH4_1, AMS_NO3_1, 1) # x, y, Last argument is degree of polynomial. OUtput are the coefficients, highest firs (1D: ax +b, first a, then b)
f = np.poly1d(p) # So we can call f(x)
plt.plot(AMS_NH4_1, AMS_NO3_1,'x')
plt.plot(AMS_NH4_1,f(AMS_NH4_1), 'b-',label="Polyfit")
plt.xlabel('AMS NH4 mass (ug/m3)')
plt.ylabel('AMS NO3 mass (ug/m3)')
plt.title('RIEcal from 2019/09/27')
RIEslope20190927 = p[0] #Slope of the fit 
plt.legend(['data','fit RIE slope: %.2f' %RIEslope20190927])

plt.show()

#--------------------------------------------------------------
fig=plt.figure(11)

p = np.polyfit(AMS_NH4_2, AMS_NO3_2, 1) # x, y, Last argument is degree of polynomial. OUtput are the coefficients, highest firs (1D: ax +b, first a, then b)
f = np.poly1d(p) # So we can call f(x)
plt.plot(AMS_NH4_2, AMS_NO3_2,'x')
plt.plot(AMS_NH4_2,f(AMS_NH4_2), 'b-',label="Polyfit")
plt.xlabel('AMS NH4 mass (ug/m3)')
plt.ylabel('AMS NO3 mass (ug/m3)')
plt.title('RIEcal from 2019/10/02')
RIEslope20191002 = p[0] #Slope of the fit 
plt.legend(['data','fit RIE slope: %.2f' %RIEslope20191002])

plt.show()

#---------------------------------------------------------------
fig=plt.figure(12)

p = np.polyfit(AMS_NH4_3, AMS_NO3_3, 1) # x, y, Last argument is degree of polynomial. OUtput are the coefficients, highest firs (1D: ax +b, first a, then b)
f = np.poly1d(p) # So we can call f(x)
plt.plot(AMS_NH4_3, AMS_NO3_3,'x')
plt.plot(AMS_NH4_3,f(AMS_NH4_3), 'b-',label="Polyfit")
plt.xlabel('AMS NH4 mass (ug/m3)')
plt.ylabel('AMS NO3 mass (ug/m3)')
plt.title('RIEcal from 2019/10/10')
RIEslope20191010 = p[0] #Slope of the fit 
plt.legend(['data','fit RIE slope: %.2f' %RIEslope20191010])

plt.show()


#---------------------------------------------------------------
fig=plt.figure(13)

p = np.polyfit(AMS_NH4_4, AMS_NO3_4, 1) # x, y, Last argument is degree of polynomial. OUtput are the coefficients, highest firs (1D: ax +b, first a, then b)
f = np.poly1d(p) # So we can call f(x)
plt.plot(AMS_NH4_4, AMS_NO3_4,'x')
plt.plot(AMS_NH4_4,f(AMS_NH4_4), 'b-',label="Polyfit")
plt.xlabel('AMS NH4 mass (ug/m3)')
plt.ylabel('AMS NO3 mass (ug/m3)')
plt.title('RIEcal from 2019/10/16')
RIEslope20191016 = p[0] #Slope of the fit 
plt.legend(['data','fit RIE slope: %.2f' %RIEslope20191016])

plt.show()


#---------------------------------------------------------------
fig=plt.figure(14)

p = np.polyfit(AMS_NH4_5, AMS_NO3_5, 1) # x, y, Last argument is degree of polynomial. OUtput are the coefficients, highest firs (1D: ax +b, first a, then b)
f = np.poly1d(p) # So we can call f(x)
plt.plot(AMS_NH4_5, AMS_NO3_5,'x')
plt.plot(AMS_NH4_5,f(AMS_NH4_5), 'b-',label="Polyfit")
plt.xlabel('AMS NH4 mass (ug/m3)')
plt.ylabel('AMS NO3 mass (ug/m3)')
plt.title('RIEcal from 2019/10/22')
RIEslope20191022 = p[0] #Slope of the fit 
plt.legend(['data','fit RIE slope: %.2f' %RIEslope20191022])

plt.show()


#---------------------------------------------------------------
fig=plt.figure(15)

p = np.polyfit(AMS_NH4_6, AMS_NO3_6, 1) # x, y, Last argument is degree of polynomial. OUtput are the coefficients, highest firs (1D: ax +b, first a, then b)
f = np.poly1d(p) # So we can call f(x)
plt.plot(AMS_NH4_6, AMS_NO3_6,'x')
plt.plot(AMS_NH4_6,f(AMS_NH4_6), 'b-',label="Polyfit")
plt.xlabel('AMS NH4 mass (ug/m3)')
plt.ylabel('AMS NO3 mass (ug/m3)')
plt.title('RIEcal from 2019/10/29')
RIEslope20191029 = p[0] #Slope of the fit 
plt.legend(['data','fit RIE slope: %.2f' %RIEslope20191029])

plt.show()
#=============================================================

#IE and RIE Calculations
#---------------

IE_default = 2E-7


#IE 20190927
IE20190927 = slope20190927*IE_default
#RIE 20190927
RIE20190927 = 4/(0.29*RIEslope20190927)

#IE 20191002
#IE20191002 = slope20191002*IE20190927
IE20191002 = slope20191002*IE_default
#RIE 20191002
RIE20191002 = 4/(0.29*RIEslope20191002)

#IE 20191010
#IE20191010 = slope20191010*IE20191002
IE20191010 = slope20191010*IE_default
#RIE 20191010
RIE20191010 = 4/(0.29*RIEslope20191010)

#IE 20191016
#IE20191016 = slope20191016*IE20191010
IE20191016 = slope20191016*IE_default
#RIE 20191016
RIE20191016 = 4/(0.29*RIEslope20191016)

#IE 20191022
#IE20191022 = slope20191022*IE20191016
IE20191022 = slope20191022*IE_default
#RIE 20191022
RIE20191022 = 4/(0.29*RIEslope20191022)

#IE 20191029
#IE20191029 = slope20191029*IE20191022
IE20191029 = slope20191029*IE_default
#RIE 20191029
RIE20191029 = 4/(0.29*RIEslope20191029)

#Plot IE Changes over time




## create a linear regression model
#model = LinearRegression()
#model.fit(x, y)
#
## predict y from the data
#x_new = np.linspace(0, 30, 100)
#y_new = model.predict(x_new[:, np.newaxis])
#
## plot the results
#plt.figure(figsize=(4, 3))
#ax = plt.axes()
#ax.scatter(x, y)
#ax.plot(x_new, y_new)
#
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#
#ax.axis('tight')
#
#