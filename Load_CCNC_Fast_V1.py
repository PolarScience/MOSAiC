# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:38:00 2020

@author: beck_i

Purpose: 
Read all CCNC files in a folder
Get date of each file from its filename
Separate Data by SuperSaturation
Delete the first x seconds of each supersaturation value
Output of SS data and CCN Number concentration as csv and hdf file:
    The file contains: Datetime, Single column for each SS, Column with CCNC Number Concentrations

Changelog: 
20200624: Finished script, its running nicely.
20200821: Tried it with new laptop, new python etc, and it does not work anymore. 
--> Solved issue: I had a pd.datetime and a python date mixed together. changed now both to pd.datetimes. 

On the old one it still works, loading 1month takes approx. 400s
Corrupt File: CCN 100 data 200521040000
Loading 1 months takes new python ca. 130-200s

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
import os #os.path: This module implements some useful functions on pathnames.
import re
from datetime import datetime
import time

#%%START=====================================================================
#Close all existing figures:
plt.close("all") #this is the same as: .atplotlib.pyplot.close("all")
#============================================================================

#%%
start = time.time()

#%%
def load_CCNC(file):
    df=pd.read_csv(file,sep=',', skiprows = [0,1,2,3,5]) #index_col=0, parse_dates = [[0,1]],names = header_list, infer_datetime_format = True
    # create datetime column
#    df['Time']= date + pd.to_timedelta(df.Time) 
#    df.sort_index(inplace=True) #sort index by datetime 
#    df.set_index('Time',inplace=True)
#    cut_frame=df[['Current SS', 'CCN Number Conc']] #

    return(df)

def get_date(file):
    """
    get date of file form its filename
    """
    #Split filepath into path and file
    head_tail = os.path.split(file)
    # search for certain number of characters in a string, returns a string
    # ie d{6} means search for 6 digits
    datestring = re.search(r'\d{6}', head_tail[1]).group()
    # startdate = datetime.strptime(datestring, '%y%m%d').date()
    startdate= pd.to_datetime(datestring, format='%y%m%d')
    return(startdate)

    
#%% Inputs
path = r'D:\All_Data\CCNC\CCNC_201909-10/'
files = sorted(glob.glob(path + '*.csv')) #Open pointer to files

outpath = Path(r'D:\All_Data\CCNC')
outfile = 'MOSAiC_CCNC_output_2.csv'
outfile_hdf = 'MOSAiC_CCNC_all_new_test.hdf'

ccndata = pd.DataFrame()

for j in files:
    # get the date from the actual file:
    filedate = get_date(j) 
    # read the file and 
    rawfile = load_CCNC(j)
    # remove whitespaces from columns
    rawfile.columns=pd.Series(rawfile.columns).str.strip()
    # set time to date + time and convert to datetime format
    # datestring = filedate + " " + rawfile.Time
    # rawfile.Time = datetime.strptime(datestring, "%y%m%d %H:%M:%S")
    rawfile.Time = filedate + pd.to_timedelta(rawfile.Time) 
    # write into a dataframe
    ccndata=ccndata.append(rawfile)  
    
#%% 
# dropping (all but first) duplicte values 
ccndata.drop_duplicates(subset ="Time", keep ='first', inplace = True) 
#ccndata.set_index('Time', inplace = True)

# Define Timeperiod to drop data and convert it to datetime format
#skipsec = 180  
#threshold_t = pd.Timedelta(skipsec, 's')
# Choose the relevant columns

used_data=ccndata[['Time','Current SS', 'CCN Number Conc']]
used_data.set_index('Time', inplace = True)
sep_data = used_data[['Current SS']].copy()
#sep_data.set_index('Time', inplace = True)
# separate all SS values in single columns, set rest of values in each column to nan
sep_data['SS0.15']=sep_data['CCN Number Conc'][sep_data['Current SS']==0.15] #geht auch
sep_data['SS0.2']=sep_data['CCN Number Conc'][sep_data['Current SS']==0.2]
sep_data['SS0.3']=sep_data['CCN Number Conc'][sep_data['Current SS']==0.3]
sep_data['SS0.5']=sep_data['CCN Number Conc'][sep_data['Current SS']==0.5] #old: sep_data['SS0.5']=sep_data['Current SS'][sep_data['Current SS']==0.5]

sep_data['SS1']=sep_data['CCN Number Conc'][sep_data['Current SS']==1]
# alternative way to do it
#sep_data['SS0.15']=sep_data.loc[sep_data['Current SS'] == 0.15]['Current SS']
#sep_data['SS0.2']=sep_data.loc[sep_data['Current SS'] == 0.2]['Current SS']
#sep_data['SS0.3']=sep_data.loc[sep_data['Current SS'] == 0.3]['Current SS']
#sep_data['SS0.5']=sep_data.loc[sep_data['Current SS'] == 0.5]['Current SS']
#sep_data['SS1']=sep_data.loc[sep_data['Current SS'] == 1]['Current SS']

numberofsamples = 180
for ss in ['SS0.15', 'SS0.2', 'SS0.3', 'SS0.5', 'SS1']: #
    #set mask: 1 if nan, nan if data is there
    mask = sep_data[ss].copy() #Series that consists nan and SS values
    mask[np.isnan(sep_data[ss])] = 1 #set all nan to 1
    mask[~np.isnan(sep_data[ss])] = np.nan #set al data values to nan
    mask= mask.interpolate(method='linear', limit=numberofsamples, limit_direction='forward') #interpolate over the nan values for a length of numberofsamples, means continue with 1 
    #set data nan if mask ==1 : dfz['C'] = np.where(dfz['E'].isnull(), dfz['E'], dfz['C'])
    sep_data[ss][~np.isnan(mask)]=np.nan #set all data values to nan where mask is not nan

# Copy data column to SS dataframe
sep_data['CCN Number Conc'] = used_data['CCN Number Conc']

"""
Gedanken: 
    [~np.isnan(mask)]=np.nan #überall wo diese Liste true ist, will ich ein nan in die sep_data[ss] schreiben

"""
#Find all non-NAN in mask: a) mask ==1 b) ~np.isnan(mask)
# Find changes in SS 
#starttime=used_data['Time'].loc[used_data['Current SS'].diff() != 0]
#used_data.set_index('Time', inplace = True)

#%%
# Tipps: 
# 
# df.add
# Maske, die nan ist wenn daten SS da sind, 1 wenn keine da sind
# interp auf diese maske anwenden, sagen wie weit ich interpolieren darf
# nur nach rechts für 3 min interpolieren, dort wo 1 ist wird es länger (die ersten 3min), 
# danach alles was 1 ist zu nan setzen. 
#loop über 3 SS strings, df.[SS] maske = 



# List of our starting points: 
#used_data.loc[starttime]
#used2=starttime.reset_index(drop=True)  #Time series of start time of each SS
#
#%%
#dropdata = pd.DataFrame()
#for x in starttime:
#    starting_time = x
#    ending_time=x+threshold_t
#    dropframe = used_data.loc[starting_time:ending_time]
##    dropframe = used_data.loc[starting_time:ending_time].index
#    dropdata.append(dropframe)
#    
#used_data=used_data.drop(zz)
    
#starter_time=starttime.iloc[4] #start time for dropping
#ending_time=starter_time+threshold_t #End time for dropping
#dropframe = used_data.loc[x:y]
#zzz=used_data.drop(used_data.loc[x:y].index)
#
#used_data.loc[starttime]:used_data.loc[starttime.threshold_t]

#%%

#used_dropped=used2.drop(k)
#to_drop=used2.loc[k:z]

#for  x in used_ata.loc[used_data['Current SS'].diff() !=0]:
    
#used_data2=used_data.drop(used_data.index[0:10])
        
#    CCNCdat
#    CCNCdat['C']=CCNCdat['Current SS'].diff()
#    CCNC_filtered = CCNCdat.[CCNCdat['C'] != 0]
 

#startdate=cut_data.Time.iloc[0]
#enddate= cut_data.Time.iloc[-1]   
#outfile = '{0}_{1}_CCN.csv'.format(startdate.date(), enddate.date())

# save data to file:These both work
#sep_data.to_csv(os.path.join(outpath, outfile), mode = 'a') 
sep_data.to_hdf(os.path.join(outpath, outfile_hdf), key='ccnc',format = 'table', append = True)
end = time.time()
print(end - start)



