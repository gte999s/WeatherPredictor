
import pandas as pd
import TempPrediction as tp
import datetime
import numpy as np

def getStationDataFrame(station):

    where = 'WBAN==' + str(station)
    df = pd.read_hdf('hourlyData.hdf',where=where)
 
    return df


def switchIndex(df):
    wban =  df.index.get_level_values(0)
    dates = df.index.get_level_values(1)
    times = df.index.get_level_values(2)

    index=0

    times=times.astype(np.int32)
    dateIndex = []
    for date in dates:
        time = times[index].item(0)
        dt = datetime.timedelta(hours = time//100, minutes = time%100)
        dateIndex.append(date+dt)
        index+=1

    df.index=dateIndex
    df['WBAN']=wban

    return df[~df.index.duplicated(keep='first')]

"""
Main File
"""

stationFile = open('wbanStation.txt','r')
data = stationFile.read()
stationFile.close()
stations = data.split('\n')

startTime = None
stopTime = None

dfs = []
#stations = stations[:3]
for index, station in enumerate(stations):
    print('reindexing station',index, 'of', len(stations))

    df = getStationDataFrame(station)
    
    if len(df) > 0:

        df = switchIndex(df)

        if df.index[0] < datetime.datetime(2009,1,1) and df.index[-1] > datetime.datetime(2017,1,1):

            dfs.append(df)

            if startTime is None or startTime < df.index[0]:
                startTime = df.index[0]
            if stopTime is None or stopTime < df.index[-1]:
                stopTime = df.index[-1]

startTime = startTime + datetime.timedelta(hours=1)
startTime=startTime.replace(minute=0)
stopTime = stopTime - datetime.timedelta(hours=1)
stopTime.replace(minute=0)

indexRange = pd.date_range(startTime,stopTime,freq='H')

for index, df in enumerate(dfs):
    print('Reindexing', index, 'of', len(dfs))
   
    dfs[index] = df.reindex(indexRange,method='nearest')

print('comcating all station info')
dfAll = pd.concat(dfs,axis=1)

dfAll.to_hdf('trainingData.hdf', 'trainingData', format='t', mode='w', complevel=1, complib='zlib')





