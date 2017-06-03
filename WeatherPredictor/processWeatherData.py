"""
This function processes downloaded NOAA zip files into a single archive of hourly data
"""

import os
import zipfile
import pandas as pd


weatherDir = "weatherData"

print(os.getcwd())
# Extract any data not already extracted
for file in os.listdir(weatherDir):
    if file.endswith('.zip'):
        curFile = os.path.join(weatherDir, file)
        with zipfile.ZipFile(curFile, 'r') as cZip:
            for member in cZip.namelist():
                destination = os.path.join(weatherDir, member)
                if not os.path.exists(destination):
                    print('Unzipping File: ', member)
                    cZip.extract(member, weatherDir)

# Collate all hourly data
hourlyDataFileName = 'hourlyData.hdf'
if os.path.isfile(hourlyDataFileName):
    os.remove(hourlyDataFileName)

     
for file in os.listdir(weatherDir):
    if file.endswith("hourly.txt"):
        print('Processing file: ', file)
        df = pd.read_csv(os.path.join(weatherDir,file),
                        index_col=['WBAN', 'Date', 'Time'],
                         na_values=('M', '  ', ' ', 'VR', 'VR ',  "  T", "   ", 'null'),
                         usecols=['WBAN', 'Date', 'Time', 'DryBulbFarenheit', 'DewPointFarenheit', 'RelativeHumidity',
                                  'WindSpeed', 'WindDirection', 'StationPressure', 'HourlyPrecip'],
                         parse_dates=True
                         )
        df = df.apply(pd.to_numeric, errors='coerce')
        if not os.path.isfile(hourlyDataFileName):
            df.to_hdf(hourlyDataFileName, 'hourlyData', format='t', mode='w', complevel=1, complib='zlib')
        else:
            df.to_hdf(hourlyDataFileName, 'hourlyData', format='t', mode='a', append=True, complevel=1, complib='zlib')
