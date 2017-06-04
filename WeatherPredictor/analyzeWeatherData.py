import pandas as pd
from matplotlib import pyplot as plt
import os

print("Running")
dataFile = os.getcwd() + '/hourlyData.hdf'

df = pd.read_hdf(dataFile, start=1, stop=100000, columns=['WBAN'])


