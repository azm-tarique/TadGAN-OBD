
"""
data-preprocess module



"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import pyplot

from matplotlib.pylab import rcParams
import seaborn as sns
rcParams['figure.figsize']=20,8
import seaborn; seaborn.set()

from sklearn.preprocessing import MinMaxScaler

import os
import sys
import warnings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')


df = pd.read_csv('torqueTrackLog.csv') 
df['GPS Time'] = pd.to_datetime(df['GPS Time']).dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3] 

df.rename(columns={'GPS Time': 'TIME', 'Engine Coolant Temperature(°C)': 'ENGINE_COOLANT_TEMP'}, inplace=True)

# select the col
df_COOLANT=df.loc[:,['TIME','ENGINE_COOLANT_TEMP']]

df_COOLANT['ENGINE_COOLANT_TEMP'] = pd.to_numeric(df_COOLANT['ENGINE_COOLANT_TEMP'],errors = 'coerce')
df_COOLANT['TIME']  = pd.to_datetime(df_COOLANT['TIME'] , errors='coerce')
df_COOLANT = df_COOLANT.set_index("TIME")
secondly_data = df_COOLANT.resample('1s').mean()

secondly_data.head()

# drop all rows with any NaN and NaT values
secondly_data.dropna(subset=['ENGINE_COOLANT_TEMP'],inplace=True)
#minutely frequency

secondly_data.plot()

plt.ylabel('secondly')
plt.savefig('secondly_data.png', bbox_inches='tight') 
secondly_data.isnull().sum()

# df1= secondly_data
# scaler=MinMaxScaler(feature_range=(0,1))
# df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

# save as csv
secondly_data.to_csv("df_COOLANT_SEC.csv")
secondly_data.to_csv("df_COOLANT_SEC2.csv")
#dataset = pd.read_csv('df_COOLANT_SEC.csv')
#Splitting intro train and test

# train_len = int(0.9* dataset.shape[0])
# dataset[0:train_len].to_csv('train_dataset.csv', index=False)
# dataset[train_len:].to_csv('test_dataset.csv', index=False)

# print('\n--Number of train datapoints is = {}\n'.format(train_len))
# print('\n--Number of test datapoints is = {}\n'.format(len(dataset)-train_len))