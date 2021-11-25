
"""
inference module
Copyright Intelense 2021

"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch
import torch.autograd as autograd
from torchvision.utils import save_image

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')
#plt.rcParams["date.autoformatter.minute"] = "%H:%M:%S"
from matplotlib import style
style.use("ggplot")
import model
import sys
import warnings
warnings.filterwarnings("ignore")


class SignalDataset(Dataset):
    def __init__(self, path):
        self.signal_df = pd.read_csv(path)
        self.signal_columns = self.make_signal_list()
        self.make_rolling_signals()

    def make_signal_list(self):
        signal_list = list()
        for i in range(-50, 50):
            signal_list.append('ENGINE_COOLANT_TEMP'+str(i))
        return signal_list

    def make_rolling_signals(self):
        for i in range(-50, 50):
            self.signal_df['ENGINE_COOLANT_TEMP'+str(i)] = self.signal_df['ENGINE_COOLANT_TEMP'].shift(i)
        self.signal_df = self.signal_df.dropna()
        self.signal_df = self.signal_df.reset_index(drop=True)

    def __len__(self):
        return len(self.signal_df)

    def __getitem__(self, idx):
        row = self.signal_df.loc[idx]
        x = row[self.signal_columns].values.astype(float)
        x = torch.from_numpy(x)
        return {'ENGINE_COOLANT_TEMP':x}
    
    
def test(test_loader, encoder, decoder, critic_x,batch_size):
    reconstruction_error = list()
    critic_score = list()
    y_true = list()

    for batch, sample in enumerate(test_loader):
    #for sample in enumerate(test_loader):
        reconstructed_signal = decoder(encoder(sample['ENGINE_COOLANT_TEMP']))
        reconstructed_signal = torch.squeeze(reconstructed_signal)

        for i in range(0, batch_size):
            x_ = reconstructed_signal[i].detach().numpy()
            x = sample['ENGINE_COOLANT_TEMP'][i].numpy()
            #y_true.append(int(sample['anomaly'][i].detach()))
            reconstruction_error.append(dtw_reconstruction_error(x, x_))
        critic_score.extend(torch.squeeze(critic_x(sample['ENGINE_COOLANT_TEMP'])).detach().numpy())

    reconstruction_error = stats.zscore(reconstruction_error)
    critic_score = stats.zscore(critic_score)
    anomaly_score = reconstruction_error * critic_score

    return anomaly_score

  

#Other error metrics - point wise difference, Area difference.
def dtw_reconstruction_error(x, x_):
    n, m = x.shape[0], x_.shape[0]
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(x[i-1] - x_[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n][m]

def unroll_signal(x):
    x = np.array(x).reshape(100)
    return np.median(x)


###############plot anomaly #####################################################
   
def plot_anomaly(anomaly_score):
   
    THRESHOLD = 2.0
    # #THRESHOLD = 0.5
    # # Including reconstructed predictions
    # #print(len(anomaly_score))
    # #print(anomaly_score)
    # # Create the dataframe
    # #anomaly_results_df = pd.DataFrame()
    
    # anomaly_results_df = pd.DataFrame(index=range(len(anomaly_score)))
    # #anomaly_results_df.index = dataset.timestamp[0:len(anomaly_score)]

    # anomaly_results_df['ENGINE_COOLANT_TEMP'] = dataset.ENGINE_COOLANT_TEMP
    # anomaly_results_df['score'] = anomaly_score.tolist()
    # #anomaly_results_df['deviation'] = anomaly_score
    # anomaly_results_df['threshold'] = THRESHOLD
    # anomaly_results_df['anomaly'] = anomaly_results_df['score'].apply(lambda dev: 1 if dev > THRESHOLD else 0)


    # anomalies = anomaly_results_df[anomaly_results_df['anomaly'] == 1]
    # #anomalies.shape
    # #anomaly_results_df[['score', 'threshold']].plot(figsize=(14, 6))
    # #plt.show()
    # #plt.savefig('anomaly.png', bbox_inches='tight')

    # ######## Visualise Anomaly Detection #########################################
    # plt.figure(1,figsize=(15,10))
    
    # plt.subplot(211)
    # plt.plot(anomaly_results_df.index, anomaly_results_df.score, label='anomaly_score')
    # plt.plot(anomaly_results_df.index, anomaly_results_df.threshold, label='threshold')
    # #plt.plot(anomaly_results_df.index, anomaly_results_df.signal, label='original-signal')
    # plt.xticks(rotation=90)
    # plt.xlabel("Timesteps")
    # plt.ylabel("Score")
    # plt.title("Anomaly_score- Threshold is= {}".format(THRESHOLD))
    # plt.legend()

    # #plt.savefig('anomaly-detection.png', bbox_inches='tight')
    # plt.subplot(212)

    # plt.plot(range(len(anomaly_score)),anomaly_results_df['score'],label='anomaly_score')
   
    # sns.scatterplot(anomalies.index,anomalies.score,marker='*',color='blue',s=200,label='detected_anomaly')

    # #plt.plot(range(len(anomaly_score)),anomaly_results_df['ENGINE_COOLANT_TEMP'],label='original-ENGINE_COOLANT_TEMP')
  
    # plt.plot(anomaly_results_df.index, anomaly_results_df.threshold, label='threshold')

    # plt.xticks(rotation=90)
    # plt.xlabel("Timesteps")
    # plt.ylabel("Score")
    # plt.title("Detected_anomaly Threshold is= {}".format(THRESHOLD))
    # plt.legend()
    
    # plt.subplots_adjust(hspace=0.5)

    #plt.savefig('anomaly-detection.png', bbox_inches='tight') 

    #anomaly_results_df[['signal']].plot(figsize=(14, 6))
    # sns.scatterplot(anomalies.index, anomalies['signal'],label='anomaly',color='red')
    # plt.savefig('anomalysignal.png', bbox_inches='tight')
    # plt.show()
    ######## Visualise Anomaly Detection ###################################
    # Setting index after N timesteps from past in test_df
    #dataset=  dataset.set_index("TIME")
    anomaly_results_df = dataset[:batch_size][['ENGINE_COOLANT_TEMP']].copy()
    anomaly_results_df.index = dataset[:batch_size].index
    # Including reconstructed predictions
    anomaly_results_df['deviation'] = anomaly_score.tolist()
    anomaly_results_df['threshold'] = THRESHOLD
    anomaly_results_df['anomaly'] = anomaly_results_df['deviation'].apply(lambda dev: 1 if dev > THRESHOLD else 0)
    anomalies = anomaly_results_df[anomaly_results_df['anomaly'] == 1]
    anomaly_results_df[['ENGINE_COOLANT_TEMP']].plot(figsize=(14, 6))
    sns.scatterplot(anomalies.index, anomalies['ENGINE_COOLANT_TEMP'],label='anomaly',color='blue')
    plt.xticks(rotation=90)
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.title("Anomaly_score- Threshold is= {}".format(THRESHOLD))
    plt.legend()
    plt.savefig('anomaly.png', bbox_inches='tight') 

##################### Hyperparameters ######################################

batch_size = 64
lr = 0.00001

n_critics= 5
latent_space_dim = 20
signal_shape = 100

######################################## load data ####################################

dataset = pd.read_csv('test_dataset.csv')
dataset=  dataset.set_index("TIME")
test_dataset = SignalDataset(path= 'test_dataset.csv')
test_loader = DataLoader(test_dataset,batch_size=batch_size, drop_last=True)

print('\n ------Number of test datapoints is {}------\n'.format(len(dataset)))
print('\n-------Detacting anomaly on {} datapoints-----\n'.format(batch_size))

########################### load model ##############################################

encoder_path = 'models/encoder.pt'
decoder_path = 'models/decoder.pt'
critic_x_path = 'models/critic_x.pt'
critic_z_path = 'models/critic_z.pt'

encoder = model.Encoder(encoder_path, signal_shape)
decoder = model.Decoder(decoder_path, signal_shape)
critic_x = model.CriticX(critic_x_path, signal_shape)
critic_z = model.CriticZ(critic_z_path)


encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))
critic_x.load_state_dict(torch.load(critic_x_path))
critic_z.load_state_dict(torch.load(critic_z_path))


anomaly_score = test(test_loader, encoder, decoder, critic_x,batch_size)
plot_anomaly(anomaly_score)    