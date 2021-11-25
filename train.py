"""
data-preprocess module

Copyright Intelense 2021

"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
from torchvision.utils import save_image
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
style.use("ggplot")

import model
import sys
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(filename='train.log', level=logging.DEBUG)



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
        #return {'ENGINE_COOLANT_TEMP':x, 'anomaly':row['anomaly']}

def critic_x_iteration(sample):
    optim_cx.zero_grad()

    x = sample['ENGINE_COOLANT_TEMP'].view(1, batch_size, signal_shape)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape) * valid_x) #Wasserstein Loss

    #The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    x_ = decoder(z)
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape) * fake_x)  #Wasserstein Loss

    alpha = torch.rand(x.shape)
    ix = Variable(alpha * x + (1 - alpha) * x_) #Random Weighted Average
    ix.requires_grad_(True)
    v_ix = critic_x(ix)
    v_ix.mean().backward()
    gradients = ix.grad
    #Gradient Penalty Loss
    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    #Critic has to maximize Cx(Valid X) - Cx(Fake X).
    #Maximizing the above is same as minimizing the negative.
    wl = critic_score_fake_x - critic_score_valid_x
    loss = wl + gp_loss
    loss.backward()
    optim_cx.step()

    return loss

def critic_z_iteration(sample):
    optim_cz.zero_grad()

    x = sample['ENGINE_COOLANT_TEMP'].view(1, batch_size, signal_shape)
    z = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape) * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape) * fake_z) #Wasserstein Loss

    wl = critic_score_fake_z - critic_score_valid_z

    alpha = torch.rand(z.shape)
    iz = Variable(alpha * z + (1 - alpha) * z_) #Random Weighted Average
    iz.requires_grad_(True)
    v_iz = critic_z(iz)
    v_iz.mean().backward()
    gradients = iz.grad
    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    loss = wl + gp_loss
    loss.backward()
    optim_cz.step()

    return loss

def encoder_iteration(sample):
    optim_enc.zero_grad()

    x = sample['ENGINE_COOLANT_TEMP'].view(1, batch_size, signal_shape)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape) * valid_x) #Wasserstein Loss

    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    x_ = decoder(z)
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape) * fake_x)

    enc_z = encoder(x)
    gen_x = decoder(enc_z)

    mse = mse_loss(x.float(), gen_x.float())
    loss_enc = mse + critic_score_valid_x - critic_score_fake_x
    loss_enc.backward(retain_graph=True)
    optim_enc.step()

    return loss_enc

def decoder_iteration(sample):
    optim_dec.zero_grad()

    x = sample['ENGINE_COOLANT_TEMP'].view(1, batch_size, signal_shape)
    z = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape) * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape) * fake_z)

    enc_z = encoder(x)
    gen_x = decoder(enc_z)

    mse = mse_loss(x.float(), gen_x.float())
    loss_dec = mse + critic_score_valid_z - critic_score_fake_z
    loss_dec.backward(retain_graph=True)
    optim_dec.step()

    return loss_dec


def train(n_epochs,n_critics):
    logging.debug('Starting training')
    print('Starting training')
    cx_epoch_loss = list()
    cz_epoch_loss = list()
    encoder_epoch_loss = list()
    decoder_epoch_loss = list()
    
    encoder_losses = []
    decoder_losses = []
    criticx_loss = []
    criticz_loss = []
    epochs = []
    
    for epoch in range(n_epochs):
        logging.debug('Epoch {}'.format(epoch))
        
        n_critics = n_critics

        cx_nc_loss = list()
        cz_nc_loss = list()

        for i in range(n_critics):
            cx_loss = list()
            cz_loss = list()

            for batch, sample in enumerate(train_loader):
                loss = critic_x_iteration(sample)
                cx_loss.append(loss)

                loss = critic_z_iteration(sample)
                cz_loss.append(loss)

            cx_nc_loss.append(torch.mean(torch.tensor(cx_loss)))
            cz_nc_loss.append(torch.mean(torch.tensor(cz_loss)))

        logging.debug('Critic training done in epoch {}'.format(epoch))
        print('Critic training done in epoch {}'.format(epoch))

        encoder_loss = list()
        decoder_loss = list()

        for batch, sample in enumerate(train_loader):
            enc_loss = encoder_iteration(sample)
            dec_loss = decoder_iteration(sample)
            encoder_loss.append(enc_loss)
            decoder_loss.append(dec_loss)

        cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
        cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
        encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
        decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))

        encoder_losses.append(torch.mean(torch.tensor(encoder_loss)))
        decoder_losses.append(torch.mean(torch.tensor(decoder_loss)))
        criticx_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
        criticz_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
        epochs.append(epoch)   

        logging.debug('Encoder decoder training done in epoch {}'.format(epoch))
        logging.debug('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))
        
        print('Encoder decoder training done in epoch {}'.format(epoch))
        print('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))



        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), encoder.encoder_path)
            torch.save(decoder.state_dict(), decoder.decoder_path)
            torch.save(critic_x.state_dict(), critic_x.critic_x_path)
            torch.save(critic_z.state_dict(), critic_z.critic_z_path)

    return  encoder_losses, decoder_losses, criticx_loss, criticz_loss, epochs     

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="number of epochs of training")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")

    parser.add_argument("--lr", type=float, default= 0.00001,
                        help="adam: learning rate")
    
    parser.add_argument("--n_critics", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")

    parser.add_argument("--latent_dim", type=int, default= 20,
                        help="dimensionality of the latent space")

    parser.add_argument("--signal_shape", type=int, default= 100,
                        help="shape of signal")                    

    opt = parser.parse_args()

  
    
    ##################### Hyperparameters ######################################

    batch_size = opt.batch_size
    lr = opt.lr
    n_epochs= opt.n_epochs
    n_critics= opt.n_critics
    latent_space_dim = opt.latent_dim
    signal_shape = opt.signal_shape

    #############################################################################

    dataset = pd.read_csv('df_COOLANT_SEC.csv')
    #dataset=  dataset.set_index("TIME")

    # dataset[['ENGINE_COOLANT_TEMP']] = dataset[['ENGINE_COOLANT_TEMP']].apply(
    #                        lambda x: StandardScaler().fit_transform(x))
    
    # # I'm selecting only numericals to scale
    # numerical = dataset.select_dtypes(include='float64').columns
    # # This will transform the selected columns and merge to the original data frame
    # dataset.loc[:,numerical] = StandardScaler().fit_transform(dataset.loc[:,numerical])

    scaler = MinMaxScaler()
    dataset['ENGINE_COOLANT_TEMP'] = scaler.fit_transform(dataset['ENGINE_COOLANT_TEMP'].values.reshape(-1,1))


    #Splitting intro train and test
    train_len = int(0.8 * dataset.shape[0])
    dataset[0:train_len].to_csv('train_dataset.csv', index=False)
    dataset[train_len:].to_csv('test_dataset.csv', index=False)

    train_dataset = SignalDataset(path='train_dataset.csv')
    test_dataset = SignalDataset(path='test_dataset.csv')
    
    print('\n--Number of train datapoints is = {}\n'.format(train_len))
    print('\n--Number of test datapoints is = {}\n'.format(len(dataset)-train_len))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    
    logging.info('Number of samples in train dataset {}'.format(len(train_dataset)))
    

    #print('Number of train datapoints is {}'.format(len(train_dataset)))

    #############save path ##############################################

    encoder_path = 'models/encoder.pt'
    decoder_path = 'models/decoder.pt'
    critic_x_path = 'models/critic_x.pt'
    critic_z_path = 'models/critic_z.pt'
    
    encoder = model.Encoder(encoder_path, signal_shape)
    decoder = model.Decoder(decoder_path, signal_shape)
    critic_x = model.CriticX(critic_x_path, signal_shape)
    critic_z = model.CriticZ(critic_z_path)

    mse_loss = torch.nn.MSELoss()

    optim_enc = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_dec = optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_cx = optim.Adam(critic_x.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_cz = optim.Adam(critic_z.parameters(), lr=lr, betas=(0.5, 0.999))

    #==================== call training ===========================================  

    encoder_losses, decoder_losses, criticx_loss, criticz_loss, epochs = train(n_epochs=n_epochs,n_critics=n_critics)
    
    #==================== Plot ===========================================    
    fig, (ax1, ax2,ax3, ax4) = plt.subplots(4, 1,figsize=(10,15))
    # make a little extra space between the subplots
    fig.subplots_adjust(hspace=0.2)


    ax1.plot(epochs, encoder_losses, label='Encoder loss')
    ax2.plot(epochs, decoder_losses, label='Decoder loss')
    ax3.plot(epochs, criticx_loss, label='criticx_loss')
    ax4.plot(epochs, criticz_loss, label='criticz_loss')


    # Set common labels
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')

    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Loss')
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    ax1.set_title("Encoder_loss (batch_size={}, lr={})".format(batch_size, lr))
    ax2.set_title("Decoder_loss (batch_size={}, lr={})".format(batch_size, lr))
    ax3.set_title("Criticx_loss (batch_size={}, lr={})".format(batch_size, lr))
    ax4.set_title("Criticz_loss (batch_size={}, lr={})".format(batch_size, lr))

    
    
    #plt.legend()
    plt.savefig('loss_epoch_%d.png' % n_epochs)
    print('\n--Training Complete----')
    
    
    
