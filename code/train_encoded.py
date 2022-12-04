import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchinfo import summary
import os
import os.path as ops
import glob
import gc
import pickle

from datautil.utils import encode_trials_from_data, get_window_indices
from datautil.dataset import ProcessedEEGDataset, EEGNetDataset, EEGInceptionDataset
from model.conv_nets import DeepConvNet, ShallowConvNet
from model.eeg_net import EEGNet
from model.inception import EEGInception
from model.autoencoder import LSTMAutoEncoder
from train.helpers import run_training_experiment, run_validation
from metrics.plots import plot_loss
from metrics.testing import test_accuracy

PCA_DATA_PATH       = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/pca/'
KPCA_DATA_PATH      = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/kpca/'
BASELINE_DATA_PATH  = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/baseline/'
LSTM_AE_PATH        = '/WAVE/users/unix/smadsen/Desktop/bci_final/models/AutoEncoder/autoencoder'
MODEL_PATH          = '/WAVE/users/unix/smadsen/Desktop/bci_final/models/'
PLOTS_PATH          = '/WAVE/users/unix/smadsen/Desktop/bci_final/plots/'

data_path = '../data/encoded'

model_list= [
  ('ShallowConvNet', ShallowConvNet(input_channels=30, hidden_channels=60, linear_layer=4140), 0.00005),
  ('DeepConvNet',    DeepConvNet(input_channels=30, linear_layer=1800), 0.00001),
  ('EEGNet',         EEGNet(input_channels=30, linear_layer=1125), 0.00001),
  ('EEGInception',   EEGInception(input_channel=30), 0.000001)
]

dataset_list = [
  ('Encoded', BASELINE_DATA_PATH)
]

autoencoder = LSTMAutoEncoder(hidden_size=30)
autoencoder.load_state_dict(torch.load(LSTM_AE_PATH))
encoder = autoencoder.encoder
for p in encoder.parameters():
  p.requires_grad=False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss() # softmax included here 
betas = (0.9, 0.99)
weight_decay=0.01

print(f'Running on device: {device}')

for dataset_name, train_data_path in dataset_list:
  print(f'Training on {dataset_name}')
  train_data_files = glob.glob(data_path+ '/*.data')
  if len(train_data_files) == 0:
  	encode_trials_from_data(train_data_path +'train/', data_path, encoder)
  	train_data_files = glob.glob(data_path+ '/*.data')
  window_indices = np.array([0])
  train_files, valid_files = train_test_split(train_data_files, train_size=0.8, shuffle=True, random_state=42)
  
  for model_name, model, learning_rate in model_list:
    print(f'Model: {model_name}')
    print(f'LR: {learning_rate} Betas: {betas} Weight Decay (L2): {weight_decay}') 
    trial_name = f'{learning_rate}_{betas}_{weight_decay}'

    if 'Inception' in model_name:
      train_dataset = EEGInceptionDataset(train_files, window_indices, window_size=1126)
      valid_dataset = EEGInceptionDataset(valid_files, window_indices, window_size=1126)
    elif 'EEGNet' in model_name:
      train_dataset = EEGNetDataset(train_files, window_indices, window_size=1126)
      valid_dataset = EEGNetDataset(valid_files, window_indices, window_size=1126)
    elif 'Shallow' in model_name:
      train_dataset = ProcessedEEGDataset(train_files, window_indices, window_size=1126)
      valid_dataset = ProcessedEEGDataset(valid_files, window_indices, window_size=1126)
    elif 'Deep' in model_name:
      train_dataset = ProcessedEEGDataset(train_files, window_indices, window_size=1126)
      valid_dataset = ProcessedEEGDataset(valid_files, window_indices, window_size=1126)
    else:
      print('Uh oh. We don\'t have that model yet!')
      continue

    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, pin_memory=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=40, shuffle=True, pin_memory=True, num_workers=2)

    input_shape = np.expand_dims(train_dataset[0][0], 0).shape
    print(summary(model, input_shape, col_names = ('input_size', 'output_size', 'num_params'), verbose = 0))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    loss, vloss, vacc = run_training_experiment(experiment_path=MODEL_PATH,
						model_name=model_name, 
                        dataset_name=dataset_name,
                        trial_name=trial_name, 
                        model=model, 
                        epochs=500, 
                        criterion=criterion, 
                        optimizer=optimizer,
						device=device,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        early_stop=30)
    

    epochs = list(range(len(loss)))

    print(f'#############################################################')
    print(f'# {model_name} - {dataset_name}                   ')
    print(f'# Val. Acc.:  {np.max(vacc)}                      ')
    print(f'# Epochs:     {len(loss) + 1}                     ')
    print(f'# LR:         {learning_rate}                     ')
    print(f'# L2:         {weight_decay}                      ')
    print(f'# Betas:      {betas}                             ')
    print(f'#############################################################')

    plot_loss(epochs, loss, vloss, model_name, dataset_name, trial_name, PLOTS_PATH)


