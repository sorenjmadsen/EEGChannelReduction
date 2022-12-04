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

from datautil.utils import fetch_trials_from_data, get_window_indices
from datautil.dataset import AutoEncoderDataset
from model.autoencoder import LSTMAutoEncoder
from train.helpers_ae import run_training_experiment_ae, run_validation_ae
from metrics.plots import plot_loss

BASELINE_DATA_PATH  = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/baseline/'
MODEL_PATH          = '/WAVE/users/unix/smadsen/Desktop/bci_final/models/'
PLOTS_PATH          = '/WAVE/users/unix/smadsen/Desktop/bci_final/plots/'

data_path = '../data/train'

model_list= [
  ('LSTMAutoEncoder', LSTMAutoEncoder(hidden_size=32, num_layers=2), 0.0001),
]

dataset_list = [
  ('Baseline', BASELINE_DATA_PATH)
]

criterion = nn.MSELoss() 
weight_decay = 0.0
betas = (0.9, 0.999)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len=15

print(f'Running on device: {device}')

for dataset_name, train_data_path in dataset_list:
  print(f'Training on {dataset_name}')
  train_data_files = glob.glob(data_path+ '/*.data')
  if len(train_data_files) == 0:
  	fetch_trials_from_data(train_data_path +'train/', data_path)
  	train_data_files = glob.glob(data_path+ '/*.data')
  window_indices = np.array([0])
  train_files, valid_files = train_test_split(train_data_files, train_size=0.8, shuffle=True, random_state=42)
  
  for model_name, model, learning_rate in model_list:
    print(f'Model: {model_name}')
    print(f'LR: {learning_rate} Betas: {betas} Weight Decay (L2): {weight_decay}') 
    trial_name = f'{learning_rate}_{betas}_{weight_decay}'

    train_dataset = AutoEncoderDataset(train_files, max_len=max_len)
    valid_dataset = AutoEncoderDataset(valid_files, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, pin_memory=True, num_workers=6)
    valid_loader = DataLoader(valid_dataset, batch_size=80, shuffle=True, pin_memory=True, num_workers=6)
    
    model.set_device(device)
    model.to(device)

    input_shape=(5, 128)
    print(summary(model, input_shape, batch_dim=0, col_names = ('input_size', 'output_size', 'num_params'), verbose = 0))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    loss, vloss = run_training_experiment_ae(experiment_path=MODEL_PATH,
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
    print(f'# Val. Loss:  {np.max(vloss)}                      ')
    print(f'# Epochs:     {len(loss) + 1}                     ')
    print(f'# LR:         {learning_rate}                     ')
    print(f'# L2:         {weight_decay}                      ')
    print(f'# Betas:      {betas}                             ')
    print(f'#############################################################')

    plot_loss(epochs, loss, vloss, model_name, dataset_name, trial_name, PLOTS_PATH)


