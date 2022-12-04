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
from datautil.dataset import ProcessedEEGDataset
from model.conv_nets import ShallowConvNet
from train.helpers import run_training_experiment

PCA_DATA_PATH       = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/pca/'
KPCA_DATA_PATH      = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/kpca/'
FRONTAL_DATA_PATH   = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/frontal/'
TEMPORAL_DATA_PATH  = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/temporal/'
PARIETAL_DATA_PATH  = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/parietal/'
OCCIPITAL_DATA_PATH = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/occipital/'
MODEL_PATH          = '/WAVE/users/unix/smadsen/Desktop/bci_final/models/'


# PCA
data_path = '../data/train'
train_data_files = glob.glob( data_path+ '/*.data')
if len(train_data_files) == 0:
  fetch_trials_from_data(PCA_DATA_PATH + 'train/', data_path)


train_data_files = glob.glob( data_path+ '/*.data')
print(f'Found {len(train_data_files)} files for splitting') 

window_indices = get_window_indices()
train_files, valid_files = train_test_split(train_data_files, train_size=0.8, shuffle=True, random_state=42)

train_dataset = ProcessedEEGDataset(train_files, window_indices)
valid_dataset = ProcessedEEGDataset(valid_files, window_indices)
train_loader = DataLoader(train_dataset, batch_size=90, shuffle=True, pin_memory=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=120, shuffle=True, pin_memory=True, num_workers=2)

input_shape = np.expand_dims(train_dataset[0][0], 0).shape

model = ShallowConvNet()
summary(model, input_shape, col_names = ('input_size', 'output_size', 'num_params'), verbose = 0)

criterion = nn.CrossEntropyLoss() # softmax included here 
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

global device 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Running on device: {device}')

run_training_experiment(experiment_path=MODEL_PATH,
						model_name='ShallowConvNet', 
                        dataset_name='PCA',
                        trial_name='20Epochs', 
                        model=model, 
                        epochs=20, 
                        criterion=criterion, 
                        optimizer=optimizer,
						device=device,
                        train_loader=train_loader,
                        valid_loader=valid_loader)


