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
from model.conv_nets import DeepConvNet
from train.helpers import run_training_experiment
from metrics.plots import plot_loss

PCA_DATA_PATH       = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/pca/'
KPCA_DATA_PATH      = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/kpca/'
FRONTAL_DATA_PATH   = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/frontal/'
TEMPORAL_DATA_PATH  = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/temporal/'
PARIETAL_DATA_PATH  = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/parietal/'
OCCIPITAL_DATA_PATH = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/occipital/'
MODEL_PATH          = '/WAVE/users/unix/smadsen/Desktop/bci_final/models/'
PLOTS_PATH          ='/WAVE/users/unix/smadsen/Desktop/bci_final/plots/'

data_path = '../data/deep/train'

dataset_list = [
  ('PCA', PCA_DATA_PATH),
  ('KPCA-linear', KPCA_DATA_PATH + 'linear/'),
  ('KPCA-poly', KPCA_DATA_PATH + 'poly/'),
  ('KPCA-rbf', KPCA_DATA_PATH + 'rbf/'),
  ('KPCA-sigmoid', KPCA_DATA_PATH + 'sigmoid/'),
  ('KPCA-cosine', KPCA_DATA_PATH + 'cosine/')
]

criterion = nn.CrossEntropyLoss() # softmax included here 
learning_rate = 0.0001
betas = (0.9, 0.95)
weight_decay=0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on device: {device}')
print(f'LR: {learning_rate} Betas: {betas} Weight Decay (L2): {weight_decay}') 
trial_name = f'{learning_rate}_{betas}_{weight_decay}'
model_name='DeepConvNet'

for dataset_name, train_data_path in dataset_list:
  print(f'Training on {dataset_name}')
  fetch_trials_from_data(train_data_path +'train/', data_path)
  train_data_files = glob.glob(data_path+ '/*.data')

  window_indices = np.array([0])
  train_files, valid_files = train_test_split(train_data_files, train_size=0.8, shuffle=True, random_state=42)

  train_dataset = ProcessedEEGDataset(train_files, window_indices, window_size=1126)
  valid_dataset = ProcessedEEGDataset(valid_files, window_indices, window_size=1126)
  train_loader = DataLoader(train_dataset, batch_size=90, shuffle=True, pin_memory=True, num_workers=2)
  valid_loader = DataLoader(valid_dataset, batch_size=120, shuffle=True, pin_memory=True, num_workers=2)

  input_shape = np.expand_dims(train_dataset[0][0], 0).shape

  model = DeepConvNet(linear_layer=1800)
  print(summary(model, input_shape, col_names = ('input_size', 'output_size', 'num_params'), verbose = 0))
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

  loss, vloss, vacc = run_training_experiment(experiment_path=MODEL_PATH,
						model_name=model_name, 
                        dataset_name=dataset_name,
                        trial_name=trial_name, 
                        model=model, 
                        epochs=300, 
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

  epochs = list(range(len(loss)))
  plot_loss(epochs, loss, vloss, model_name, dataset_name, trial_name, PLOTS_PATH)


