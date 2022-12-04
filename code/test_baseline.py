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
from datautil.dataset import ProcessedEEGDataset, EEGNetDataset, EEGInceptionDataset
from model.conv_nets import DeepConvNet, ShallowConvNet
from model.eeg_net import EEGNet
from model.inception import EEGInception
from train.helpers import run_training_experiment
from metrics.plots import plot_loss
from metrics.testing import test_accuracy

def find_state_dict(dataset_name, weights_path):
  models = glob.glob(weights_path+ '*')
  for m in models:
    if dataset_name in os.path.split(m)[-1][0:len(dataset_name)]:
      return m

PCA_DATA_PATH       = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/pca/'
KPCA_DATA_PATH      = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/kpca/'
BASELINE_DATA_PATH  = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/baseline/'
MODEL_PATH          = '/WAVE/users/unix/smadsen/Desktop/bci_final/models/'
PLOTS_PATH          = '/WAVE/users/unix/smadsen/Desktop/bci_final/plots/'

data_path = '../data/test'

model_list= [
  ('ShallowConvNet', ShallowConvNet(input_channels=128, hidden_channels=60, linear_layer=4140), 
                                                      '/WAVE/users/unix/smadsen/Desktop/bci_final/models/to_test/shallow/'),
  ('DeepConvNet',    DeepConvNet(input_channels=128, linear_layer=1800), 
                                                      '/WAVE/users/unix/smadsen/Desktop/bci_final/models/to_test/deep/'),
  ('EEGNet',         EEGNet(input_channels=128, linear_layer=1125), 
                                                      '/WAVE/users/unix/smadsen/Desktop/bci_final/models/to_test/eegnet/'),
  ('EEGInception',   EEGInception(input_channel=128),  
                                                      '/WAVE/users/unix/smadsen/Desktop/bci_final/models/to_test/inception/')
]

dataset_list = [
  ('Baseline', BASELINE_DATA_PATH)
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on device: {device}')

for dataset_name, test_data_path in dataset_list:
  print(f'Testing on {dataset_name}')
  fetch_trials_from_data(test_data_path +'test/', data_path)
  test_data_files = glob.glob(data_path+ '/*.data')
  window_indices = np.array([0])
  
  for model_name, model, weights_path in model_list:
    if 'Inception' in model_name:
      test_dataset = EEGInceptionDataset(test_data_files, window_indices, window_size=1126)
    elif 'EEGNet' in model_name:
      test_dataset = EEGNetDataset(test_data_files, window_indices, window_size=1126)
    else:
      test_dataset = ProcessedEEGDataset(test_data_files, window_indices, window_size=1126)

    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=True, pin_memory=True, num_workers=2)
    input_shape = np.expand_dims(test_dataset[0][0], 0).shape
    print(summary(model, input_shape, col_names = ('input_size', 'output_size', 'num_params'), verbose = 0))
    state_dict_path = find_state_dict(dataset_name, weights_path)
    print(f'Model found at {state_dict_path}')
    model.load_state_dict(torch.load(state_dict_path))

    model.to(device)
    test_acc = test_accuracy(model, test_loader, device)
    

    print(f'#############################################################')
    print(f'# {model_name} - {dataset_name}                   ')
    print(f'# Test Acc.:  {np.max(test_acc)}                      ')
    print(f'#############################################################')


