import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchinfo import summary
import os
import os.path as op
import glob
import gc
import pickle

from datautil.utils import fetch_trials_from_data, get_window_indices
from datautil.dataset import ChannelSelectedDataset
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

CHANNEL_DICT_PATH   = '../sample_data/channel_dict.dat'
MODEL_PATH          = '../models/'

data_path = '../sample_data/test'

channel_dict = pickle.load(open(CHANNEL_DICT_PATH, 'rb'))

model_list= [
  ('ShallowConvNet', '../models/to_test/shallow/'),
  ('DeepConvNet',    '../models/to_test/deep/'),
  ('EEGNet',         '../models/to_test/eegnet/'),
  ('EEGInception',   '../models/to_test/inception/')
]

dataset_list = [
  ('frontal',   '', 'F'),
  ('parietal',  '', 'P'),
  ('reduced_parietal',  '', 'P'),
]

parietal_selections = ['CCP4h', 'CCP3h', 
              'CP2', 'CCP2h', 'CCP1h', 
              'CP3', 'Pz', 'CPz', 'CPP1h', 
              'CPP3h', 'CP1', 'P1', 'P2', 
              'CCP6h', 'CPP4h']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on device: {device}')

for dataset_name, _, channel_loc in dataset_list:
  print(f'Testing on {dataset_name}')
  test_data_files = glob.glob(data_path+ '/*.data')
  window_indices = np.array([0])

  selections = []
  for ch in channel_dict:
    if channel_loc in ch:
      selections.append(channel_dict[ch])
  selected_channels = np.array(selections)

  if 'reduced' in dataset_name:
    model_list= [
       ('ShallowConvNet', '../models/to_test/shallow/'),
    ]
    selections = []
    for ch in channel_dict:
      if ch in parietal_selections:
        selections.append(channel_dict[ch])
    selected_channels = np.array(selections)

  

  for model_name, weights_path in model_list:
    print(f'Model: {model_name}')

    if 'Inception' in model_name:
      test_dataset = ChannelSelectedDataset(test_data_files, window_indices, selected_channels, features_first=True)
      model = EEGInception(input_channel=test_dataset.channel_count)
    elif 'EEGNet' in model_name:
      test_dataset = ChannelSelectedDataset(test_data_files, window_indices, selected_channels, time_last=True)
      model = EEGNet(input_channels=test_dataset.channel_count, linear_layer=1125)
    elif 'Shallow' in model_name:
      test_dataset = ChannelSelectedDataset(test_data_files, window_indices, selected_channels)
      model = ShallowConvNet(input_channels=test_dataset.channel_count, hidden_channels=60, linear_layer=4140)
    elif 'Deep' in model_name:
      test_dataset = ChannelSelectedDataset(test_data_files, window_indices, selected_channels)
      model = DeepConvNet(input_channels=test_dataset.channel_count, linear_layer=1800)
    else:
      print('Uh oh. We don\'t have that model yet!')
      continue

    
    test_loader = DataLoader(test_dataset, batch_size=30, shuffle=True, pin_memory=True, num_workers=0)
    input_shape = np.expand_dims(test_dataset[0][0], 0).shape
    # print(summary(model, input_shape, col_names = ('input_size', 'output_size', 'num_params'), verbose = 0))
    state_dict_path = find_state_dict(dataset_name, weights_path)
    # print(f'Model found at {state_dict_path}')
    model.load_state_dict(torch.load(state_dict_path, map_location=device))

    model.to(device)
    test_acc = test_accuracy(model, test_loader, device)
    

    print(f'#############################################################')
    print(f'# {model_name} - {dataset_name}                   ')
    print(f'# Test Acc.:  {np.max(test_acc)}                      ')
    print(f'#############################################################')


