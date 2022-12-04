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

PCA_DATA_PATH       = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/pca/'
KPCA_DATA_PATH      = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/kpca/'
BASELINE_DATA_PATH  = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/baseline/'
CHANNEL_DICT_PATH   = '/WAVE/users/unix/smadsen/Desktop/bci_final/datasets/channel_dict.dat'
MODEL_PATH          = '/WAVE/users/unix/smadsen/Desktop/bci_final/models/'
PLOTS_PATH          = '/WAVE/users/unix/smadsen/Desktop/bci_final/plots/'

data_path = '../data/train'

channel_dict = pickle.load(open(CHANNEL_DICT_PATH, 'rb'))

model_list= [
  ('ShallowConvNet', '/WAVE/users/unix/smadsen/Desktop/bci_final/models/to_test/shallow/'),
]

dataset_list = [
  ('parietal_reduced',  BASELINE_DATA_PATH, 'P'),
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on device: {device}')

'''
    First, we perform some exploration on our validation set to see which electrodes
    contribute most to the performance of the model. Using the top 15 electrodes, we then
    retrain the network only using those channels in particular. We then evaluate this on
    our test set.
'''

selections = []

for dataset_name, train_data_path, channel_loc in dataset_list:
  print(f'Validation on {dataset_name}')
  #fetch_trials_from_data(test_data_path +'train/', data_path)train_data_files = glob.glob(data_path+ '/*.data')
  data_files = glob.glob(data_path+ '/*.data')
  if len(data_files) == 0:
  	fetch_trials_from_data(train_data_path +'train/', data_path)
  	data_files = glob.glob(data_path+ '/*.data')
  window_indices = np.array([0])
  train_files, valid_files = train_test_split(data_files, train_size=0.8, shuffle=True, random_state=42)
  window_indices = np.array([0])

  test_data_files=valid_files

  selections = []
  names = []
  zero_idxs = []
  zero_names = []
  for ch in channel_dict:
    if channel_loc in ch:
      selections.append(channel_dict[ch])
      names.append(ch)
    
  selected_channels = np.array(selections)

  acc_dict = {}

  accs = []

  for model_name, weights_path in model_list:
     for idx, name in enumerate(names):
      zero_idx=idx
      if 'Inception' in model_name:
        test_dataset = ChannelSelectedDataset(test_data_files, window_indices, selected_channels, zero_idx=zero_idx, features_first=True)
        model = EEGInception(input_channel=test_dataset.channel_count)
      elif 'EEGNet' in model_name:
        test_dataset = ChannelSelectedDataset(test_data_files, window_indices, selected_channels, zero_idx=zero_idx, time_last=True)
        model = EEGNet(input_channels=test_dataset.channel_count, linear_layer=1125)
      elif 'Shallow' in model_name:
        test_dataset = ChannelSelectedDataset(test_data_files, window_indices, selected_channels, zero_idx=zero_idx)
        model = ShallowConvNet(input_channels=test_dataset.channel_count, hidden_channels=60, linear_layer=4140)
      elif 'Deep' in model_name:
        test_dataset = ChannelSelectedDataset(test_data_files, window_indices, selected_channels, zero_idx=zero_idx)
        model = DeepConvNet(input_channels=test_dataset.channel_count, linear_layer=1800)
      else:
        print('Uh oh. We don\'t have that model yet!')
        continue

    
      test_loader = DataLoader(test_dataset, batch_size=40, shuffle=True, pin_memory=True, num_workers=2)
      state_dict_path = find_state_dict(dataset_name, weights_path)
      model.load_state_dict(torch.load(state_dict_path))

      model.to(device)
      test_acc = test_accuracy(model, test_loader, device)
      accs.append(test_acc)
      acc_dict[test_acc] = name
    

      print(f'#############################################################')
      print(f'  {model_name} - {dataset_name} w/o {name}                  ')
      print(f'  Val. Acc.:  {np.max(test_acc)}                      ')
      print(f'#############################################################')

  accs = np.array(accs)
  accs.sort()
  selections = []
  for i in range(15):
    print(f'Rank {i+1}: {acc_dict[accs[i]]} - {accs[i]}')
    selections.append(acc_dict[accs[i]])

  for i, name in enumerate(names):
    if name in selections:
      zero_idxs.append(i)
      zero_names.append(name)
  zero_idxs = np.array(zero_idxs)


  # Examine the validation set accuracy with all 15 electrodes missing.

  for model_name, weights_path in model_list:
    if True:
      name = zero_names
      zero_idx = zero_idxs
      if 'Inception' in model_name:
        test_dataset = ChannelSelectedDataset(test_data_files, window_indices, selected_channels, zero_idx=zero_idx, features_first=True)
        model = EEGInception(input_channel=test_dataset.channel_count)
      elif 'EEGNet' in model_name:
        test_dataset = ChannelSelectedDataset(test_data_files, window_indices, selected_channels, zero_idx=zero_idx, time_last=True)
        model = EEGNet(input_channels=test_dataset.channel_count, linear_layer=1125)
      elif 'Shallow' in model_name:
        test_dataset = ChannelSelectedDataset(test_data_files, window_indices, selected_channels, zero_idx=zero_idx)
        model = ShallowConvNet(input_channels=test_dataset.channel_count, hidden_channels=60, linear_layer=4140)
      elif 'Deep' in model_name:
        test_dataset = ChannelSelectedDataset(test_data_files, window_indices, selected_channels, zero_idx=zero_idx)
        model = DeepConvNet(input_channels=test_dataset.channel_count, linear_layer=1800)
      else:
        print('Uh oh. We don\'t have that model yet!')
        continue

    
      test_loader = DataLoader(test_dataset, batch_size=40, shuffle=True, pin_memory=True, num_workers=2)
      state_dict_path = find_state_dict(dataset_name, weights_path)
      model.load_state_dict(torch.load(state_dict_path))

      model.to(device)
      test_acc = test_accuracy(model, test_loader, device)

      print(f'#############################################################')
      print(f'  {model_name} - {dataset_name} w/o {name}                  ')
      print(f'  Val. Acc.:  {np.max(test_acc)}                      ')
      print(f'#############################################################')


data_path = '../data/train'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss() # softmax included here 
betas = (0.9, 0.95)
weight_decay=0.01
if len(selections) == 0:
  selections=['CCP4h', 'CCP3h', 'CP2', 'CCP2h', 'CCP1h', 'CP3', 'Pz', 'CPz', 'CPP1h', 'CPP3h', 'CP1', 'P1', 'P2', 'CCP6h', 'CPP4h']
print(selections)

selected_channels = []
for ch in channel_dict:
  if ch in selections:
    selected_channels.append(channel_dict[ch])
selected_channels = np.array(selected_channels)

'''
    Retrain ShallowConvNet again!
'''

for dataset_name, train_data_path, _ in dataset_list:
  print(f'Training on {dataset_name} with only {selections}')
  train_data_files = glob.glob(data_path+ '/*.data')
  if len(train_data_files) == 0:
  	fetch_trials_from_data(train_data_path +'train/', data_path)
  	train_data_files = glob.glob(data_path+ '/*.data')
  window_indices = np.array([0])


  train_files, valid_files = train_test_split(train_data_files, train_size=0.8, shuffle=True, random_state=42)

  model_list= [
    ('ShallowConvNet', 0.00005),
  ]
  
  for model_name, learning_rate in model_list:
    print(f'Model: {model_name}')
    print(f'LR: {learning_rate} Betas: {betas} Weight Decay (L2): {weight_decay}') 
    trial_name = f'{learning_rate}_{betas}_{weight_decay}'



    if 'Inception' in model_name:
      train_dataset = ChannelSelectedDataset(train_files, window_indices, selected_channels, features_first=True)
      valid_dataset = ChannelSelectedDataset(valid_files, window_indices, selected_channels, features_first=True)
      model = EEGInception(input_channel=train_dataset.channel_count)
    elif 'EEGNet' in model_name:
      train_dataset = ChannelSelectedDataset(train_files, window_indices, selected_channels, time_last=True)
      valid_dataset = ChannelSelectedDataset(valid_files, window_indices, selected_channels, time_last=True)
      model = EEGNet(input_channels=train_dataset.channel_count, linear_layer=1125)
    elif 'Shallow' in model_name:
      train_dataset = ChannelSelectedDataset(train_files, window_indices, selected_channels)
      valid_dataset = ChannelSelectedDataset(valid_files, window_indices, selected_channels)
      model = ShallowConvNet(input_channels=train_dataset.channel_count, hidden_channels=60, linear_layer=4140)
    elif 'Deep' in model_name:
      train_dataset = ChannelSelectedDataset(train_files, window_indices, selected_channels)
      valid_dataset = ChannelSelectedDataset(valid_files, window_indices, selected_channels)
      model = DeepConvNet(input_channels=train_dataset.channel_count, linear_layer=1800)
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

    #Testing time!

data_path = '../data/test'
model_list= [
  ('ShallowConvNet', '/WAVE/users/unix/smadsen/Desktop/bci_final/models/to_test/shallow/')
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on device: {device}')

for dataset_name, test_data_path, channel_loc in dataset_list:
  print(f'Testing on {dataset_name}')
  test_data_files = glob.glob(data_path+ '/*.data')
  if len(test_data_files) == 0:
  	fetch_trials_from_data(test_data_path +'test/', data_path)
  	test_data_files = glob.glob(data_path+ '/*.data')
  window_indices = np.array([0])

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


