from torch.utils.data import Dataset, DataLoader
import torch
import os
import pickle
import numpy as np

class ProcessedEEGDataset(Dataset):
  def __init__(self, trial_files, window_indices, num_classes=4, num_trials=320, window_size=500):
    self.trial_files = trial_files
    print(f'Found {len(self.trial_files)} trials')
    self.num_trials = len(self.trial_files)
    self.window_indices = window_indices
    self.num_windows = len(window_indices)
    self.window_size = window_size
    self.num_classes = num_classes

  def __len__(self):
    return self.num_trials * self.num_windows

  def __getitem__(self, idx):
    trial_idx = int(idx/self.num_windows)
    trial_file = self.trial_files[trial_idx]
    window = self.window_indices[idx % self.num_windows]

    X_y = pickle.load(open(os.path.join(trial_file), 'rb'))
    label = X_y[1]
    ohe = np.zeros(self.num_classes)
    ohe[label] += 1
    data = X_y[0][:, window: window + self.window_size].T

    if data.shape[0] < self.window_size:
      print(f'Issue with trial {trial_file} at window {window}')
      
    data = np.expand_dims(data, 0)
    return torch.from_numpy(data), torch.from_numpy(ohe)

class EEGNetDataset(Dataset):
  '''
      Permutes the data befire returning so time index is last.
  '''
  def __init__(self, trial_files, window_indices, num_classes=4, num_trials=320, window_size=500):
    self.trial_files = trial_files
    print(f'Found {len(self.trial_files)} trials')
    self.num_trials = len(self.trial_files)
    self.window_indices = window_indices
    self.num_windows = len(window_indices)
    self.window_size = window_size
    self.num_classes = num_classes

  def __len__(self):
    return self.num_trials * self.num_windows

  def __getitem__(self, idx):
    trial_idx = int(idx/self.num_windows)
    trial_file = self.trial_files[trial_idx]
    window = self.window_indices[idx % self.num_windows]

    X_y = pickle.load(open(os.path.join(trial_file), 'rb'))
    label = X_y[1]
    ohe = np.zeros(self.num_classes)
    ohe[label] += 1
    data = X_y[0][:, window: window + self.window_size].T

    if data.shape[0] < self.window_size:
      print(f'Issue with trial {trial_file} at window {window}')
      
    data = np.expand_dims(data, 0)
    return torch.from_numpy(data).permute(0, 2, 1), torch.from_numpy(ohe)

class EEGInceptionDataset(Dataset):
  '''
      Permutes the data befire returning so time index is last and feature_dim is first.
      [1, T, D] -> [D, 1, T]
  '''
  def __init__(self, trial_files, window_indices, num_classes=4, num_trials=320, window_size=500):
    self.trial_files = trial_files
    print(f'Found {len(self.trial_files)} trials')
    self.num_trials = len(self.trial_files)
    self.window_indices = window_indices
    self.num_windows = len(window_indices)
    self.window_size = window_size
    self.num_classes = num_classes

  def __len__(self):
    return self.num_trials * self.num_windows

  def __getitem__(self, idx):
    trial_idx = int(idx/self.num_windows)
    trial_file = self.trial_files[trial_idx]
    window = self.window_indices[idx % self.num_windows]

    X_y = pickle.load(open(os.path.join(trial_file), 'rb'))
    label = X_y[1]
    ohe = np.zeros(self.num_classes)
    ohe[label] += 1
    data = X_y[0][:, window: window + self.window_size].T

    if data.shape[0] < self.window_size:
      print(f'Issue with trial {trial_file} at window {window}')
      
    data = np.expand_dims(data, 0)
    return torch.from_numpy(data).permute(2, 0, 1), torch.from_numpy(ohe)

class ChannelSelectedDataset(Dataset):
  def __init__(self, trial_files, window_indices, selected_channels, zero_idx=None, num_classes=4, window_size=1126, time_last=False, features_first=False):
    self.trial_files = trial_files
    print(f'Found {len(self.trial_files)} trials')
    self.num_trials = len(self.trial_files)
    self.window_indices = window_indices
    self.num_windows = len(window_indices)
    self.window_size = window_size
    self.num_classes = num_classes
    self.selected_channels = selected_channels
    self.channel_count = len(selected_channels)
    self.zero_idx = zero_idx
    print(f'Selecting {self.channel_count} of 128')
    self.time_last = time_last
    self.features_first = features_first

  def __len__(self):
    return self.num_trials * self.num_windows

  def __getitem__(self, idx):
    trial_idx = int(idx/self.num_windows)
    trial_file = self.trial_files[trial_idx]
    window = self.window_indices[idx % self.num_windows]

    X_y = pickle.load(open(os.path.join(trial_file), 'rb'))
    label = X_y[1]
    ohe = np.zeros(self.num_classes)
    ohe[label] += 1
    data = X_y[0][:, window: window + self.window_size].T

    if data.shape[0] < self.window_size:
      print(f'Issue with trial {trial_file} at window {window}')

    data = data[:, self.selected_channels]

    if self.zero_idx is not None:
      data[:, self.zero_idx] = data[:, self.zero_idx]*0
      
    data = np.expand_dims(data, 0)
    if self.time_last:        # EEGNet
      return torch.from_numpy(data).permute(0, 2, 1), torch.from_numpy(ohe)
    if self.features_first:    # EEG Inception
      return torch.from_numpy(data).permute(2, 0, 1), torch.from_numpy(ohe)

    return torch.from_numpy(data), torch.from_numpy(ohe)

class EncodedDataset(Dataset):
  def __init__(self, trial_files, window_indices, encoder, num_classes=4, window_size=1126, time_last=False, features_first=False):
    self.trial_files = trial_files
    print(f'Found {len(self.trial_files)} trials')
    self.num_trials = len(self.trial_files)
    self.window_indices = window_indices
    self.num_windows = len(window_indices)
    self.window_size = window_size
    self.num_classes = num_classes
    self.encoder = encoder
    for p in self.encoder.parameters():
      p.requires_grad=False

    self.time_last = time_last
    self.features_first = features_first

  def __len__(self):
    return self.num_trials * self.num_windows

  def __getitem__(self, idx):
    trial_idx = int(idx/self.num_windows)
    trial_file = self.trial_files[trial_idx]
    window = self.window_indices[idx % self.num_windows]

    X_y = pickle.load(open(os.path.join(trial_file), 'rb'))
    label = X_y[1]
    ohe = np.zeros(self.num_classes)
    ohe[label] += 1
    data = X_y[0][:, window: window + self.window_size].T

    if data.shape[0] < self.window_size:
      print(f'Issue with trial {trial_file} at window {window}')

    data, _ = self.encoder(torch.from_numpy(data).float())
    data = np.expand_dims(data, 0)
    
    if self.time_last:        # EEGNet
      return data.permute(0, 2, 1), torch.from_numpy(ohe)
    if self.features_first:    # EEG Inception
      return data.permute(2, 0, 1), torch.from_numpy(ohe)

    return data, torch.from_numpy(ohe)

class AutoEncoderDataset(Dataset):
  def __init__(self, trial_files, num_classes=4, trial_len=1126, max_len=500):
    self.trial_files = trial_files
    print(f'Found {len(self.trial_files)} trials')
    self.num_trials = len(self.trial_files)
    self.max_len = max_len
    self.num_classes = num_classes
    self.trial_len = trial_len

  def pad_sequences(self, data):
    '''
      Pads sequences of EEG data:
        data : [N x M] -> [max_len x M]
      ----------
      Returns:
        data: zero-padded sample
        mask: array of 1s for loss
    '''
    data_len = data.shape[0]
    channels = data.shape[1]
    data = torch.transpose(data, 0, 1) # [M x N]
    pad = torch.zeros((channels, self.max_len - data_len)) # [M, max_len-N]
    data = torch.cat((data,pad), dim=-1)
    mask = torch.cat((torch.ones(data_len), torch.zeros(pad.shape[1])), dim=0)
    return torch.transpose(data, 0, 1), mask

  def __len__(self):
    return self.num_trials * 20

  def __getitem__(self, idx):
    trial_idx = int(idx/20)
    trial_file = self.trial_files[trial_idx]

    X_y = pickle.load(open(os.path.join(trial_file), 'rb'))
    label = X_y[1]
    ohe = np.zeros(self.num_classes)
    ohe[label] += 1
    window_size = np.random.randint(int(self.max_len/4)*3, self.max_len + 1)
    window = np.random.randint(0, self.trial_len - window_size)
    data = torch.from_numpy(X_y[0][:, window: window + window_size].T)
    data_len = data.shape[0]
    data, mask = self.pad_sequences(data)

    if data.shape[0] < window_size:
      print(f'Issue with trial {trial_file} at window {window}')
      
    return data.float(), mask.float()
