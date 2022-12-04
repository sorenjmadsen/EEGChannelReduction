import glob
import os
import pickle
import numpy as np
import torch

def fetch_trials_from_data(data_path, local_dir):
  subject_files = glob.glob((f'{data_path}/*.edf.data'))
  print(subject_files)
  top_level = os.path.split(data_path)[-1]
  for file in subject_files:
    subject = os.path.split(file)[-1]
    new_dir = subject.split('.')[0]
    os.makedirs(local_dir, exist_ok = True)
    print(file)
    X_y = pickle.load(open(file, 'rb'))
    for i, data in enumerate(X_y):
      if data[0].shape[1] < 1126:
        print(f'Issue with trial{i} in {file}')
      pickle.dump(data, open(os.path.join(local_dir, 'subject' + new_dir +' _' + str(i)+ '.data'), 'wb'))

def encode_trials_from_data(data_path, local_dir, encoder):
  subject_files = glob.glob((f'{data_path}/*.edf.data'))
  print(subject_files)
  top_level = os.path.split(data_path)[-1]
  for file in subject_files:
    subject = os.path.split(file)[-1]
    new_dir = subject.split('.')[0]
    os.makedirs(local_dir, exist_ok = True)
    print(file)
    X_y = pickle.load(open(file, 'rb'))
    for i, data in enumerate(X_y):
      if data[0].shape[1] < 1126:
        print(f'Issue with trial{i} in {file}')
      enc, _ = encoder(torch.from_numpy(data[0].T).float())
      data = (enc.numpy().T, data[1])
      pickle.dump(data, open(os.path.join(local_dir, 'subject' + new_dir +' _' + str(i)+ '.data'), 'wb'))

def get_window_indices(trial_len=1126, window_size=500):
  total_windows = trial_len-window_size
  window_indices = np.arange(total_windows)
  return window_indices
