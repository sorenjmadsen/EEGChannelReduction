import torch
import torch.nn as nn

# Activations used in the model
class Square(nn.Module):
  '''
  Point-wise squaring of a tensor
  '''
  def __init__(self):
    super(Square, self).__init__()
  def forward(self, x):
    return torch.pow(x, 2)
  
class Log(nn.Module):
  '''
  Point-wise natural log of a tensor
  '''
  def __init__(self):
    super(Log, self).__init__()
  def forward(self, x):
    return torch.log(x)
