import torch
import torch.nn as nn

from model.activations import Log, Square

class ShallowConvNet(nn.Module):
  def __init__(self, 
               input_channels=30,
               hidden_channels=40, 
               kernel_size_temporal_conv=25,  
               kernel_size_pool=75, 
               stride_pool=15, 
               linear_layer=1080,        
               non_linearity='square',
               log_non_linearity=True):
    super(ShallowConvNet,self).__init__()
    activation_name = 'Square'
    activation_layer = Square()

    self.model = nn.Sequential()
    self.model.add_module('Temporal Conv', 
                          nn.Conv2d(1, hidden_channels, kernel_size=(kernel_size_temporal_conv,1)))
    self.model.add_module('Spatial Filter', 
                          nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, input_channels)))
    self.model.add_module('Batch Normalization', nn.BatchNorm2d(hidden_channels))
    self.model.add_module(activation_name, activation_layer)
    self.model.add_module('Mean Pooling', nn.AvgPool2d(kernel_size=(kernel_size_pool,1), stride=(stride_pool,1)))

    if log_non_linearity:
      self.model.add_module('Log', Log())

    self.model.add_module('Flatten',nn.Flatten()) 
    self.model.add_module('Linear',nn.Linear(linear_layer, 4))

  def forward(self, x):
    return self.model(x)


class DeepConvNet(nn.Module):
  def __init__(self, 
               input_channels=30,
               hidden_channels1=25, 
               hidden_channels2=50, 
               hidden_channels3=100,
               hidden_channels4=200, 
               linear_layer=400,
               kernel_size_temporal_conv=10,  
               kernel_size_pool=3, 
               stride_pool=3,         
               non_linearity='elu',
               log_non_linearity=True):
    super(DeepConvNet,self).__init__()
    if non_linearity == 'elu':
      activation_name = 'ELU'
      activation_layer = nn.ELU(inplace=True)
    elif non_linearity == 'relu':
      activation_name = 'ReLU'
      activation_layer = nn.ReLU()
    elif non_linearity == 'gelu':
      activation_name = 'GELU'
      activation_layer = nn.GELU()
    else:
      activation_name = 'Square'
      activation_layer = Square()


    # Conv-Pool Block 1 
    self.model = nn.Sequential()
    self.model.add_module('Temporal Conv', 
                          nn.Conv2d(1, hidden_channels1, kernel_size=(kernel_size_temporal_conv,1)))
    self.model.add_module('Spatial Filter', 
                          nn.Conv2d(hidden_channels1, hidden_channels1, kernel_size=(1, input_channels)))
    
    
    self.model.add_module('Batch Normalization 1', nn.BatchNorm2d(hidden_channels1))
    self.model.add_module(activation_name + '1', activation_layer)
    self.model.add_module('Max Pooling 1', nn.MaxPool2d(kernel_size=(kernel_size_pool,1), stride=(stride_pool,1)))

    # # Conv-Pool Block 2 
    self.model.add_module('Spatial Filter2', 
                           nn.Conv2d(hidden_channels1, hidden_channels2, kernel_size=(10,1)))
    self.model.add_module('Batch Normalization 2', nn.BatchNorm2d(hidden_channels2))
    self.model.add_module(activation_name + '2', activation_layer)
    self.model.add_module('Max Pooling 2', nn.MaxPool2d(kernel_size=(kernel_size_pool,1), stride=(stride_pool,1)))


    # # Conv-Pool Block 3 
    self.model.add_module('Spatial Filter3', 
                          nn.Conv2d(hidden_channels2, hidden_channels3, kernel_size=(10,1)))
    self.model.add_module(activation_name +'3', activation_layer)
    self.model.add_module('Batch Normalization 3', nn.BatchNorm2d(hidden_channels3))
    
    self.model.add_module('Max Pooling 3', nn.MaxPool2d(kernel_size=(kernel_size_pool,1), stride=(stride_pool,1)))


    # # Conv-Pool Block 4 
    self.model.add_module('Spatial Filter 4', 
                          nn.Conv2d(hidden_channels3, hidden_channels4, kernel_size=(10,1)))
    self.model.add_module('Batch Normalization 4', nn.BatchNorm2d(hidden_channels4))
    self.model.add_module(activation_name+'4', activation_layer)
    self.model.add_module('Max Pooling 4', nn.MaxPool2d(kernel_size=(kernel_size_pool,1), stride=(stride_pool,1)))


    # Check this 
    # if log_non_linearity:
    #   self.model.add_module('Log', Log())

    self.model.add_module('Flatten',nn.Flatten()) 
    self.model.add_module('Linear',nn.Linear(linear_layer, 4))

  def forward(self, x):
    return self.model(x)
