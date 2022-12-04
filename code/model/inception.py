import torch
import torch.nn as nn

class EEGInception(nn.Module):
  def __init__(self, input_channel=30, number_classes=4,linear_layer=324288):
    super(EEGInception, self).__init__()

    self.inception1 = Initial_IncepBlk(input_channel=input_channel,b_neck=48)
    self.inception2 =  Intermediate_IncepBlk(input_channel=288,b_neck=48)
    
    self.inception3 =  Intermediate_IncepBlk(input_channel=288,b_neck=48)
    self.inception4 =  Intermediate_IncepBlk(input_channel=288,b_neck=48)
    self.inception5 =  Intermediate_IncepBlk(input_channel=288,b_neck=48)
    self.inception6 =  Intermediate_IncepBlk(input_channel=288,b_neck=48)

    self.residual1  =  Residual_Mod(in_ch=input_channel, out_ch=288)
    self.residual2  =  Residual_Mod(in_ch=288, out_ch=288)

    self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(linear_layer, number_classes)


  def forward(self, x):
    # Residual module 1 
    res1 = self.residual1(x)
    
    # First 3 inception blocks 
    x = self.inception1(x)
    x = self.inception2(x)
    x = self.inception3(x)


    # Adding residual module 1 & inception3 
    res_sum1 = x+res1
    

    # Residual module 2 
    res2 = self.residual2(res_sum1)

    x = self.inception4(res_sum1)
    x = self.inception5(x)
    x = self.inception6(x)

    # Adding residual module 2 & inception6 
    res_sum2 = x+res2

    x = self.avgpool(res_sum2)
    x = self.flatten(x)
    x = self.fc1(x)

    return x

class Residual_Mod(nn.Module):
  def __init__(self, in_ch, out_ch, **kwargs):
    super(Residual_Mod, self).__init__()
    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1,1), **kwargs)
    self.batchnorm = nn.BatchNorm2d(out_ch)

  def forward(self, x):
    return self.relu(self.batchnorm(self.conv(x)))


####################################################################################################################################################################################
class Initial_IncepBlk(nn.Module):
# EEG Inception: Initial Inception block for a 4-class dataset 
# Difference between Initial & intermediate is the structure of the bottle neck layer
# Bottlneck layer for INITIAL Inception Block INCREASES input data dimensions 
  def __init__(self, input_channel=30,b_neck=48):
    super(Initial_IncepBlk, self).__init__()                
    self.bottleneck = nn.Conv2d(input_channel, b_neck, kernel_size=(1,1),padding='same')
    self.branch1 = nn.Conv2d(b_neck, b_neck, kernel_size=(1,25),padding='same')
    self.branch2 = nn.Conv2d(b_neck, b_neck, kernel_size=(1,75),padding='same')
    self.branch3 = nn.Conv2d(b_neck, b_neck, kernel_size=(1,125),padding='same')
    self.branch4 = nn.Conv2d(b_neck, b_neck, kernel_size=(1,175),padding='same')
    self.branch5 = nn.Conv2d(b_neck, b_neck, kernel_size=(1,225),padding='same')
    # Pooling layer
    self.poollayer = nn.Sequential(
        nn.MaxPool2d(kernel_size=(1,25), stride=1, padding=(0,12)),
        nn.Conv2d(input_channel, b_neck, kernel_size=(1,1), padding='same')
    )
    # BatchNorm & activation function 
    self.bnconcat = nn.BatchNorm2d(b_neck*6) #CHECK ME 
    self.relu = nn.ReLU()

    # Concatenating all the filters 
  def forward(self, x):
    bneck = self.bottleneck(x)
    x1 = self.branch1(bneck)
    x2 = self.branch2(bneck)
    x3 = self.branch3(bneck)
    x4 = self.branch4(bneck)
    x5 = self.branch5(bneck)
    xpool = self.poollayer(x)
    x = torch.cat([x1, x2, x3, x4, x5, xpool],dim=1)
    x = self.bnconcat(x)
    x = self.relu(x)
    return x

####################################################################################################################################################################################
class Intermediate_IncepBlk(nn.Module):

# EEG Inception: Intermidieate Inception block for a 4-class dataset 
# Difference between Initial & intermediate is the structure of the bottle neck layer
# Bottlneck layer for INTERMEDIATE Inception Block DECREASES input data dimensions 

  def __init__(self, input_channel=288, b_neck=48):
      
    super(Intermediate_IncepBlk, self).__init__()

    # intermediate_conv=input_channel//b_neck
    # output_conv=input_channel//b_neck
    # output_pool=input_channel//b_neck


    self.bottleneck = nn.Conv2d(input_channel, b_neck, kernel_size=(1,1),padding='same')
    
    self.branch1 = nn.Conv2d(b_neck, b_neck, kernel_size=(1,25),padding='same')

    self.branch2 = nn.Conv2d(b_neck, b_neck, kernel_size=(1,75),padding='same')
    
    self.branch3 = nn.Conv2d(b_neck, b_neck, kernel_size=(1,125),padding='same')

    self.branch4 = nn.Conv2d(b_neck, b_neck, kernel_size=(1,175),padding='same')

    self.branch5 = nn.Conv2d(b_neck, b_neck, kernel_size=(1,225),padding='same')

    # Pooling layer
    self.poollayer = nn.Sequential(
        nn.MaxPool2d(kernel_size=(1,25), stride=1, padding=(0,12)),
        nn.Conv2d(input_channel, b_neck, kernel_size=(1,1), padding='same')
    )

    self.bnconcat = nn.BatchNorm2d(b_neck*6)
    self.relu = nn.ReLU()
    
    # Concatenating all the filters 
  def forward(self, x):
    bneck = self.bottleneck(x)
    x1 = self.branch1(bneck)
    x2 = self.branch2(bneck)
    x3 = self.branch3(bneck)
    x4 = self.branch4(bneck)
    x5 = self.branch5(bneck)
    xpool = self.poollayer(x)
    x = torch.cat([x1, x2, x3, x4, x5, xpool],dim=1)
    x = self.bnconcat(x)
    x = self.relu(x)
    return x




