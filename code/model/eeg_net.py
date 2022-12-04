import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, bias=False):
    super(SeparableConv2d, self).__init__()
    self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                               groups=in_channels, bias=bias, padding='same')
    self.pointwise = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1, bias=bias)
  def forward(self, x):
    out = self.depthwise(x)
    out = self.pointwise(out)
    return out

class EEGNet(nn.Module):
    def __init__(self, input_channels=30, filters1=125,filters2=25, num_classes=4, linear_layer=1125, dropout=0.5, depth=2):
        super(EEGNet,self).__init__()

        self.temporal=nn.Sequential(
            nn.Conv2d(1,filters1,kernel_size=[1,64],stride=1, bias=False,
                padding='same'), 
            nn.BatchNorm2d(filters1),
        )

        self.depthwise=nn.Sequential(
            nn.Conv2d(filters1,filters1*depth,kernel_size=[input_channels,1],bias=False,
                groups=filters1),
            nn.BatchNorm2d(filters1*depth),
            nn.ELU(inplace=True),
        )

        self.seperable=nn.Sequential(
            SeparableConv2d(depth*filters1, filters2, kernel_size=[1,16]),
            nn.BatchNorm2d(filters2),
            nn.ELU(True),
        )

        self.avgpool = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)   
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

        
        self.fc2 = nn.Linear(linear_layer, num_classes)

    def forward(self,x):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.seperable(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc2(x)
        return x
