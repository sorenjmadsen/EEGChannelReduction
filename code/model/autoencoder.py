import torch
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self, input_size=128, hidden_size=64, num_layers=2):
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                        dropout=0.2, bidirectional=False)
    self.activation=nn.Tanh()
  def forward(self, x):
    outputs, (hidden, cell) = self.lstm(x)
    return self.activation(outputs), (self.activation(hidden), self.activation(cell))
    
class Decoder(nn.Module):
  def __init__(self, input_size=128, hidden_size=64, output_size=128, num_layers=2):
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_layers = num_layers
    
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                        dropout=0.2, bidirectional=False)
    self.activation=nn.Tanh()
    self.fc = nn.Linear(hidden_size, output_size)
    
        
  def forward(self, x, hidden):
    output, (hidden, cell) = self.lstm(x, hidden)  
    prediction = self.fc(output)   
    return prediction, (hidden, cell)

class LSTMAutoEncoder(nn.Module):
  def __init__(self, input_size=128, hidden_size=64, output_size=128, num_layers=2):
    super(LSTMAutoEncoder, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_layers = num_layers

    self.encoder = Encoder(input_size, hidden_size, num_layers)
    self.decoder = Decoder(input_size, hidden_size, output_size, num_layers)

  def set_device(self, device):
    self.device = device

  def forward(self, x):
    batch_size, seq_len, _ = x.shape
    _, enc_hidden = self.encoder(x)
    temp_input = torch.zeros((batch_size,1,self.input_size), dtype=torch.float).to(self.device)
    hidden = enc_hidden

    outputs = []
    for t in range(seq_len):
      temp_input, hidden = self.decoder(temp_input, hidden)
      outputs.append(temp_input)
    out = torch.cat(outputs, dim=1)
    return out
