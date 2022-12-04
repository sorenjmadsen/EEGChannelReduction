import torch
from tqdm import tqdm

def test_accuracy(model, dataloader, device):
  model.eval()
  total_correct = 0
  total_instances = 0

  with torch.no_grad():
    for inputs, labels in tqdm(dataloader):
      inputs, labels = inputs.to(device), labels.to(device)
      preds = torch.argmax(model(inputs.float()), dim=-1)
      labels = torch.argmax(labels, dim=-1)
      correct_predictions = torch.sum(preds==labels).item()
      total_correct+=correct_predictions
      total_instances+=len(inputs)

  return (total_correct/total_instances)
