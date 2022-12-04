from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np

def lr_lambda(iter):
  return 1 / (1 + 0.002 * iter ** 2)

def apply_mask(output, masks):
  out = torch.zeros_like(output)
  for i, (op, mask) in enumerate(zip(output, masks)):
    for j, m in enumerate(mask):
      out[i][j] = op[j]*m
  return out

def train_one_epoch_ae(epoch, model, train_loader, loss_fn, optimizer, device, writer):
  running_loss = 0.0
  last_loss = 0.0
  # pbar = tqdm(enumerate(train_loader), total=len(train_loader))
  for i, (inputs, mask) in enumerate(train_loader):
    inputs, mask = inputs.to(device), mask.to(device)
    batch, seq_len, _ = inputs.shape
    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Make predictions for this batch
    output = model(inputs.float())
    output = apply_mask(output, mask)

    reverse_idx = torch.arange(seq_len - 1, -1, -1).long()

    # Compute the loss and its gradients
    loss = loss_fn(output, inputs[:, reverse_idx, :])
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    # Gather data and report
    running_loss += loss.item() / batch

  return running_loss / len(train_loader)

def run_validation_ae(model, dataloader, criterion, device):
  # We don't need gradients on to do reporting
  model.eval()

  total_correct = 0
  total_instances = 0
  running_vloss = 0.0

  with torch.no_grad():
    for i, (inputs, mask) in enumerate(dataloader):
      inputs, mask = inputs.to(device), mask.to(device)
      batch, seq_len, _ = inputs.shape
      preds = model(inputs.float())
      preds = apply_mask(preds, mask)
      reverse_idx = torch.arange(seq_len - 1, -1, -1).long()
      loss = criterion(preds, inputs[:, reverse_idx, :])
      running_vloss += loss.item() / batch

  avg_vloss = running_vloss / len(dataloader)
  return avg_vloss

def run_training_loop_ae(experiment_path,
					  model_name, 
                      dataset,
                      trial_name,
                      model, 
                      epochs, 
                      criterion, 
                      optimizer,
					  device,
                      train_dataloader,
                      valid_dataloader,
					  writer,
                      early_stop=20,
                      schedule=True):
  best_val_loss = np.inf
  progress_tracker = 0

  losses = []
  val_losses =[]
  best_model = model.state_dict()
  if schedule:
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20)

  for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()

    avg_loss = train_one_epoch_ae(epoch, model, train_dataloader, criterion, optimizer, device, writer)
    avg_vloss = run_validation_ae(model, valid_dataloader, criterion, device)
    print('train_loss: {} \t val_loss: {} '.format(avg_loss, avg_vloss))
    if schedule:
      lr_scheduler.step(avg_vloss)

    losses.append(avg_loss)
    val_losses.append(avg_vloss)

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'train' : avg_loss, 'valid' : avg_vloss },
                    epoch + 1)
    writer.flush()


    # Track best performance, and save the model's state
    if avg_vloss < best_val_loss:
      best_val_loss = avg_vloss
      model_path = experiment_path + f'/{model_name}/{dataset}_{trial_name}_{best_val_loss}_{epoch}'
      best_model = model.state_dict()
      progress_tracker = 0
    else:
      progress_tracker += 1
    
    if progress_tracker > early_stop:
      print(f'Early stop at epoch: {epoch+1}')
      torch.save(best_model, model_path)
      break
  return losses, val_losses

def run_training_experiment_ae(experiment_path,
							model_name, 
                            dataset_name,
                            trial_name, 
                            model, 
                            epochs, 
                            criterion, 
                            optimizer,
							device,
                            train_loader,
                            valid_loader,
                            early_stop=20
                            ):
  model.to(device)
  writer = SummaryWriter(os.path.join(experiment_path,model_name), comment=dataset_name+'_'+trial_name)
  loss, vloss= run_training_loop_ae(experiment_path,
				  model_name, 
	              dataset_name,
	              trial_name, 
	              model, 
	              epochs, 
	              criterion, 
	              optimizer,
				  device,
	              train_loader,
	              valid_loader,
				  writer,
                  early_stop=early_stop)
  return loss, vloss
