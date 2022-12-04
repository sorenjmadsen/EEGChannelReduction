from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

def lr_lambda(iter):
  return 1 / (1 + 0.002 * iter ** 2)

def train_one_epoch(epoch_index, model, train_loader, loss_fn, optimizer, device, writer):
  running_loss = 0.0
  last_loss = 0.0
  #pbar = tqdm(enumerate(train_loader), total=len(train_loader))
  for i, (inputs, labels) in enumerate(train_loader):
    inputs, labels = inputs.to(device), labels.to(device)
    batch_size,_,_,_ = inputs.shape
    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Make predictions for this batch
    outputs = model(inputs.float())

    # Compute the loss and its gradients
    loss = loss_fn(outputs, labels)
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    #pbar.set_description(f'Loss: {torch.round(loss, decimals=4)} \t')

    # Gather data and report
    running_loss += loss.item() / batch_size

  return (running_loss / len(train_loader))

def run_validation(model, dataloader, criterion, device):
  # We don't need gradients on to do reporting
  model.eval()

  total_correct = 0
  total_instances = 0
  running_vloss = 0.0

  with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloader):
      inputs, labels = inputs.to(device), labels.to(device)
      batch_size,_,_,_ = inputs.shape
      outputs = model(inputs.float())
      loss = criterion(outputs, labels)

      preds = torch.argmax(outputs, dim=-1)
      labels = torch.argmax(labels, dim=-1)

      correct_predictions = torch.sum(preds==labels).item()
      total_correct+=correct_predictions
      total_instances+=len(inputs)

      running_vloss += loss / batch_size
  
  val_acc = round(total_correct/total_instances, 5)


  avg_vloss = running_vloss.cpu() / len(dataloader)
  return val_acc, avg_vloss

def run_training_loop(experiment_path,
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
  best_val_acc = 0
  progress_tracker = 0

  losses = []
  val_losses =[]
  val_accs = []
  best_model = model.state_dict()
  if schedule:
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20)

  for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()

    avg_loss = train_one_epoch(epoch, model, train_dataloader, criterion, optimizer, device, writer)
    val_acc, avg_vloss = run_validation(model, valid_dataloader, criterion, device)
    print('train_loss: {} \t val_loss: {} \t val_acc: {}'.format(avg_loss, avg_vloss, val_acc))
    if schedule:
      lr_scheduler.step(avg_vloss)

    losses.append(avg_loss)
    val_losses.append(avg_vloss)
    val_accs.append(val_acc)

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'train' : avg_loss, 'valid' : avg_vloss },
                    epoch + 1)
    writer.add_scalars('Validation Accuracy',
                       {'valid': val_acc})
    writer.flush()


    # Track best performance, and save the model's state
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      model_path = experiment_path + f'/{model_name}/{dataset}_{trial_name}_{best_val_acc}_{epoch}'
      best_model = model.state_dict()
      progress_tracker = 0
    else:
      progress_tracker += 1
    
    if progress_tracker > early_stop:
      print(f'Early stop at epoch: {epoch+1}')
      torch.save(best_model, model_path)
      break
  return losses, val_losses, val_accs


def run_training_experiment(experiment_path,
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
  loss, vloss, vacc = run_training_loop(experiment_path,
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
  return loss, vloss, vacc
