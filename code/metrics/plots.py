import matplotlib.pyplot as plt 

def plot_loss(epochs, loss, vloss, model_name, dataset_name, trial_name, plots_path):
  plt.plot(epochs, loss, label='train')
  plt.plot(epochs, vloss, label='valid')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title(f'Training vs. Validation Loss - {model_name}:{dataset_name}_{trial_name}')
  plt.legend()
  plt.savefig(plots_path + f'{model_name}_{dataset_name}_loss.jpg', bbox_inches='tight', dpi=150)
  plt.show(block=False)
  plt.close('all')
