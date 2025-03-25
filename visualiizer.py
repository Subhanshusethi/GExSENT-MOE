import time
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
import torch
import torch.nn.functional as F

class PeriodicPlotter:
  def __init__(self, sec, xlabel='', ylabel='', scale=None):

    self.xlabel = xlabel
    self.ylabel = ylabel
    self.sec = sec
    self.scale = scale

    self.tic = time.time()

  def plot(self, data):
    if time.time() - self.tic > self.sec:
      plt.cla()

      if self.scale is None:
        plt.plot(data)
      elif self.scale == 'semilogx':
        plt.semilogx(data)
      elif self.scale == 'semilogy':
        plt.semilogy(data)
      elif self.scale == 'loglog':
        plt.loglog(data)
      else:
        raise ValueError("unrecognized parameter scale {}".format(self.scale))

      plt.xlabel(self.xlabel); plt.ylabel(self.ylabel)
      ipythondisplay.clear_output(wait=True)
      
      ipythondisplay.display(plt.gcf())

      self.tic = time.time()

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"] 
    
def custom_collate_fn(batch):
    # Extract features and pad them
    features = [item['features'] for item in batch]
    max_len = max([f.shape[0] for f in features])
    features = [torch.nn.functional.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in features]
    features = torch.stack(features)  # Stack to form a batch tensor

    # Stack tokenized text inputs (they should have the same shape)
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    # Stack sentiment labels
    sentiment = [item['sentiment'] for item in batch]

    # Construct batch dictionary
    inputs = {
        "features": features,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sentiment": sentiment
    }

    return inputs
