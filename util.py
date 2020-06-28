import matplotlib.pyplot as plt
import numpy as np

def get_plot(history):
  plt.figure(1, figsize=(15, 3))
  plt.subplot(121)
  plt.plot(history.history['loss'])
  plt.xlabel('Epoch')
  plt.ylabel('Training Loss')
  plt.subplot(122)
  plt.plot(history.history['accuracy'])
  plt.xlabel('Epoch')
  plt.ylabel('Training Accuracy')

def print_nonzeros(model):
    nonzero = total = 0
    for i in range(len(model.variables)):
      if "kernel" in model.variables[i].name:
        name=model.variables[i].name
        tensor=model.variables[i].numpy()
        nz_count=np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'Active: {nonzero}, Pruned : {total - nonzero}, Total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
 
 