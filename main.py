import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers

from copy import deepcopy
from dataset import mnist_dataset
from model import LeNet_300_100, train, retrain
from util import get_plot, print_nonzeros
from prune_model import prune_weights

batch_size = 128
lr=0.001
num_epochs = 20
img_rows, img_cols = 28, 28
input_size= 784
num_classes=10

# load preprocessed MNIST dataset
x_train, y_train, x_test, y_test=mnist_dataset(num_classes) 

# build model
model = LeNet_300_100(input_size, num_classes) 
opt = keras.optimizers.Adam(learning_rate=lr)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

#path to save original model
model_path="lenet300-100.h5"               
model,history=train(model,x_train, y_train,batch_size,num_epochs,model_path) #training 
original_loss,original_accuracy = model.evaluate(x_test, y_test, verbose=0)  #testing
print("Result of Original Model:")
print('Original Model loss:', original_loss)
print('Original Model accuracy:', original_accuracy)
print_nonzeros(model)
get_plot(history)

#Iterative training (prune+retrain) in single iteration and we run until we get pruning accuracy >= original model accuracy.
#Threshold is calculated based on standard deviation of weights and sensitivity/quality factor.

for i in range(1,10):
  pretrained_model= keras.models.load_model(model_path)  
  quality_parameter = 1.6      #sensitivity factor to calculate threshold

  #Pruning
  for layerid in range(len(pretrained_model.layers)):
    layer=pretrained_model.layers[layerid]
    weight=layer.get_weights()
    if len(weight) > 0:
            temp_weight=deepcopy(weight)
            updated_weights = prune_weights(temp_weight[0],quality_parameter) #function call to prune weight based on threshold
            temp_weight[0]= updated_weights
            layer.set_weights(temp_weight)   #set layers weights with pruned weight
  #save pruned model
  pretrained_model.save("pruned_lenet_300-100.h5") 
  pruned_loss,pruned_accuracy = pretrained_model.evaluate(x_test, y_test, verbose=0) # test pruned model

  # iterative retraining exit condition
  if pruned_accuracy >=original_accuracy:   
    print(f'Final Result:')
    print_nonzeros(pretrained_model)
    print('Final Loss:', pruned_loss)
    print('Final Accuracy:',pruned_accuracy)
    pretrained_model.save("final_model_pruned.h5")
    exit(0)
  
  print(f'Result of {i} Iteration:')
  print_nonzeros(pretrained_model)   #function to measure pruned weights and compression rate
  print('Loss after Pruning:', pruned_loss)
  print('Accuracy after Pruning:',pruned_accuracy)
   
  pruned_model=  keras.models.load_model("pruned_lenet_300-100.h5",compile=False) #load pruned model
  opt = keras.optimizers.Adam(learning_rate=lr/10)  # update optimizer with lr=lr/10
  pruned_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  retrain_model_path="retrain_lenet.h5"
  
  #retrain model with 10 epochs to learn important weights those are removed  during pruning
  retrain_model,loss_value = retrain(pruned_model,opt,x_train, y_train,batch_size,10,retrain_model_path) 
  retrain_loss,retrain_accuracy  = retrain_model.evaluate(x_test, y_test, verbose=0)

  print_nonzeros(retrain_model) 
  print('Loss after Retraining:', retrain_loss)
  print('Accuracy after Reraining:',retrain_accuracy)
  model_path=retrain_model_path #update model path with latest model to continue pruning
  

