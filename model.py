 import numpy as np
from copy import deepcopy
import tensorflow as tf
import tensorflow.keras as keras

def LeNet_300_100(input_shape, num_classes):
        model=keras.Sequential([keras.layers.Dense(300,input_shape=(784,), activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')])
        return model

def train(model,x_train, y_train,batch_size,num_epochs,model_path):
  history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1)
  model.save(model_path)
  return model,history

def retrain(model,opt,x_train, y_train,batch_size,num_epochs,model_path):
  
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
  
  loss_fn= tf.keras.losses.BinaryCrossentropy()

  
  for epoch in range(num_epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for batch_id, (x_train_batch, y_train_batch) in enumerate(train_dataset):

        with tf.GradientTape() as tape:

            logits = model(x_train_batch, training=True) 
            loss_value = loss_fn(y_train_batch, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        updated_grads=[]
        for var, g in zip(model.trainable_variables, grads):
            if 'kernel' in var.name:
                weight_copy = deepcopy(var.numpy())
                mask = tf.constant(np.where(weight_copy == 0, 0, 1),dtype=np.float32)
                updated_grads.append(tf.multiply(g,mask))
            else:
                updated_grads.append(g)
                

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        
        opt.apply_gradients(zip(updated_grads, model.trainable_weights))

        # Log every 200 batches.
        if batch_id % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (batch_id, float(loss_value))
            )
            print("Seen so far: %s samples" % ((batch_id + 1) * batch_size))
  model.save(model_path)
  return model, loss_value

