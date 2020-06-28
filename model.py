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

