import tensorflow as tf
import tensorflow.keras as keras

def mnist_dataset(num_classes):
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
    x_train = x_train.reshape(60000, 784) # reshape to feed it directly to Fully connected layer
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:'
        , x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test