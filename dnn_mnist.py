%%time
#########################################################
###By PSW: Test rapide d'un DNN avec TensorFlow KeRas ###
#########################################################
###1. Load Data and Splot Data
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, Activation
#from tensorflow.keras.utils import utils
from tensorflow.keras.utils import to_categorical
 
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
 
###2.Preprocess
X_train = X_train.reshape(60000, 784)     
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')     
X_test = X_test.astype('float32')     
X_train /= 255    
X_test /= 255
classes = 10
Y_train = to_categorical(Y_train, classes)     
Y_test =  to_categorical(Y_test, classes)
 
###3. Set up parameters
input_size = 784
batch_size = 100    
hidden_neurons = 400    
epochs = 30
 
###4.Build the model
model = Sequential()     
model.add(Dense(hidden_neurons, input_dim=input_size)) 
model.add(Activation('relu'))     
model.add(Dense(classes, input_dim=hidden_neurons)) 
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', 
    metrics=['accuracy'], optimizer='adadelta')
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
 
###5.Test 
score = model.evaluate(X_test, Y_test, verbose=1)
print('\n''Test accuracy:', score[1]) 
#Test accuracy: 0.983
