from __future__ import print_function

"""
Light-weight demo of SimpleSharpener, Spiking_BRelu, and Softmax_Decode for a fully connected net on mnist.
"""

import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.datasets import cifar10 
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adadelta
from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
from whetstone.callbacks import SimpleSharpener, WhetstoneLogger
from neuro.model_printer import neuro

numClasses = 10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, numClasses)
y_test = to_categorical(y_test, numClasses)

x_train = np.reshape(x_train, (50000,32*32*3))
x_test = np.reshape(x_test, (10000,32*32*3))

key = key_generator(num_classes=10, width=100)

model = Sequential()
model.add(Dense(32*32, input_shape=(32*32*3,)))
model.add(Spiking_BRelu())
model.add(Dense(32*32))
model.add(Spiking_BRelu())
model.add(Dense(16*16))
model.add(Spiking_BRelu())
model.add(Dense(16*16))
model.add(Spiking_BRelu())
model.add(Dense(8*8))
model.add(Spiking_BRelu())
model.add(Dense(8*8))
model.add(Spiking_BRelu())
model.add(Dense(100))
model.add(Spiking_BRelu())
model.add(Softmax_Decode(key))

simple = SimpleSharpener(start_epoch=40, steps=5, epochs=True, bottom_up=True)

# Create a new directory to save the logs in.
log_dir = './simple_logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = WhetstoneLogger(logdir=log_dir, sharpener=simple)

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=80, callbacks=[simple, logger])
# neuro(model, key)

# print(x_test, y_test)
# with open("output", "w") as f:
#     f.write(x_test, y_test)

print(model.evaluate(x_test, y_test))
