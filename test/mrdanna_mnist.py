from __future__ import print_function

"""
Light-weight demo of SimpleSharpener, Spiking_BRelu, and Softmax_Decode for a fully connected net on mnist.
"""

import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adadelta
from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
from whetstone.callbacks import SimpleSharpener, WhetstoneLogger
from neuro.model_printer import neuro
import timeit

def print_data(x_train, y_train, x_test, y_test):

	f = open("training.txt", "w")
	for i in range(len(x_train)):
		for j in range(len(x_train[i])):
			f.write("%f " % (x_train[i][j]))
		f.write("%d\n" % y_train[i])
	f.close()
	f = open("testing.txt", "w")
	for i in range(len(x_test)):
		for j in range(len(x_test[i])):
			f.write("%f " % (x_test[i][j]))
		f.write("%d\n" % y_test[i])	

def benchmark(time_delta, x_train, y_train, x_test, y_test):

	num_neurons = 28*28 + 256 + 64 + 100
	num_synapses = 28*28 * 256 + 256 * 64 + 64 * 100

	train_accuracy = model.evaluate(x_train, y_train)
	test_accuracy = model.evaluate(x_test, y_test)

	print(train_accuracy[1], test_accuracy[1], time_delta, num_neurons, num_synapses)

numClasses = 10
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = np.reshape(x_train, (60000,28*28))
x_test = np.reshape(x_test, (10000,28*28))

y_train = to_categorical(y_train, numClasses)
y_test = to_categorical(y_test, numClasses)

key = key_generator(num_classes=10, width=100, overlapping=False)

model = Sequential()
model.add(Dense(64, input_shape=(28*28,)))
model.add(Spiking_BRelu())
model.add(Dense(64))
model.add(Spiking_BRelu())
model.add(Dense(100))
model.add(Spiking_BRelu())
model.add(Softmax_Decode(key))

simple = SimpleSharpener(start_epoch=5, steps=5, epochs=True, bottom_up=True)

# Create a new directory to save the logs in.
# log_dir = './simple_logs'
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)

# logger = WhetstoneLogger(logdir=log_dir, sharpener=simple)

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])

time_delta = timeit.default_timer()
model.fit(x_train, y_train, batch_size=128, epochs=21, callbacks=[simple])
time_delta = timeit.default_timer() - time_delta

neuro(model, key, "mrdanna")

print(model.evaluate(x_test, y_test))
