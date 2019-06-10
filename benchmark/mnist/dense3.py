from __future__ import print_function

"""
Light-weight demo of SimpleSharpener, Spiking_BRelu, and Softmax_Decode for a fully connected net on mnist.
"""

import os

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adadelta
from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
from whetstone.callbacks import AdaptiveSharpener, WhetstoneLogger
from neuro.model_printer import neuro

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

def testValueLayer(model, layer_index, data):

	print(model.layers[layer_index].name)
	print(data)
	intermediate_layer_model = Model(input=model.input, output=model.layers[layer_index].output)
	intermediate_output = intermediate_layer_model.predict(data)
	print(model.layers[layer_index].get_weights())
	print(intermediate_output)
	weights = model.layers[layer_index].get_weights()[0]
	for j in range(len(weights[0])):
		sum = 0
		for i in range(len(weights)):
			sum += weights[i][j] * data[0][i]
		print(sum)

numClasses = 10
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = np.reshape(x_train, (60000,28*28))
x_test = np.reshape(x_test, (10000,28*28))

# print_data(x_train, y_train, x_test, y_test)

y_train = to_categorical(y_train, numClasses)
y_test = to_categorical(y_test, numClasses)

key = key_generator(num_classes=10, width=40)

model = Sequential()
model.add(Dense(28*28, input_shape=(28*28,)))
model.add(Spiking_BRelu())
model.add(Dense(1600))
model.add(Spiking_BRelu())
model.add(Dense(40))
model.add(Spiking_BRelu())
model.add(Softmax_Decode(key))

adaptive = AdaptiveSharpener(verbose = True, min_init_epochs = 50, rate = 0.25, cz_rate = 0.0625, critical = 0.75, patience = 2, sig_increase = 0.1, sig_decrease = 0.1)

max_epochs = 1000

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=max_epochs, callbacks=[adaptive])



print(model.evaluate(x_test, y_test))

# testValueLayer(model, 0, x_test[0].reshape(1, 28*28))


neuro(model, key, "danna2")
neuro(model, key, "whetstone")

