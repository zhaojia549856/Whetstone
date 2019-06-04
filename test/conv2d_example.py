
from __future__ import print_function

import os
import numpy as np
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Reshape, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import Adadelta
import keras.backend as K
from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
from whetstone.callbacks import AdaptiveSharpener
from whetstone.utils import copy_remove_batchnorm
from neuro.model_printer import neuro


def print_entry(entry):
	s = ""
	for i in range(len(entry)):
		for j in range(len(entry[i])):
			s += str(entry[i][j][0]) + " "
	return s

def classify(entry):
	return int(sum(entry)/len(entry) > 0.5)

def testValueLayer(model, layer_index, data):
	print(model.layers[layer_index].name)
	print(data)
	intermediate_layer_model = Model(input=model.input, output=model.layers[layer_index].output)
	intermediate_output = intermediate_layer_model.predict(data)
	print(model.layers[layer_index].get_weights())
	print(intermediate_output)

numClasses = 2
num_problem, num_testing = 10000, 10
img_rows, img_cols = 4, 4

np.random.seed(0)

y_train, y_test = [], []

x_train = np.random.uniform(size=num_problem*img_cols*img_rows).reshape(num_problem, img_rows*img_cols)
for i in range(len(x_train)):
	y_train.append(classify(x_train[i]))

x_test = np.random.uniform(size=num_testing*img_cols*img_rows).reshape(num_testing, img_rows*img_cols)
for i in range(len(x_test)):
	y_test.append(classify(x_test[i]))


x_train = x_train.reshape(num_problem, img_rows, img_cols, 1)
x_test = x_test.reshape(num_testing, img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


f = open("data.txt", "w+")
for i in range(num_problem):
	f.write("%s %d\n" % (print_entry(x_train[i]), y_train[i]))
for i in range(num_testing):
 	f.write("%s %d\n" % (print_entry(x_test[i]), y_test[i]))
f.close()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = to_categorical(np.array(y_train), numClasses)
y_test = to_categorical(np.array(y_test), numClasses)

key = key_generator(num_classes=2, width=10, overlapping=False)
# key = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]];

model = Sequential()
model.add(Conv2D(3, (3, 3), padding='same', activation=None, use_bias=True, input_shape=input_shape))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Conv2D(2, (3, 3), padding='same', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(3, (3, 3), padding='same', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())

model.add(Flatten())
model.add(Dense(16, activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())

model.add(Dense(10, activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Softmax_Decode(key))


adaptive = AdaptiveSharpener(verbose=True, min_init_epochs=10)

max_epochs = 100

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=max_epochs, callbacks=[adaptive])


new_model = copy_remove_batchnorm(model)


# Test both the original and the "copy" and compare their accuracy.
score = model.evaluate(x_test, y_test)[1]
new_model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
score_new = new_model.evaluate(x_test, y_test)[1]

# code for test layer output!!
testValueLayer(new_model, 1, test[0].reshape(1,4,4,1))


print('score with batchnorm           =', score)
print('score after removing batchnorm =', score_new)
print('They should be the same.')

neuro(new_model, key, "nida")




