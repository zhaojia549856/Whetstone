from __future__ import print_function

"""
Convolutional net trained on mnist using the ScheduledSharpener
Uses batch normalization layers during training, which are removed in the final product.
Should achieve 99%+ accuracy.
"""

import numpy as np
import keras
import keras.backend as K
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Reshape, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import Adadelta
from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
from whetstone.utils import copy_remove_batchnorm
from whetstone.callbacks import AdaptiveSharpener
from neuro.model_printer import neuro
import sys

def print_data(x_train, y_train, x_test, y_test):

	f = open("training.txt", "w")
	for i in range(len(x_train)):
		for x in range(len(x_train[i])):
			for y in range(len(x_train[i][x])):
				for z in range(len(x_train[i][x][y])):
					f.write("%f " % (x_train[i][x][y][z]))
		f.write("%d\n" % y_train[i])
	f.close()
	f = open("testing.txt", "w")
	for i in range(len(x_test)):
		for x in range(len(x_test[i])):
			for y in range(len(x_test[i][x])):
				for z in range(len(x_test[i][x][y])):
					f.write("%f " % (x_test[i][x][y][z]))
		f.write("%d\n" % y_test[i])	
	f.close()

def testValueLayer(model, layer_index, data, fout):

	fout.write(model.layers[layer_index].name)
	fout.write("\n")
	intermediate_layer_model = Model(input=model.input, output=model.layers[layer_index].output)
	intermediate_output = intermediate_layer_model.predict(data)
	# print(model.layers[layer_index].get_weights())
	if isinstance(intermediate_output[0][0], (np.ndarray)): 
		for z in range(len(intermediate_output[0][0][0])):
			for y in range(len(intermediate_output[0][0])):
				for x in range(len(intermediate_output[0])):
					fout.write("%d " % intermediate_output[0][x][y][z])
				fout.write("\n")
			fout.write("\n")
	else:
		for i in range(len(intermediate_output[0])):
			fout.write("%d " % intermediate_output[0][i])
		fout.write("\n")
	fout.flush() 

def print_debug_spike(model, x_test, y_test):
	X = model.predict_classes(x_test)
	f_wrong = open("wrong", "w")
	f_spike = open("spike", "w")
	answer = -1

	for i in range(len(x_test)):
	    correct = y_test[i][X[i]];

	    for j in range(len(y_test[i])):
	    	if y_test[i][j] == 1: 
	    		answer = j
	    		break

	    for x in range(len(x_test[i])):
	    	for y in range(len(x_test[i][x])):
	    		for z in range(len(x_test[i][x][y])):
	    			f_spike.write("%f " % x_test[i][x][y][z])
	    f_spike.write("%d %d\n" % (X[i], answer)) 

	    for j in range(len(model.layers)):
	    	if "spiking" in model.layers[j].name or "max" in model.layers[j].name or "flatten" in model.layers[j].name:
	    		testValueLayer(model, j, x_test[i].reshape(1, len(x_test[i]), len(x_test[i][0]), 3), f_spike)
	    f_spike.write("+++++++++++++++++++++++++++++++++++++++++++++\n")
	    f_spike.flush()

	    if correct:
	        correct = True
	    else:
	        correct = False

	        for x in range(len(x_test[i])):
	        	for y in range(len(x_test[i][x])):
	        		for z in range(len(x_test[i][x][y])):
	    				f_wrong.write("%f " % (x_test[i][x][y][z]))
	    	f_wrong.write("%d %d\n" % (X[i], answer))
	    	f_wrong.flush()

numClasses = 100
img_rows, img_cols = 32, 32
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# print_data(x_train, y_train, x_test, y_test)

y_train = keras.utils.to_categorical(y_train, numClasses)
y_test = keras.utils.to_categorical(y_test, numClasses)

key = key_generator(num_classes=numClasses, width=numClasses*4, overlapping=True)

model = Sequential()
model.add(Conv2D(64, (7, 7), padding='same', activation=None, use_bias=True, input_shape=input_shape))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Conv2D(128, (7, 7), padding='same', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (7, 7), padding='same', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Conv2D(128, (7, 7), padding='same', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1600, activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Dense(numClasses*4, activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Softmax_Decode(key))

adaptive = AdaptiveSharpener(verbose=True, min_init_epochs=50,  rate = 0.25, cz_rate = 0.0625, critical = 0.75, patience = 2, sig_increase = 0.1, sig_decrease = 0.1)

max_epochs = 1000000 # Stop training if model isn't fully sharpened after 100 epochs.

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=max_epochs, callbacks=[adaptive])


new_model = copy_remove_batchnorm(model)
new_model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])

# Test both the original and the "copy" and compare their accuracy.
score = model.evaluate(x_test, y_test)[1]
score_new = new_model.evaluate(x_test, y_test)[1]
print('score with batchnorm           =', score)
print('score after removing batchnorm =', score_new)
print('They should be the same.')

neuro(new_model, key, "whetstone")
neuro(new_model, key, "danna2")

# print_debug_spike(new_model, x_test, y_test)
