from __future__ import print_function

"""
Convolutional net trained on mnist using the ScheduledSharpener
Uses batch normalization layers during training, which are removed in the final product.
Should achieve 99%+ accuracy.
"""

import numpy as np
import keras
import keras.backend as K
from keras.datasets import mnist
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
		for j in range(len(x_train[i])):
			f.write("%f " % (x_train[i][j]))
		f.write("%d\n" % y_train[i])
	f.close()
	f = open("testing.txt", "w")
	for i in range(len(x_test)):
		for j in range(len(x_test[i])):
			f.write("%f " % (x_test[i][j]))
		f.write("%d\n" % y_test[i])	
	f.close()

def testValueLayer(model, layer_index, data):

	print(model.layers[layer_index].name)
	intermediate_layer_model = Model(input=model.input, output=model.layers[layer_index].output)
	intermediate_output = intermediate_layer_model.predict(data)
	# print(model.layers[layer_index].get_weights())
	if isinstance(intermediate_output[0][0], (np.ndarray)): 
		for z in range(len(intermediate_output[0][0][0])):
			for y in range(len(intermediate_output[0][0])):
				for x in range(len(intermediate_output[0])):
					sys.stdout.write("%d " % intermediate_output[0][x][y][z])
				sys.stdout.write("\n")
			sys.stdout.write("\n")
	else:
		for i in range(len(intermediate_output[0])):
			sys.stdout.write("%d " % intermediate_output[0][i])
		sys.stdout.write("\n")
	sys.stdout.flush() 

def print_debug_spike(model, x_test, y_test):

	X = model.predict_classes(x_test)

	f = open("touch", "w")

	for i in range(len(x_test)):
	    correct = y_test[i][X[i]];
	    if correct:
	        correct = True
	    else:
	        correct = False
	        for x in range(len(x_test[i])):
	        	for y in range(len(x_test[i][x])):
	    			f.write("%f " % (x_test[i][x][y]))
	    	f.write("\n")

	    for x in range(len(x_test[i])):
	    	for y in range(len(x_test[i][x])):
	    		sys.stdout.write("%f " % x_test[i][x][y])
	    sys.stdout.write("%d %d\n" % (correct, X[i]))
	    sys.stdout.flush()
	    testValueLayer(model, 1, x_test[i].reshape(1,28,28,1))
	    testValueLayer(model, 3, x_test[i].reshape(1,28,28,1))
	    testValueLayer(model, 4, x_test[i].reshape(1,28,28,1))
	    testValueLayer(model, 6, x_test[i].reshape(1,28,28,1))
	    # testValueLayer(model, 7, x_test[i].reshape(1,28,28,1))
	    # testValueLayer(model, 9, x_test[i].reshape(1,28,28,1))
	    # testValueLayer(model, 11, x_test[i].reshape(1,28,28,1))
	    # testValueLayer(model, 13, x_test[i].reshape(1,28,28,1))
	    # testValueLayer(model, 14, x_test[i].reshape(1,28,28,1))
	    # testValueLayer(model, 16, x_test[i].reshape(1,28,28,1))
	    # testValueLayer(model, 18, x_test[i].reshape(1,28,28,1))
	    # print("====================================================")

numClasses = 10
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, numClasses)
y_test = keras.utils.to_categorical(y_test, numClasses)

key = key_generator(num_classes=10, width=40, overlapping=False)

model = Sequential()
model.add(Conv2D(16, (7, 7), padding='same', activation=None, use_bias=True, input_shape=input_shape))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Conv2D(32, (7, 7), padding='same', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), padding='same', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Conv2D(32, (5, 5), padding='same', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Conv2D(32, (3, 3), padding='same', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Flatten())
model.add(Dense(400, activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Dense(40, activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Softmax_Decode(key))

adaptive = AdaptiveSharpener(verbose=True, min_init_epochs=10)

max_epochs = 200 # Stop training if model isn't fully sharpened after 100 epochs.

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


# neuro(new_model, key, "whetstone")
# neuro(new_model, key, "danna2")


