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
from keras.models import Sequential, model_from_json, load_model
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Reshape, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import Adadelta
from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
from whetstone.utils import copy_remove_batchnorm
from whetstone.callbacks import AdaptiveSharpener
from neuro.model_printer import neuro
import os

numClasses = 10
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# for i in range(len(x_train)):
# 	x_train[i] = x_train[i][:4]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')[:100]
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, numClasses)[:100]
y_test = keras.utils.to_categorical(y_test, numClasses)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
exit(0)

key = key_generator(num_classes=10, width=40, overlapping=False)

model = Sequential()
model.add(Conv2D(6, (7, 7), padding='same', activation=None, use_bias=True, input_shape=input_shape))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Conv2D(8, (7, 7), padding='same', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (5, 5), padding='same', activation=None, use_bias=True))
# model.add(BatchNormalization())
# model.add(Spiking_BRelu())
# model.add(Conv2D(64, (5, 5), padding='same', activation=None, use_bias=True))
# model.add(BatchNormalization())
# model.add(Spiking_BRelu())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
# model.add(Conv2D(128, (3, 3), padding='same', activation=None, use_bias=True))
# model.add(BatchNormalization())
# model.add(Spiking_BRelu())
model.add(Flatten())
model.add(Dense(64, activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Dense(40, activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Spiking_BRelu())
model.add(Softmax_Decode(key))

adaptive = AdaptiveSharpener(verbose=True, min_init_epochs=10)

max_epochs = 10 # Stop training if model isn't fully sharpened after 100 epochs.

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=max_epochs, callbacks=[adaptive])



new_model = copy_remove_batchnorm(model)
neuro(new_model, key, "danna")



# Test both the original and the "copy" and compare their accuracy.
# score = model.evaluate(x_test, y_test)[1]
# score_new = model.evaluate(x_test, y_test)[1]


# print('score with batchnorm           =', score)
# print('score after removing batchnorm =', score_new)
# print('They should be the same.')

