
from __future__ import print_function

import os
import numpy as np
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adadelta
import keras.backend as K
from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator, Input_Spiking
from whetstone.callbacks import SimpleSharpener, WhetstoneLogger
from neuro.model_printer import neuro

def classification(k, b, x, y):
    if k * x + b > y:
        return 1
    else:
        return 0

numClasses = 2
num_problem = 10000
num_testing = 1000
k = 3.0
b = -1.0
x_train = []
y_train = []
x_test = []
y_test = []
np.random.seed(0)
for i in range(num_problem):
    x_train.append([np.random.uniform(), np.random.uniform()])
    y_train.append(classification(k, b, x_train[-1][0], x_train[-1][1]))
for i in range(1000):
    x_test.append((np.random.uniform(), np.random.uniform()))
    y_test.append(classification(k, b, x_test[-1][0], x_test[-1][1]))

f = open("data.txt", "w+")
# for i in range(num_problem):
# 	f.write("%f %f %d\n" % (x_train[i][0], x_train[i][1], y_train[i]))
for i in range(1000):
 	f.write("%f %f %d\n" % (x_test[i][0], x_test[i][1], y_test[i]))

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = to_categorical(y_train, numClasses)
y_test = to_categorical(y_test, numClasses)
key = key_generator(num_classes=2, width=10, overlapping=False)
key = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]];

model = Sequential()
model.add(Dense(10, input_shape=(2,)))
model.add(Spiking_BRelu())
model.add(Dense(30))
model.add(Spiking_BRelu())
model.add(Dense(10))
model.add(Spiking_BRelu())
model.add(Softmax_Decode(key))

simple = SimpleSharpener(start_epoch=5, steps=5, epochs=True, bottom_up=True)

log_dir = './simple_logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = WhetstoneLogger(logdir=log_dir, sharpener=simple)

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=100, callbacks=[simple, logger])
# print(model.get_weights())
# print(model.get_config())
# print(model.summary())
# neuro(model, key, "danna2")
print(model.evaluate(x_test, y_test))

# converter = tf.lite.TocoConverter.from_keras_model_file(model)
# converter.post_training_quantize = True
# tflite_quantized_model = converter.convert()
# open("quantized_model.tflite", "wb").write(tflite_quantized_model)
model.save('my_model.h5')

# w = model.get_weights()[0]
# b = model.get_weights()[1]

# X = model.predict_classes(x_test)

# with open("output", "w") as f:
#     for i in range(1000):
#         correct = y_test[i][X[i]];
#         if correct:
#             correct = True
#         else:
#             correct = False  
#         f.write("X= %lf %lf predicted: %d %s\n" % (float(x_test[i][0]), float(x_test[i][1]), X[i], correct))

