from __future__ import print_function

import numpy as np
import keras
import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Layer
from keras.optimizers import Adadelta
from os import sys
import math 

class neuro():
    def __init__(self, model, key, format, path=None, filename=None):
        self.model = model
        self.key = key
        if path == None:
            self.file_path = "./"
            self.supply = "./supply.txt"
        else:
            self.file_path = path + "../../"
            self.supply = path + "./supply.txt"
        self.loading()

        if format == "danna2":
            danna2(self)
        elif format == "whetstone":
            whetstone(self)
        elif format == "nida":
            print("Using Nida model (useless model)")
            nida(self)
        else: 
            print("Invalid model")

    def load_config(self):
        self.neurons = []; self.synapses = []
        self.input_size = list(self.model.layers[0].get_config()["batch_input_shape"][1:])
        self.z_start = 0
        self.neuron_id = 0 
        self.synapse_id = 0

    def loading(self):

        self.load_config()

        model = self.model
        self.construct_supply_file(model.layers[0])

        for layer in model.layers[1:]:
            if type(layer) == keras.layers.Conv2D:
                print("keras.layers.Conv2D")

                self.whetstone_v2(layer)

            elif type(layer) == keras.layers.MaxPooling2D:
                print("keras.layer.MaxPooling2D")
                (x, y) = layer.get_config()["pool_size"] 
                self.last_layer_size = [self.last_layer_size[0]/x, self.last_layer_size[1]/y, self.last_layer_size[2]]
                
                #loading neuron + add z_start
                self.load_maxpooling_neurons()

                #update to current size
                self.load_maxpooling_synapses(x, y)

            elif type(layer) == keras.layers.Flatten:
                print("keras.layer.Flatten")
                self.neurons[-1] = self.flatten(self.neurons[-1])
                self.last_layer_size = [self.neurons[-1], None, None]

            elif type(layer) == keras.layers.Dense:
                print("keras.layers.Dense")
                
                weights, thresholds = layer.get_weights()

                self.load_dense_neurons(0.5 - thresholds)
                self.load_dense_synapses(weights)

            elif type(layer) == keras.layers.normalization.BatchNormalization:
                print("ERROR: Can't handle BatchNormalization layer")

        # flat neuron layer for printing 
        self.xy_size = 0
        for i in range(len(self.neurons)): 
            self.neurons[i] = self.flatten(self.neurons[i])
            if len(self.neurons[i]) > self.xy_size:
                self.xy_size = len(self.neurons[i])

    def construct_supply_file(self, layer):
        if type(layer) == keras.layers.Dense:
            print("first a dense layer")

            weights, thresholds = layer.get_weights()
            self.load_dense_neurons(0.5 - thresholds)
            f = open(self.supply, "w")
            f.write("Dense\n")
            f.write("KEY: \n")
            for i in range(len(self.key)):
                f.write(" ".join(str(self.key[i][j]) for j in range(len(self.key[i]))))
                f.write("\n")

            f.write("L1: \n")
            for i in range(len(weights[0])):
                entry = str(i) + " "
                for j in range(len(weights)):
                    entry += "%d:%lf " % (j, weights[j][i])
                f.write(entry + "\n")
            f.close()

        elif type(layer) == keras.layers.Conv2D:
            print("first a conv2d layer")
            
            config = layer.get_config()
            weights, thresholds = layer.get_weights()
            filters = config["filters"]
            kernel_size = config["kernel_size"]
            self.last_layer_size = [self.input_size[0], self.input_size[1], filters]

            self.load_conv2d_neurons(0.5 - thresholds)
            
            #write first layer to supply file
            f = open(self.supply, "w")
            f.write("Conv2D\n")

            f.write("KEY: \n")
            for i in range(len(self.key)):
                f.write(" ".join(str(self.key[i][j]) for j in range(len(self.key[i]))))
                f.write("\n")
                
            f.write("L1: \n")  

            for z2 in range(self.last_layer_size[2]):
                for y2 in range(self.last_layer_size[1]):
                    for x2 in range(self.last_layer_size[0]):
                        entry = str(z2*self.last_layer_size[1]*self.last_layer_size[0] + y2*self.last_layer_size[0] + x2) + " "
                        for x1 in range(kernel_size[1]):
                            for y1 in range(kernel_size[0]):
                                x = x2 + x1 - kernel_size[0]/2
                                y = y2 + y1 - kernel_size[1]/2
                                if x >= 0 and x < self.input_size[0] and y >= 0 and y < self.input_size[1]:
                                    entry += "%d:%lf " % (x*self.input_size[0]+y, weights[x1][y1][0][z2])
                        f.write(entry + "\n")
            f.close()   
        else:
            print("load_first_layer error: Never seen first layer type %s" % str(type(layer)))
            exit(0)          

    def flatten(self, layer):
        if len(np.array(layer).shape) == 1: 
            return layer
        layer = np.array(layer)
        shape = layer.shape
        size = 1
        for i in range(len(shape)):
            size *= shape[i]
        layer = layer.reshape(size)
        layer = layer.tolist()
        return layer

    def num_neurons(self):
        count = 0
        for layer in self.neurons:
            if len(np.array(layer).shape) != 1:
                count += len(flatten(layer))
            else: 
                count += len(layer)
        return count

    def num_synapses(self):
        count = 0
        for layer in self.synapses:
            count += len(layer)
        return count

    def load_conv2d_neurons(self, thresholds):
        layer = np.empty([self.last_layer_size[0], self.last_layer_size[1], len(thresholds)], dtype=type(neuron))
        for x in range(self.last_layer_size[0]):
            for y in range(self.last_layer_size[1]):
                for z in range(len(thresholds)):
                    id = z*self.last_layer_size[1]*self.last_layer_size[0] + y*self.last_layer_size[0]+ x
                    layer[x][y][z] = neuron(x, y, thresholds[z], id + self.neuron_id, self.z_start + z)
        self.neurons.append(layer.tolist())
        self.z_start += len(thresholds)
        self.neuron_id += id + 1

    def load_conv2d_synapses(self, filters, kernel_size, weights):
        if kernel_size[0] % 2 == 0:
            print("Can't deal with even kernal size")
            exit(1)

        layer = [] 
        id = self.synapse_id

        for x2 in range(self.last_layer_size[0]):
            for y2 in range(self.last_layer_size[1]):
                for z2 in range(filters):
                    for x1 in range(kernel_size[0]):
                        for y1 in range(kernel_size[1]):
                            for z1 in range(self.last_layer_size[2]):
                                x = x2 + x1 - kernel_size[0]/2
                                y = y2 + y1 - kernel_size[1]/2
                                if x >= 0 and x < self.last_layer_size[0] and y >= 0 and y < self.last_layer_size[1]:
                                    try: 
                                        layer.append(synapse(self.neurons[-2][x][y][z1], 
                                            self.neurons[-1][x2][y2][z2], weights[x1][y1][z1][z2], id))
                                        self.neurons[-1][x2][y2][z2].append_pre_synapse(layer[-1])
                                        id += 1
                                    except IndexError as e:
                                        print(e)
                                        print("Index in the neurons layer: (", x, y, z1, ",", x2, y2, z2, ")")
                                        print("Index in the network layer: (", x, y, self.z_start + z1 - self.last_layer_size[2], 
                                            ",", x2, y2, self.z_start + z2, ")")
                                        exit(1)
        self.synapse_id = id
        self.synapses.append(layer)
        self.last_layer_size[2] = filters

    def load_maxpooling_neurons(self):
        layer = []
        for x in range(self.last_layer_size[0]):
            layer.append([])
            for y in range(self.last_layer_size[1]):
                layer[x].append([])
                for z in range(self.last_layer_size[2]):
                    id = z*self.last_layer_size[1]*self.last_layer_size[0] + y*self.last_layer_size[0]+ x
                    layer[x][y].append(neuron(x, y, 0.5, id + self.neuron_id, self.z_start + z))
        self.neurons.append(layer)
        self.z_start += self.last_layer_size[2]
        self.neuron_id += id + 1

    def load_maxpooling_synapses(self, kernal_x, kernal_y):
        layer = [] 
        id = self.synapse_id
        for z in range(self.last_layer_size[2]):
            for y in range(self.last_layer_size[1]*kernal_y):
                for x in range(self.last_layer_size[0]*kernal_x):
                    try:
                        layer.append(synapse(self.neurons[-2][x][y][z], self.neurons[-1][x/kernal_x][y/kernal_y][z], 1, id))
                        self.neurons[-1][x/kernal_x][y/kernal_y][z].append_pre_synapse(layer[-1])
                        id += 1
                    except IndexError as e:
                        print(e)
        self.synapses.append(layer)
        self.synapse_id = id

    def load_dense_neurons(self, thresholds):
        layer = []
        for i in range(len(thresholds)):
            layer.append(neuron(0, i, thresholds[i], i + self.neuron_id, self.z_start))
        self.neurons.append(layer)
        self.z_start += 1
        self.neuron_id += i + 1

    def load_dense_synapses(self, weights):
        layer = []
        id = self.synapse_id
            
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                layer.append(synapse(self.neurons[-2][i], self.neurons[-1][j], weights[i][j], id))
                self.neurons[-1][j].append_pre_synapse(layer[-1])
                id += 1
        self.synapses.append(layer)
        self.synapse_id = id

    def whetstone_v1(self, layer):
        #setup neurons
        weights, thresholds = layer.get_weights()
        config = layer.get_config()
        filters = config["filters"]
        kernel_size = config["kernel_size"]

        #increase z axis starting address  
        self.load_conv2d_neurons(0.5 - thresholds)
        
        #update to current size 
        self.load_conv2d_synapses(filters, kernel_size, weights)

    def whetstone_v2(self, layer):
        weights, thresholds = layer.get_weights()
        config = layer.get_config()
        filters = config["filters"]
        kernel_size = list(config["kernel_size"])
        kernel_size.append(filters)

        self.Init()

        self.PA(self.last_layer_size, kernel_size, 2, 2)

    def PA(self, n, w, i, d):
        last_layer_neurons = self.neurons[-1]
        half_size = int(w[0])/2

        #middle neurons 
        neurons = np.empty(w, dtype=type(neuron)) #need change w
        id = self.neuron_id
        for z in range(w[2]):
            for y in range(w[1]):
                for x in range(w[0]):
                    neurons[x][y][z] = neuron(x, y, 0.5, id, self.z_start+z)
                    id += 1
        self.neuron_id = id


        synapses = [] 
        id = self.synapse_id
        #input synapses
        for z in range(n[2]):
            j = 0
            for y in range(n[1]-half_size):
                for x in range(n[0]-half_size):
                    if x == 0 and y == 0: 
                        for r in range(half_size+1):
                            for c in range(half_size+1):
                                synapses.append(synapse(last_layer_neurons[c][r][z], neurons[half_size+c][half_size+r][z], 1, id, delay=d))
                                id += 1
                    elif x == 0: 
                        j += half_size
                        for c in range(half_size+1): 
                            synapses.append(synapse(last_layer_neurons[x+c][y+half_size][z], neurons[half_size+c][-1][z], 1, id, delay=d+i*j))
                            id += 1
                    elif y == 0: 
                        for r in range(half_size+1):
                            synapses.append(synapse(last_layer_neurons[x+half_size][y+r][z], neurons[-1][half_size+r][z], 1, id, delay=d+i*j))
                            id +=1 
                    else: 
                        synapses.append(synapse(last_layer_neurons[x+half_size][y+half_size][z], neurons[-1][-1][z], 1, id, delay=d+i*j))
                        id += 1
                    j += 1 

        #inner synapses
        for z in range(w[2]):
            for y in range(w[1]):
                for x in range(w[0]):
                    synapses.append(synapse(self.init_neuron, neurons[x][y][z], -2, id, delay=d+i*(n[0]*n[1])))
                    id += 1 
                    if x != w[0] - 1: 
                        synapses.append(synapse(neurons[x+1][y][z], neurons[x][y][z], 1, id, delay=i))
                        id += 1
                    if y != w[1] - 1:
                        synapses.append(synapse(neurons[x][y+1][z], neurons[x][y][z], 1, id, delay=n[0]*i))
                        id += 1

        #negative charge

        #output synapses



        self.synapse_id = id 
        self.neurons.append(neurons)
        self.synapses.append(synapses)


            #something wrong with the x axis

        #test

        f = open("test.net", "w");
        # Whetstone num_neurons num_synapses inputs outputs
        #IMPORTANT
        f.write("Whetstone %d %d %d\n" % (n[0]*n[1]*n[2] + w[0]*w[1]*w[2] + 1, n[0]*n[1]*n[2], w[0]*w[1]*w[2]))

        for z in range(n[2]):
            for y in range(n[1]):
                for x in range(n[0]):
                    f.write(last_layer_neurons[x][y][z].print_neuron("I"))
        f.write(self.init_neuron.print_neuron("H"))

        for z in range(w[2]):
            for y in range(w[1]):
                for x in range(w[0]):
                    f.write(neurons[x][y][z].print_neuron("O"))


        for i in range(len(self.synapses)):
            for j in range(len(self.synapses[i])):
                f.write(self.synapses[i][j].print_synapse())




    def Init(self):
        
        # create init neuron 
        self.init_neuron = neuron(0, 0, 0.5, self.neuron_id, self.z_start)

        self.neuron_id += 1
        self.z_start += 1

        #connect inputs with the init neuron
        inputs = self.neurons[0]      
        synapses = []

        id = self.synapse_id

        for z in range(self.last_layer_size[2]):
            for y in range(self.last_layer_size[1]):
                for x in range(self.last_layer_size[0]):
                    synapses.append(synapse(inputs[x][y][z], self.init_neuron, 1, id, 0))
                    id += 1 
        self.synapses.append(synapses)
        self.synapse_id = id

        #didn't save init_neuron to self.neurons


    # def PB(self, i, d, w, init, negative=None): 









        



class nida():
    def __init__(self, neuro, filename="nida.net"):
        self.print_model(neuro, filename)

    def print_model(self, neuro, filename):
        f = open(neuro.file_path + filename, "w")
        f.write("# CONFIG\nDIMS %d %d %d\nGRAN 1\n\nEND-CONFIG\n# NETWORK\n" % (neuro.xy_size, neuro.xy_size, neuro.z_start))
        for layer in neuro.neurons: 
            for i in range(len(layer)):
                f.write(self.print_neuron(layer[i],"N"))
        for layer in neuro.synapses:
            for i in range(len(layer)):
                f.write(self.print_synapse(layer[i]))
        for i in range(len(neuro.neurons[0])):
            f.write(self.print_neuron_IO(neuro.neurons[0][i],"INPUT"))

        for i in range(len(neuro.neurons[-1])):
            f.write(self.print_neuron_IO(neuro.neurons[-1][i],"OUTPUT"))
        f.close()      
    
    def print_neuron(self, neuron, type):
        return "%s %d %d %d %lf %lf\n" % (type, neuron.x, neuron.y, neuron.z, neuron.threshold, neuron.refc)

    def print_neuron_IO(self, neuron, type):
        return "%s %d %d %d %d\n" % (type, neuron.id, neuron.x, neuron.y, neuron.z)

    def print_synapse(self, synapse):
        return "S %d %d %d %d %d %d %lf %lf\n" % (synapse.pre_n.x, synapse.pre_n.y, synapse.pre_n.z, synapse.post_n.x, synapse.post_n.y, synapse.post_n.z, synapse.weight, synapse.delay)

#waiting on Nick
class mrdanna():
    def __init__(self, neuro, filename="mrdanna.net"):
        self.neuro = neuro

class danna2():
    def __init__(self, neuro, filename="danna2.net"):
        self.neuro = neuro
        self.print_model(neuro, filename)

    def print_model(self, neuro, filename):
        f = open(neuro.file_path + filename, "w")
        print(neuro.file_path + filename)
        f.write("# MODEL DANNA2\nVersion: 0.1\nw %d h %d l 0 s 0\n" % (neuro.xy_size, neuro.xy_size))

        for i in range(len(neuro.neurons)):
            for j in range(len(neuro.neurons[i])):
                f.write(self.print_neuron(neuro.neurons[i][j]))
                for s in neuro.neurons[i][j].pre_syn:
                    f.write(self.print_synapse(s))

        for i in range(len(neuro.neurons[0])):
            f.write(self.print_I(neuro.neurons[0][i]))

        for i in range(len(neuro.neurons[-1])):
            f.write(self.print_O(neuro.neurons[-1][i], i))

    def print_neuron(self, neuron):
        return "N %d %d %d %d\n" % (neuron.x+neuron.z*self.neuro.xy_size, neuron.y, neuron.threshold*1023, neuron.refc)

    def print_synapse(self, synapse):
        return "\tS %d %d %d %d\n" % (synapse.pre_n.x+synapse.pre_n.z*self.neuro.xy_size, synapse.pre_n.y, synapse.weight*1023, synapse.delay)

    def print_I(self, neuron):
        return "I %d %d %d\n" % (neuron.id, neuron.x+neuron.z*self.neuro.xy_size, neuron.y)

    def print_O(self, neuron, id):
        return "O %d %d %d\n" % (id, neuron.x+neuron.z*self.neuro.xy_size, neuron.y)

class whetstone():
    def __init__(self, neuro, filename="whetstone.net"):
        self.print_model(neuro, filename)

    def print_model(self, neuro, filename):
        f = open(neuro.file_path + filename, "w");
        # Whetstone num_neurons num_synapses inputs outputs
        f.write("Whetstone %d %d %d\n" % (neuro.num_neurons(), len(neuro.neurons[0]), len(neuro.neurons[-1])))

        for i in range(len(neuro.neurons[0])):
            f.write(self.print_neuron(neuro.neurons[0][i], "I"))
        for i in range(len(neuro.neurons[-1])):
            f.write(self.print_neuron(neuro.neurons[-1][i],"O"))

        for layer in neuro.neurons[1:-1]:
            for i in range(len(layer)):
                f.write(self.print_neuron(layer[i],"H"))

        for layer in neuro.synapses:
            for i in range(len(layer)):
                f.write(self.print_synapse(layer[i]))

    def print_neuron(self, neuron, type):
        return "+ %s %d %f %d %d %d\n" % (type, neuron.id, neuron.threshold, neuron.x, neuron.y, neuron.z)

    def print_synapse(self, synapse):
        return "| S %d %f %f %d %d\n" % (synapse.id, synapse.weight, synapse.delay, synapse.pre_n.id, synapse.post_n.id)

class neuron():

    def __init__(self, x, y, threshold, id, z=0, refc=0):
        self.x = x
        self.y = y
        self.z = z
        self.threshold = threshold
        self.refc = refc
        self.id = id
        self.pre_syn = []

    def append_pre_synapse(self, synapse):
        self.pre_syn.append(synapse)

    def print(self):
        print("N (%d,%d,%d) id:%d thre:%f refc:%f" % (self.x, self.y, self.z, self.id, self.threshold, self.refc))

    def print_neuron(self, type):
        return "+ %s %d %f %d %d %d\n" % (type, self.id, self.threshold, self.x, self.y, self.z)

class synapse():

    def __init__(self, pre_n, post_n, weight, id=0, delay=0):
        self.pre_n = pre_n
        self.post_n = post_n
        self.weight = weight
        self.delay = delay
        self.id = id

    def print(self):
        print("S %d (%d,%d,%d)->(%d,%d,%d) weight:%f delay:%d" % (self.id, self.pre_n.x, self.pre_n.y, self.pre_n.z, self.post_n.x, self.post_n.y, self.post_n.z, self.weight, self.delay))

    def print_synapse(self):
        return "| S %d %f %f %d %d\n" % (self.id, self.weight, self.delay, self.pre_n.id, self.post_n.id)


