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
        # self.print_whetstone()
        # whetstone(self)
        # nida(self)
        danna2(self)

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

                #setup neurons
                weights, thresholds = layer.get_weights()
                config = layer.get_config()
                filters = config["filters"]
                kernel_size = config["kernel_size"]

                #increase z axis starting address  
                self.load_conv2d_neurons(0.5 - thresholds)
                
                #update to current size 
                self.load_conv2d_synapses(filters, kernel_size, weights)

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
        print(self.last_layer_size)

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
        return "N %d %d %d 0\n" % (neuron.x+neuron.z*self.neuro.xy_size, neuron.y, neuron.threshold*1023)

    def print_synapse(self, synapse):
        return "\tS %d %d %d 0\n" % (synapse.pre_n.x+synapse.pre_n.z*self.neuro.xy_size, synapse.pre_n.y, synapse.weight*1023)

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




#TODO: move this to its individual model
    # def print_nida(self, filename="nida.net"):
    #     f = open(self.file_path + filename, "w");
    #     f.write("# CONFIG\nDIMS %d %d %d\nGRAN 1\n\nEND-CONFIG\n# NETWORK\n" % (self.xy_size, self.xy_size, self.z_start))
    #     for neuron in self.neurons: 
    #         for y in range(len(layer)):
    #             f.write(layer[y].print_nida("N"))
    #     for synapse in self.synapses:
    #         for i in range(len(synapse)):
    #             f.write(layer[i].print_nida())
    #     for i in range(len(self.neurons[0])):
    #         f.write(self.neurons[0][i].print_nida_IO("INPUT"))

    #     for i in range(len(self.neurons[-1])):
    #         f.write(self.neurons[-1][i].print_nida_IO("OUTPUT"))
    #     f.close()

    # def print_whetstone(self, filename="whetstone.net"):
    #     print(self.file_path + filename)
    #     f = open(self.file_path + filename, "w");
    #     # Whetstone num_neurons num_synapses inputs outputs
    #     f.write("Whetstone %d %d %d\n" % (self.num_neurons(), len(self.neurons[0]), len(self.neurons[-1])))

    #     for i in range(len(self.neurons[0])):
    #         f.write(self.neurons[0][i].print_whetstone("I"))
    #     for i in range(len(self.neurons[-1])):
    #         f.write(self.neurons[-1][i].print_whetstone("O"))

    #     for layer in self.neurons[1:-1]:
    #         for i in range(len(layer)):
    #             f.write(layer[i].print_whetstone("H"))

    #     for layer in self.synapses:
    #         for i in range(len(layer)):
    #             f.write(layer[i].print_whetstone())

    # def print_danna2(self, filename="danna2.txt"):
    #     syn_t = len(self.neurons[0]) * len(self.neurons[1])
    #     f = open(filename, "w");
    #     f.write("# MODEL DANNA2\nVersion: 0.1\nw %d h %d l 0 s 0\n" % (self.xy_size, self.xy_size))

    #     for i in range(len(self.neurons)):
    #         for j in range(len(self.neurons[i])):
    #             f.write(self.neuron[i][j])

    #     for i in range(2, len(self.neurons)):
    #         for j in range(len(self.neurons[i])):
    #             f.write(self.neurons[i][j].print_danna2())
    #             f.write("".join(s.print_danna2() for s in self.synapses[syn_t:syn_t+len(self.neurons[i-1])]))
    #             syn_t += len(self.neurons[i-1])
    #     f.write("".join("I %d %d %d\n" % (i, i, 0) for i in range(len(self.neurons[1]))))
    #     f.write("".join("O %d %d %d\n" % (i, i, len(self.neurons)) for i in range(len(self.neurons[-1]))))

    # def print_neuron(self, neuron):
    #     return "N %d %d %d 0\n" % (self.x, self.y, self.threshold*1023)

    # def print_synapse(self, synapse):
    #     return "\tS %d %d %d 0\n" % (self.pre_n.x, self.pre_n.y, self.weight*1023)


#TODO: move to its individual model
class neuron():

    def __init__(self, x, y, threshold, id, z=0, refc=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.threshold = threshold
        self.refc = refc
        self.id = id
        self.pre_syn = []

    def append_pre_synapse(self, synapse):
        self.pre_syn.append(synapse)

    # def print_out(self, i, type):
    #     return "%-6s Neuron %2d: %10s [threshold:%lf]\n" % (type, i, self.print_coord(), self.threshold)

    # def print_nida(self, type):
    #     return "%s %d %d %d %lf %lf\n" % (type, self.x, self.y, self.z, self.threshold, self.refc)

    # def print_nida_IO(self, type):
    #     return "%s %d %d %d %d\n" % (type, self.id, self.x, self.y, self.z)

    # def print_nida_output(self, type):
    #     return "%s %d %d %d 0\n" % (type, self.x, self.x, self.y)

    # def print_whetstone(self, type):
    #     return "+ %s %d %f %d %d %d\n" % (type, self.id, self.threshold, self.x, self.y, self.z)

    # def print_danna2(self):
    #     return "N %d %d %d 0\n" % (self.x, self.y-1, self.threshold*1023)

    # def print_coord(self):
    #     return "%d|%d" % (self.x, self.y)

class synapse():

    def __init__(self, pre_n, post_n, weight, id=0, delay=1):
        self.pre_n = pre_n
        self.post_n = post_n
        self.weight = weight
        self.delay = delay
        self.id = id

    # def print_out(self, i):
    #     return "Synapse %2d: %19s [weights:%lf]\n" % (i, self.print_coord(), self.weight)

    # def print_nida(self):
    #     return "S %d %d %d %d %d %d %lf %lf\n" % (self.pre_n.x, self.pre_n.y, self.pre_n.z, self.post_n.x, self.post_n.y, self.post_n.z, self.weight, self.delay)

    # def print_whetstone(self):
    #     return "| S %d %f %f %d %d\n" % (self.id, self.weight, self.delay, self.pre_n.id, self.post_n.id)

    # def print_danna2(self):
    #     return "\tS %d %d %d 0\n" % (self.pre_n.x, self.pre_n.y-1, self.weight*1023)

    # def print_coord(self):
    #     return "%s->%s" % (self.pre_n.print_coord(), self.post_n.print_coord())
