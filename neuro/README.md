# SDNN to SNN Converter

----
# Overview 

The goal of this project is to convert Whetstone trained spiking deep neural networks to TENNLab's spiking neural networks. This project include `neuro`, `neuron`, `synapse`, `WHETSTONE`, `DANNA2`, `MRDANNA`, and `NIDA` class. The `neuro` class decomposes a Keras network to `neuron`s and `synapse`s. Depends on the user, `neuro` calls one of `WHETSTONE`, `DANNA2`, `MRDANNA`, and `NIDA` to print the network in different architecture. 

Note: this project requires a Keras trained network model, so please place the code after the trained network. 

----
# Running the Converter and the Output Files 

Make sure you have a trained Keras network model and a Softmax layer key. 

The neuro class is the main converter, to convert a Keras network `model` with Softmax key `key` to TENNLab's `whetstone` archiecture, call neuro with the following arguments: 

```
neuro(model, key, "whetstone")
```

This will convert the network to a whetstone architecture SNN. The converter creates two files: `whetstone.net` and `supply.txt`. `whetstone.net` contains the actual network, and `supply.txt` has the key and first layer synapse weights. The `supply.txt` is required for running classification application to perform softmax and mimic the first layer SDNN behavior.   

The neuro class accept an optional argument `path` that will create the output files in user specified location. `path` is `supply.txt`'s location, and `whetstone.net` is placed two directory up with `supply.txt`. Please note, the `whetstone.net` location is default to place relatively with `supply.txt` location base on the current class2d application. If `path` is not set, the files are created in the current directory. 

An example of save `whetstone.txt` in `~/neuro/apps/class2d` and `supply.txt` in `~/neuro/apps/class2d/data/mnist`: 

```
neuro(model, key, "whetstone", "~/neuro/apps/class2d/data/mnist")
```

---- 
# The Components

## neuro Class

`neuro` is the main converter, it breaks the Keras network into components for SNN. Running the `neuro` class include two branches, first is the main branch that set up the convertion enviroment, and second is the convert branch that calls the different convertion functions. 


### main brach 

An simple graph for the main branch: 

```
__init__()-|                                                                  |-> architecture classes 
           |->loading()                                 loading()          loading()
                 |                                         | |                |
                 |->load_config()->construct_supply_file()-| |->convert brach-|

```

The main branch first enter from `__init__()`, setting up the arguments, and then goes to `loading()`. The `loading()` function first calls `load_config()` to initialize the global variables, and then calls `construct_supply_files()` with Softmax key and first layer synpase weights. The `construct_supply_files()` function creates `supply.txt` and a global init neuron for negative threshold (spectial case in DNN). After this, it goes to a for loops that detects layer types and calls the actual convertion functions, and this part is call the convert brach. 


### convert brach

The convert brach coverts the for loop in the `loading()` function, but more important is the actual convertion functions. The for loop reads the layers in Keras network, and calls different convertion function. 


#### Conv2D

If the layer is a conv2d layer, the layer can be convert to SNN in two different version by two differen convertion functions: `whetstone_v1()` and `whetstone_v2()`. 

`whetstone_v1()` converts the layer by `load_conv2d_neurons()` and `load_conv2d_synapses()`, and `whetstone_v2()` uses `Init()`, `PA()`, `PB()`, and `PC()`.  


#### Maxpulling 

If the layer is a maxpulling layer, the layer is converted by `load_maxpooling_neurons()` and `load_maxpooling_synapses()`. 

#### Flatten

The Flatten layer reduce a 2D layer to a 1D layer, so the convertion is done by the for loop. 

#### Dense

Dense layer uses `load_dense_neurons()` and `load_dense_synapses()` to converts and decompose the layer to SNN neurons and synapses. 


## Architecture Classes

The converted, decomposed SNN can be written in different architecture format. This converter supports `Whetstone` and `DANNA2`, and `MrDANNA` maybe comming soon. Depends on the user, the `neuro` class calls different archicture class. Every archicture class has a `print_model()` that print the architecture constants, read the neurons and synapses, and print it out by calling `print_neuron()` and `print_synapse()` (sometime `print_neuron_IO()` too). 


































