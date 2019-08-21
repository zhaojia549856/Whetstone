# SDNN to SNN Converter

----
# Overview 

The goal of this project is to convert Whetstone trained spiking deep neural networks to TENNLab's spiking neural networks. This project include `neuro`, `neuron`, `synapse`, `whetstone`, `danna2`, `mrdanna`, and `nida` class. The `neuro` class decomposes a Keras network to `neuron`s and `synapse`s. Depends on the user, `neuro` calls one of `whetstone`, `danna2`, `mrdanna`, and `nida` to print the network in different architecture. 

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

## neuro class

`neuro` is the main converter, it breaks the Keras network into components for SNN. 


































