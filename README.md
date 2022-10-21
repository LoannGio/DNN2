# Presentation
This repository contains main code features of the paper __Deep Neural Network for DrawiNg Networks, (DNN)²__, appeared in the Proceedings of the 29th International Symposium on Graph Drawing and Network Visualization (GD 2021) and available at https://arxiv.org/abs/2108.03632.

It contains research code that was not written to be extensively maintained. 

The repository does not provides the model and benchmarks from the paper. However, all the methods (DNN layers, data processing, metrics) code are shared for reproducibility.

# Requirements
This project was conducted using some ``dependencies" whose behavior might change according to their version. Here follows the list of the versions that were used (and hopefully work) for these dependencies.
* Python 3.8.5
* Pyspark 3.0.1 (Scalable data processing) ; can require specific installation (on Windows OS)
* TF/Keras 2.4.1 and TF/Keras 2.9.0 (Deep Learning) ; requires additionnal dependencies (CUDA, cudNN)
* tulip-python 5.5.1 (Graph management)

All Python dependencies are listed in [requirements.txt](requirements.txt).

# Organisation
* [data](data/) contains various graphs, including Rome Graphs taken from the [Graph Drawing](http://www.graphdrawing.org/data.html) website in format .tlpb.gz
* [code](code/) python code of (DNN)²
* [train.py](train.py) Example of main file to train a model using DNN2_Wrapper class.
* [predict.ipynb](predict.ipynb) Notebook providing examples to generate DNN2 predictions, visualize them and compute aesthetic metrics.

More precisely, [code](code/) contains:
* [DNN2_Wrapper.py](code/DNN2_Wrapper.py) Contains a class that paritions the different stages of data preprocessing and model training. 
* [graph_preprocessing.py](code/graph_preprocessing.py) All the operations we need to do on/with graphs.
* [spark_preprocessing.py](code/spark_preprocessing.py) Scalable handling of data generation.
* [sequences.py](code/sequences.py) Data structure used to feed the model during training.
* [losses.py](code/losses.py) Custom loss functions. Contains ``stress`` and ``tsNET`` losses.
* [graph_layers.py](code/graph_layer.py) Our implementation of some Graph Neural Network layers
* [models.py](code/models.py) Code to build the Neural network architecture.
* [custom_callbacks.py](code/custom_callbacks.py) VisualSampleCallback, a custom callback that, at the end of each epoch, saves some graph layout examples of that epoch model. Used to keep visual tracking of the model layouts evolution throughut the training
* [SOTA_layouts.py](code/SOTA_layouts.py) Interfaces to layout graphs with state of the art layout algorithms and trained (DNN)².
* [custom_graphs.py](code/custom_graphs.py) Provides interface to easily generate some common graphs
* [metrics.py](code/metrics.py) Metrics used to evaluate layouts. 

# Hyper parameters
This version of the code enables some design choices.

**Loss function** : *stress* and *tsNET* loss functions are implemented. To reproduce *tsNET* training as explained in (DNN)² paper, the model has to be trained twice (stage 1 and stage 2). The *lambda* parameters of each stage is commented in [DNN2_Wrapper.py](code/DNN2_Wrapper.py). For transfer learning, simply give the weights to be loaded in the corresponding parameter in [train.py](train.py).

**Topological structure** : *Chebyshev filters* and *GCN filters* are implemented. For *Chebyshev filters*, a max degree *K* has to be set (> 0 to be meaningful). For *GCN filters* the parameter *K* should be set to 0.

**Nodes features** : *unique node id*, *random metric* and *2D layout* (supported by tulip-python API) are supported and several can be given as input to the model. More information are directly commented in [train.py](train.py).

**Other hyper-parameters** : most other common hyper-parameters (e.g., batch size, optimizer) can be provided through the `DNN2_Wrapper` class, more specifically in its methods `__init__`, `prepare_data` and `train`.

# Contacts
Loann Giovannangeli: loann.giovannangeli@u-bordeaux.fr\
Frederic Lalanne: frederic.lalanne@u-bordeaux.fr\
David Auber: david.auber@u-bordeaux.fr\
Romain Giot: romain.giot@u-bordeaux.fr\
Romain Bourqui: romain.bourqui@u-bordeaux.fr

# Cite
```
@inproceedings{giovannangeli2021dnn2,
  title={{Deep Neural Network for DrawiNg Networks, (DNN)\textsuperscript{2}}},
  author={{Giovannangeli, Loann and Lalanne, Frederic and Auber, David and Giot, Romain and Bourqui, Romain}},
  booktitle={{International Symposium on Graph Drawing and Network Visualization}},
  pages={375--390},
  year={2021},
  organization={Springer}
}
```
