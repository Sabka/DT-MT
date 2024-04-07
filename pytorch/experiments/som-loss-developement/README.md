## SOM loss developement in supervised setup

In this folder, you can find implementation of experiment in which we compare MLP model with 
introduced SOM loss and without. For SOM loss computation, we need to pretrain SOM on such dataset.

We implemented the experiment with two different table datasets - Wine and Zoo datasets.

1. Wine
We implement som training and saving of SOM for Wine dataset in file `som.py`.
Implementation of MLP with Wine dataset is in file `model.py` and MLP with auxilary SOM loss 
are in file `model-gauss-w-init.py` and `model-tarv-ramp.py`. 

2. Zoo
We implement som training and saving of SOM for Wine dataset in file `som-zoo.py`.
Implementation of MLP with Zoo dataset with both setups is in file model-zoo.py. 
If we want vanilla MLP setup without auxilary loss, we set kappa = 0. Same is applicable for previus dataset.
