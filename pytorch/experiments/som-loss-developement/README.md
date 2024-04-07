## SOM loss developement in supervised setup

In this folder, you can find implementation of experiment in which we compare MLP model with 
introduced SOM loss and without. 

We implemented the experiment with two different table datasets - Wine and Zoo datasets.
Implementation of MLP with Wine dataset is in file `model.py` and MLP with auxilary SOM loss 
are in file `model-gauss-w-init.py` and `model-tarv-ramp.py`. 

Implementation of MLP with Zoo dataset with bots setups is in file model-zoo.py. 
If we want vanilla MLP setup without auxilary loss, we set kappa = 0.
