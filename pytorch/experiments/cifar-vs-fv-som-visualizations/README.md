## CIFAR10 vs. CIFAR10 FV inputs


In this experiment, we compared som metrices of Self-organizing map in case when vanilla CIFAR10 dataset is used as the input and when inputs are feature vectors from pretrained Mean Teacher model. Dataset CIFAR10 needs to be prepared as described in README in root of repo.

We pretrained checkpoint of MT model named `checkpoint.300.ckpt` and use it to produce feature vectors. This checkpoint had size over 300MB, so it cant be stored in the GH repo, but it was optained by cloning [MT repo](github.com/CuriousAI/mean-teacher) and running `python cifar10_test.py`. This script is copied also in this repo in folder experimets.

The main experiment files are `cifar10-som.py` and `fv-som.py`. They both train SOM on one type of input data. 
`fv-som.py` takes MT checkpoint (line 293) and produce feature vectors, than use it as input for SOM. Both codes produce figures into figs and figsf folders respectively and SOM checkpoints in the folder models. 

`som-weights-visualization.py` takes checkpoint of SOM trained using vanilla CIFAR10 and visualize weights of model.
