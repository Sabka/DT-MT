## CIFAR10 vs. CIFAR10 FV inputs


In this experiment, we compared som metrices in case when vanilla CIFAR10 dataset is used as the input and when inputs are feature vectors from pretrained Mean Teacher model. 

We pretrained checkpoint of MT model named `checkpoint.300.ckpt` and use it to produce feature vectors. This checkpoint had size over 300MB, so it cant be stored in the GH repo, but it was optained by cloning [MT repo](github.com/CuriousAI/mean-teacher) and running 
```
python cifar10_test.py
```.

