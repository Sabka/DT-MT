This folder contain implementation of our model MT-SOM which is semisupervised model inspired by Mean Teacher model. The difference is in consistency loss. MT-SOM model use loss based on Self-organizing map model. This consistency loss is called SOM loss and is the euclidian distance of winner neurons of feature vectors of student and teacher model from MT.

In this folder we also implement experiment - classification of CIFAR10 dataset with 4000 labeled samples.

Steps to run model:

1. Prepare CIFAR-10 dataset as  described in root directory.
2. Pretrain MT model to get feature vectors. Run `python supervised.py` in the conda environment described in root directory. It will pretrain MT model for 10 epochs. Student accuracy should be arround 65%.
3. In `ckpts/cifar10/<datetime of training>/convlarge,Adam,200epochs,b256,lr0.2/checkpoint.10.ckpt` you will run checkpoint of pretrained model.
4. Then you should change the path to your checkpoint in file `som-from-fv.py`, line 270 and file `mt-som.py`, line 57. Or you can use default checkpoint with student accuracy 66%.
5. Then you need to train SOM running `python som_from_fv.py` which use file `checkpoint_fv.py` to create feature vectors from mean teacher checkpoint. The SOM weights will be saved in `fv_som_results/pretrained_som-{ep}ep.pt` for several epochs. By figures of SOM in folder `figs`, you can choose epoch with best SOM metrices (we choose ep. 79, if you choose different, change mt-som.py, line 54).
6. Finally you can run model `mt-som.py` which will use SOM checkpoint and MT checkpoint and continue training MT-SOM model using original MT student weights and SOM loss.

