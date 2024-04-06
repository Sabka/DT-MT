# Diploma thesis - Semisupervised learning of deep neural networks

This repo contains implementation of experiments for my diploma thesis. The title of my thesis is Semisupervised learning of deep neural networks.
I created several experiments, each in separate folder in folder pytorch/experiments.

Short description of experiments:
  1. bmt-animacy: Experiment test performance of semisupervised model Binary Mean Teacher on binary classification task. Experiment investigate model further, with standard image dataset CIFAR10. We focused on experimentation with different**very small** portions of labeled data, in which BMT model achieve much better performance than supervised baseline.
  2. cifar-vs-fv-som-visualizations: **TODO**
  3. som-loss-developement: **TODO**
  4. semisup: **TODO**


I run this experiment on GPU servers at FMFI UK, so it can be useful for my colegues to describe how I run it:

1. Conda environment - since we do not install all needed packages or libs locally, we use conda environments. For this project, it is possible to create such env like this:


: **TODO**
  

2. How to use screen?

`screen -S name-of-the-screen` 

run command to be execute

`ctrl + A` and `ctrl + D`

to return to screen 

`screen -ls` - list of all screens, find id

`screen -r [id/name]`

to terminate screen - `ctrl + D`


