## Short description of experiments:
  1. bmt-animacy: Experiment test performance of semisupervised model Binary Mean Teacher on binary classification task. Experiment investigate model further, with standard image dataset CIFAR10. We focused on experimentation with different**very small** portions of labeled data, in which BMT model achieve much better performance than supervised baseline.
  2. cifar-vs-fv-som-visualizations: In this experiment we compared qualitative and quantitative metrices of Self-organizing map with two input types - vanilla CIFAR10 dataset and feature vectors of CIFAR10 from pretrained Mean Teacher. 
  3. som-loss-developement: In this experiment, we focused on unsupervised SOM-based loss function. We used this loss function in our model MLP-SOM and tested its performance in supervised setup.
  4. semisup: It this final experiment, we implemented and tested model MT-SOM witch is based on Mean Teacher model and use proposed SOM-based loss function.

