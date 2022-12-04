# EEG Channel Reduction
Dimensionality reduction for electroencephalography data to be used in deep learning models.

In this project, we explored the viability of various data reduction techniques applied to EEG data such as PCA, kPCA, LSTM autoencoders, and channel selection. Much of this code needs tweaking if you'd like to get everything up and running yourself, but there is a sample program that's available to try things out. This sample program loads a couple of our pretrained models and gathers accuracy on a subset of test data taken from a larger test set. See our paper, hopefully coming soon, for more information on what we explored.


## Installation and Setup

To get everything up and running, download what you need from this repo, then execute:

...
conda create -f environment.yml
...
