# EEG Channel Reduction
Dimensionality reduction for electroencephalography data to train deep learning models for motor imagery classification.

In this project, we explored the viability of various data reduction techniques applied to EEG data such as PCA, kPCA, LSTM autoencoders, and channel selection. Much of this code needs tweaking if you'd like to get everything up and running yourself, but there is a sample program that's available to try things out. This sample program loads a couple of our pretrained models and gathers accuracy on a subset of test data taken from a larger test set called the High Gamma Dataset. See our paper, hopefully coming soon, for more information on what we explored.


## Installation and Setup

This code requires conda, if you'd like the easiest setup, but you're welcome to use anything else so long as you follow the requirements outlined in the ``` environment.yml ``` file.

To get everything up and running, download what you need from this repo, then execute:

```
$ conda create -f environment.yml
```

After installing all necessary dependencies, activate your environment and run the sample script in ```/code/```:
```
$ python run_sample_program.py
```

## Questions
Feel free to direct any comments, questions, or requests to soren.j.madsen@gmail.com. This is a work in progress to make more accessible, and it won't be at the top of my list for a little while. 
