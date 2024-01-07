# pytorch-template

The purpose of this repo is to provide a template for deep-learnng running experiments (i.e. hyper-parameter) search using pytorch. Weights and Biases is used to define runs (single
set of model parameters) and sweeps (a hyper-parameter search space to search using
grid/random/bayes) and log results and artifacts.

In order to use this template you need to sign up for an account in [Weights and Biases](https://wandb.ai) and ensure your api key is available as an environment variable (`WANDB_API_KEY`) (e.g. via .env file).

- Available deep learning architectures are found in `source/library/architectures.py`
- Training logic is found in `source/library/experiment.py`
    - Currently, training uses a combination of early stopping and learning rate decay. Once early stopping is triggered, the learning rate is halved. This happens `num_reduce_learning_rate` times.
- Default/template runs/sweeps for each architecutre are found in `experiments/templates`
- `Makefile` contains helpful commands and examples.
- This repo is meant to be used as a template and starting point, not as a framework. Meaning, the you'll want to 

# Branches

- `mnst-snapshot` contains an example of training a fully-connected and convolutional neural network on the mnist dataset (1 channel; gray)
- `smile-snapshot`: 

# Example - Running CNN Random Search

In order to use this template you need to sign up for an account in Weights and Biases and ensure
your api key is available as an environment variable (`WANDB_API_KEY`) (e.g. via .env file).

If we had the file `experiments/runs/run_cnn_100.yaml` (where `100` stands for the 100th experiment we're trying), then we can run the following command:

```
python cli.py sweep \
    -config_file=$$(ls experiments/sweeps/sweep_cnn_*.yaml | sort -V | tail -n 1) \
    -runs=70
```

This command will grab the file `sweep_cnn_*.yaml` where `*` is the highest version, in this case `100`.

This command can also be ran from the Makefile with `make sweep_cnn` and will run the config file with the highest version.

The results will be logged to the console and to Weights and Biases
