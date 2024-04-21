# Learning to defer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides an implementation of ["Learning to defer"](https://proceedings.neurips.cc/paper/2018/hash/09d37c08f7b129e96277388757530c72-Abstract.html) using the EM approach (or Monte Carlo EM to be precise). Please see the wiki page for further details on the mathematical formulation of the implementation.

> NOTE: the implementation here is not limited to a single human (or expert), but also works in the presence of multiple experts.

## Required packages

The implementation is mainly written in [JAX](https://github.com/google/jax) with the data loading taken from [MLX-Data](https://github.com/ml-explore/mlx-data).

The list of Python packages required to run the implementation is placed inside `requirements.txt`.

## Data

Data is organised in JSON file where each file is a list of JSON object, each consists of two key - value pairs: `file` and `label`. For example, a JSON file has the following content:
```json
[
    {
        "file": "path/to/sample_1",
        "label": 0
    },
    {
        "file": "path/to/sample_2",
        "label": 8
    },
    ...
]
```

To run the code, at least 4 files need to be passed to the following arguments:
- `--train-files`
- `--train-groundtruth-file`
- `--test-files`
- `--test-groundtruth-file`

> NOTE: one can pass a list of files in case of multiple experts (see an example in `run.sh`).

## Experiment track and monitoring

[Mlflow](https://mlflow.org/) is used in experiment management. To track an experiment, please run `mlflow_server.sh`, then open the uri: `http://127.0.0.1:8080`. Please refer to *Mlflow* website if further details are required.

## An example of a run

The following presents the results of *"Learning to defer" to 2 synthetic experts where each expert has a label noise rate of 0.3 on CIFAR-10. The deferral model is a Pre-act ResNet-10, while the classifier is a 3-layer CNN. The x-axis is the training epoch.

### Testing accuracy

![testing accuracy](/img/accuracy.png)

### Coverage measured on the test set

![testing coverage](/img/coverage_test.png)

### Coverage measured on the training set

![training coverage](/img/coverage_train.png)

### Training loss of the classifier (3-layer CNN)

![classifier's loss](/img/loss_clf.png)

### Training loss of the deferral model (Pre-act ResNet-10)

![Deferral model's loss](/img/loss_dfr.png)
