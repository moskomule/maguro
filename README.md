# maguro

`maguro` is a simple job scheduler for multiple-GPU enviroments.

## Usage

Prepare a task list file at first (suppose the file name is `tuna` ).

```
python train.py --model resnet20
python train.py --model resnet56
python train.py --model vgg19
```

Then, if you run these tasks three times each,

```
maguro tuna -n 3
```

You can use `--dryrun` to check which commands will be executed.

## Requirements

* Python >= 3.7

This library relies only on the standard libraries and `nvidia-smi`.

## Installation

`pip install -U git+https://github.com/moskomule/maguro`

## Limitation

Currently, each job is assumed to use a single GPU.
