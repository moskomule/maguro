# maguro

`maguro` is a simple job scheduler for multiple-GPU enviroments.

## Usage

Suppose you want to run the following command `NUM_REPEAT` times.

```
python train.py --model resnet20
```

Then, if you run these tasks three times each,

```
maguro -r NUM_REPEAT -n NUM_GPU_PER_TRIAL python train.py --model resnet20
```

You can use `--dryrun` to check which commands will be executed.

## Requirements

* Python >= 3.7

This library relies only on the standard libraries and `nvidia-smi`.

## Installation

`pip install -U git+https://github.com/moskomule/maguro`
