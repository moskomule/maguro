# maguro

`maguro` is a simple job scheduler for multiple-GPU enviroments.

## Requirements

* Python >= 3.7

This library relies only on the standard libraries and `nvidia-smi`.

## Installation

`pip install -U git+https://github.com/moskomule/maguro`

## Usage

### Submit job

```bash
maguro push [-r,--num_repeat 1] [-g,--num_gpus 1] [--logdir maglog] COMMAND
```

Here, `COMMAND` is like `python ~/train.py`. `--num_gpus` specifies the number of GPUs required to run the submitted job.

**Note that paths in `COMMAND` should be absolute if you run from different directory** 

### Run jobs

```bash
maguro run [--forever]
```

### List remaining jobs

```bash
maguro list [--all]
```

### Delete jobs

```bash
maguro delete IDS
```

This `IDS` can be checked by `maguro list --all`.