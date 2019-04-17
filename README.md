# maguro

`maguro` is a simple job scheduler for multiple-GPU enviroments.

## usage

Prepare a task list at first (suppose the file name is `tuna` ).

```
python train.py --model resnet20
python train.py --model resnet56
python train.py --model vgg19
```

Then, if you run these tasks three times each,

```
maguro tuna -n 3
```
