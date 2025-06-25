# CF-DTI: Drug-Target Interaction Prediction Based on Coarse-to-Fine Feature Extraction Framework

<div align="left">


</div>



## Datasets
The `datasets` folder contains all experimental data used in CF-DTI.


## Run `CF-DTI on Our Data to Reproduce Results

To train CF-DTI, where we provide the basic configurations for all hyperparameters in `config.py`.

Select the dataset by selecting one of the following code in main.py:
```
$ args = parser.parse_args(['--data','bindingdb','--split','random'])
$ args = parser.parse_args(['--data','biosnap','--split','random'])
$ args = parser.parse_args(['--data','human','--split','random'])
$ args = parser.parse_args(['--data','celegans','--split','random'])
```

Then, you can directly run the following command with one or two GPUs. 
```
$ torchrun --standalone --nproc_per_node=1 main.py
$ torchrun --standalone --nproc_per_node=2 main.py
```
