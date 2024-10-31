# Deep-diffeomorphic networks for conditional brain templates
### [Paper](https://www.biorxiv.org/content/10.1101/2024.07.05.602288)
Official implementation of "Deep-diffeomorphic networks for conditional brain templates"

Pytorch code for *Deep-diffeomorphic networks for conditional brain templates*, bioRxiv 2024.

## Dependencies

We recommend setting up an anaconda environment and installing all dependencies as,

```bash
conda env create -f environment.yml
conda activate deep-diff
```

## Data

The configuration file `config.txt` contains a list of hyperparameters for the network. You will need to specify a path to the training data and an unconditional template in this file. The training dataset and unconditional template should be in the form of `.nii.gz` files. If you want to use a predefined dataset split, please set `predefined_split` to `True` and provide appropriate paths to the csv files containing lists of the training and validation scan paths along with ages in an `age_at_scan` column. We expect a minimum of these two columns in the csv files, although other covariates can be included as well, although the `datasets.py` and `trainer.py` files will need to be modified to accommodate additional covariates.

Training and validation data should be linearly registered to a common reference space. To do this, we recommend registering the data to the unconditional template using ANTs (http://stnava.github.io/ANTs/). 

## Training
To train a purely geometric conditional brain template construction network, run the following command from ./sh, ensuring that `train.sh` is executable (run `chmod +x train.sh` if not):
```bash
./train.sh -c config.txt -i <run_id> -d <cuda_device_no> -u True 
```

This will train the network using the configuration specified in `config.txt` on the specified cuda device. The `-u` flag is used to specify whether to save full checkpoints during training. The optional `-e` flag can be used to provide a path to a checkpoint to resume training from.

## Acknowledgements:
We have made use of the [VoxelMorph](https://github.com/voxelmorph/voxelmorph) library as part of this project.
