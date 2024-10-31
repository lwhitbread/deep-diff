# Deep-diffeomorphic networks for conditional brain templates
### [Paper](https://www.biorxiv.org/content/10.1101/2024.07.05.602288)
Official implementation of "Deep-diffeomorphic networks for conditional brain templates"

Pytorch code for *Deep-diffeomorphic networks for conditional brain templates*, bioRxiv 2024.

## Abstract

Deformable brain templates are an important tool in many neuroimaging analyses. Conditional templates (e.g., age-specific templates) have advantages over single population templates by enabling improved registration accuracy and capturing common processes in brain development and degeneration. Conventional methods require large, evenly-spread cohorts to develop conditional templates, limiting their ability to create templates that could reflect richer combinations of clinical and demographic variables. More recent deep-learning methods, which can infer relationships in very high dimensional spaces, open up the possibility of producing conditional templates that are jointly optimised for these richer sets of conditioning parameters. We have built on recent deep-learning template generation approaches using a diffeomorphic (topology-preserving) framework to create a purely geometric method of conditional template construction that learns diffeomorphisms between: (i) a global or group template and conditional templates, and (ii) conditional templates and individual brain scans. We evaluated our method, as well as other recent deep-learning approaches, on a dataset of cognitively normal participants from the Alzheimer's Disease Neuroimaging Initiative (ADNI), using age as the conditioning parameter of interest. We assessed the effectiveness of these networks at capturing age-dependent anatomical differences. Our results demonstrate that while the assessed deep-learning methods have a number of strengths, they require further refinement to capture morphological changes in ageing brains with an acceptable degree of accuracy. The volumetric output of our method, and other recent deep-learning approaches, across four brain structures (grey matter, white matter, the lateral ventricles and the hippocampus), was measured and showed that although each of the methods captured some changes well, each method was unable to accurately track changes in all of the volumes. However, as our method is purely geometric it was able to produce T1-weighted conditional templates with high spatial fidelity and with consistent topology as age varies, making these conditional templates advantageous for spatial registrations. The use of diffeomorphisms in these deep-learning methods represents an important strength of these approaches, as they can produce conditional templates that can be explicitly linked, geometrically, across age as well as to fixed, unconditional templates or brain atlases. The use of deep-learning in conditional template generation provides a framework for creating templates for more complex sets of conditioning parameters, such as pathologies and demographic variables, in order to facilitate a broader application of conditional brain templates in neuroimaging studies. This can aid researchers and clinicians in their understanding of how brain structure changes over time, and under various interventions, with the ultimate goal of improving the calibration of treatments and interventions in personalised medicine. The code to implement our conditional brain template network is available at: https://github.com/lwhitbread/deep-diff.

**Keywords**: conditional templates, diffeomorphic networks, neuroimaging

## Dependencies

We recommend setting up an anaconda environment and installing all dependencies using the provided `environment.yml` file. To do this, run the following commands from the root directory of the repository:

```bash
conda env create -f environment.yml
conda activate deep-diff
```

## Data

The configuration file `config.txt` contains a list of hyperparameters for the network. You will need to specify a path to the training data and an unconditional template in this file. The training dataset and unconditional template should be in the form of `.nii.gz` files. If you want to use a predefined dataset split, please set `predefined_split` to `True` and provide appropriate paths to the csv files containing lists of the training and validation scan paths along with ages in an `age_at_scan` column. We expect a minimum of these two columns in the csv files, although other covariates can be included as well, although the `datasets.py` and `trainer.py` files will need to be modified to accommodate additional covariates.

Training and validation data should be linearly registered to a common reference space. To do this, we recommend registering the data to the unconditional template using ANTs (http://stnava.github.io/ANTs/). 

## Training
To train a purely geometric conditional brain template construction network, run the following command from `./sh`, ensuring that `train.sh` is executable (run `chmod +x train.sh` if not):
```bash
./train.sh -c config.txt -i <run_id> -d <cuda_device_no> -u True 
```

This will train the network using the configuration specified in `config.txt` on the specified cuda device. The `-u` flag is used to specify whether to save full checkpoints during training. The optional `-e` flag can be used to provide a path to a checkpoint to resume training from.

## Acknowledgements:
We have made use of the [VoxelMorph](https://github.com/voxelmorph/voxelmorph) library as part of this project.
