_____________________________________________________________________

<div align="center">

# Benchmarking image to image transformation model in medical imaging for misalignment error

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Develop benchmark model for image to image transformation model in medical imaging domain.

- Templates from https://github.com/ashleve/lightning-hydra-template was used.

- `Pytorch-lighting` + `Hydra` + `Tensorboard` was used for experiment-tracking

- Use `TensorBoard` for visualization (below)
```bash
 tensorboard --logdir logs 
 ``` 
 
## Installation

#### MAMBA

```bash
# clone project
git clone https://github.com/danny4159/Misalign-benchmark.git
cd Misalign-benchmark

# Install mamba [if not available]
conda install mamba -n base -c conda-forge

# Install environments
mamba env create -f environment.yaml
```

Install other libraries manually using pip and conda.


## Dataset
#### Dataset 
Grand challenge 'SynthRAD' MR to CT on Pelvis

#### Preprocessing
MR: N4 correction -> Nyul Histogram Matching -> z-score norm each patient -> -1~1 minmax norm each patient
CT: 5% 95% percentile clip -> z-score norm whole patient -> -1 ~ 1 minmax norm whole patient

#### File Format: 
h5

#### Dataset download
https://drive.google.com/drive/folders/19a9VF9TYMyg6TAnOyRokn4d46_Nfhvfa?usp=sharing

Download this preprocessed MR to CT dataset, and insert it into the 'data/SynthRAD_MR_CT_Pelvis' folder.


## How to run

#### Training
```bash
# PGAN
python misalign/train.py model='pgan.yaml' trainer.devices=[0] tags='synthRAD_PGAN_train'

# Proposed (Meta-learning)
python misalign/train.py model='proposed_A_to_B.yaml' trainer.devices=[0] tags='synthRAD_Proposed_train'
```
Various other models are implemented as well.

#### Test
Test phase is automatically executed after the training has been completed. If you want to run the Test phase manually

1st. In the train.py file, set the ckpt_path for the test execution.

2nd. Run the code
```bash
# PGAN
python misalign/train.py model='pgan.yaml' trainer.devices=[0] tags='synthRAD_PGAN_train' train=False

# Proposed (Meta-learning)
python misalign/train.py model='proposed_A_to_B.yaml' trainer.devices=[0] tags='synthRAD_Proposed_train' train=False
```


## References

```cite
1. S. U. Dar, M. Yurt, L. Karacan, A. Erdem, E. Erdem and T. Çukur, "Image Synthesis in Multi-Contrast MRI With Conditional Generative Adversarial Networks," in IEEE Transactions on Medical Imaging, vol. 38, no. 10, pp. 2375-2388, Oct. 2019, doi: 10.1109/TMI.2019.2901750.
```