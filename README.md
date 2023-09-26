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
git clone git@github.com:KHRyu8985/misalign-benchmark.git
cd misalign-benchmark

# Install mamba [if not available]
conda install mamba -n base -c conda-forge

# Install environments
mamba env create -f environment.yaml
```

## Dataset
https://drive.google.com/drive/folders/1hOb0TQtz7k5AzzPWUUU-lRHTE9ZS_G2M?usp=sharing
Download this preprocessed IXI dataset, and insert it into the 'data' folder.

## How to run

Train model with default configuration

```bash
# train on CPU
python misalign/train.py trainer=cpu

# train on GPU
python misalign/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python misalign/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python misalign/train.py trainer.max_epochs=20 data.batch_size=64
```

Running script is in scripts folder
```bash
bash scripts/run_misalign.sh 0 0 # run 4 benchmarks for misalign with 0, 0
```

for example:
```bash
python misalign/train_cgan.py data.misalign_x=6 task_name="CycleGAN" tags=["xy60"]  
# insert misalignment in x direction by stdev of 6 pixels, with task_name and tags (for saving)

python misalign/train_pgan.py data.misalign_x=4 task_name="PixelGAN" tags=["xy40"]  # Same for pGAN
```


## References

```cite
1. S. U. Dar, M. Yurt, L. Karacan, A. Erdem, E. Erdem and T. Çukur, "Image Synthesis in Multi-Contrast MRI With Conditional Generative Adversarial Networks," in IEEE Transactions on Medical Imaging, vol. 38, no. 10, pp. 2375-2388, Oct. 2019, doi: 10.1109/TMI.2019.2901750.
```
2. IXI Dataset (Normal): http://brain-development.org/ixi-dataset
3. BRATS dataset (Abnormalizty): https://sites.google.com/site/braintumorsegmentation/home/brats2015
