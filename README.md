# TS-NeRF

# Installation
The implementation of the code's speed-up is based on the [instant-NGP](https://github.com/kwea123/ngp_pl) architecture.

## Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 6GB (Tested with RTX 2080 Ti), CUDA 11.3 (might work with older version)
* 32GB RAM (in order to load full size images)

## Software

* Clone this repo by `git clone https://github.com/liam6699/TS-NeRF.git`
* Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n ngp_pl python=3.8` to create a conda environment and activate it by `conda activate ngp_pl`)
* Python libraries
    * Install pytorch by `pip install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
    * Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) (pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Upgrade `pip` to >= 22.1 and run `pip install models/csrc/` (please run this each time you `pull` the code)

# Datasets, Pre-trained Model and Checkpoint
Run the following shell command to automatically download and install the datasets, pre-trained model and checkpoint for first stage.
```
bash ./download_data.sh
```
In addition, for compatibility with general hardware configurations, it is recommended that dataset archive sizes be kept within 512*512 pixels.


# Reproduction of the results
Run the following shell command to get the reproduced results (Quickstart):
```
cd TS-NeRF

python train.py --root_dir data/trex --exp_name trex__style --dataset_name colmap --stage second_stage --weight_path ckpts/last.ckpt --style_target "Pixar 3D style" --num_epochs 1 
```

It will train the `trex` scene for 30k steps (each step with 8192 rays), and perform one testing at the end. The reproduction of results  will be shown in `./results/colmap/trex__style`.

More options can be found in [opt.py](opt.py).



# Acknowledgments

Our code is based on [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://github.com/kwea123/ngp_pl).  
The implementation of the Nearest neighbor vector searcher are based on [High-Resolution Image Synthesis with Latent Diffusion Models](https://github.com/CompVis/latent-diffusion.git).  
The implementation of Consistency metric(Temporal Warping Error) is derived from [Learning Blind Video Temporal Consistency](https://github.com/phoenix104104/fast_blind_video_consistency).


