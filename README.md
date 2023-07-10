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

# Supported Datasets

1.  NSVF data

Download preprocessed datasets (`Synthetic_NeRF`, `Synthetic_NSVF`, `BlendedMVS`, `TanksAndTemples`) from [NSVF](https://github.com/facebookresearch/NSVF#dataset). **Do not change the folder names** since there is some hard-coded fix in my dataloader.

2.  NeRF++ data

Download data from [here](https://github.com/Kai-46/nerfplusplus#data).

3.  Colmap data

For custom data, run `colmap` and get a folder `sparse/0` under which there are `cameras.bin`, `images.bin` and `points3D.bin`. The following data with colmap format are also supported:

  *  [nerf_llff_data](https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=sharing) 
  *  [mipnerf360 data](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)
  *  [HDR-NeRF data](https://drive.google.com/drive/folders/1OTDLLH8ydKX1DcaNpbQ46LlP0dKx6E-I). Additionally, download colmap pose estimation from [here](https://drive.google.com/file/d/1TXxgf_ZxNB4o67FVD_r0aBUIZVRgZYMX/view?usp=sharing) and extract to the same location.

4. RTMV data

Download data from [here](http://www.cs.umd.edu/~mmeshry/projects/rtmv/). To convert the hdr images into ldr images for training, run `python misc/prepare_rtmv.py <path/to/RTMV>`, it will create `images/` folder under each scene folder, and will use these images to train (and test).

After preparing the data as described above, it is recommended that the data be put into ./data. For example `./data/trex`.
# Pre-trained Model Preparation
* Download data from [checkpoints of the VGG](https://drive.google.com/drive/folders/1lwoYBeOGnz3pa4YFw3UeF6pKnmcYCaBC?usp=drive_link), then put `fc_encoder_iter_160000.pth` and `vgg_normalised.pth` into `./pretrained_StyleVAE`.
* Download data from [ArtBench data](https://drive.google.com/drive/folders/1gXg2yCvVMrGtUs-XIVY4IMri0y3oVCjU?usp=drive_link), then decompress `rdm.zip` and  put `rdm` into `./Latent_Diffusion/data`.
# Training and Testing
1. First Stage(Quickstart):
```
python train.py --root_dir <data/trex> --exp_name trex__style --dataset_name colmap --stage first_stage --num_epochs 5
```
2. Second Stage(Quickstart):
```
python train.py --root_dir <data/trex> --exp_name trex__style --dataset_name colmap --stage second_stage --weight_path <ckpts/colmap/trex__style/first_stage.ckpt> --style_target "Pixar 3D style" --num_epochs 1 
```

It will train the `Trex` scene for 30k steps (each step with 8192 rays), and perform one testing at the end. The training process should finish within about 5 minutes (saving testing image is slow, add `--no_save_test` to disable). The test results will be shown in `./results`.

More options can be found in [opt.py](opt.py).

For other public dataset training, please refer to the scripts under `benchmarking`.



# Acknowledgments

Our code is based on [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://github.com/kwea123/ngp_pl).  
The implementation of the Nearest neighbor vector searcher are based on [High-Resolution Image Synthesis with Latent Diffusion Models](https://github.com/CompVis/latent-diffusion.git).  
The implementation of Consistency metric(Temporal Warping Error) is derived from [Learning Blind Video Temporal Consistency](https://github.com/phoenix104104/fast_blind_video_consistency).


