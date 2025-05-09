# VocV1(VogenVoc) Breathiness Generator Model - Training Script

This repository contains the training script for the breathiness generator model of the VocV1(VogenVoc) vocoder. It has been tested to work on **PyTorch 2.5** with **Python 3.12**.

Refer to the [VogenSVS repository](https://github.com/aqtq314/VogenSVS) for details on the vocoder itself as well as inference usage. Refer to the [paper](https://doi.org/10.1109/TASLP.2023.3321191) for more technical details.

This codebase is heavily based on the [UnivNet repository](https://github.com/maum-ai/univnet). Additionally, it uses code from [RMVPE](https://github.com/Dream-High/RMVPE) with [pretrained model](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx), serving as a better-performing F0 estimator than CREPE.


## Pip Dependencies

```plaintext
librosa==0.10.*
matplotlib==3.9.*
onnxruntime-gpu==1.18.*
pyworld==0.3.*
soundfile==0.12.*
tqdm==4.67.*
torch==2.5.*
```


## Pretrained ONNX Model Dependencies

- Download [pretrained RMVPE model weights](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx) and place it at `assets/[rmvpe]/rmvpe.onnx`

- Download [pretrained harmonics/breathiness separator model weights](https://github.com/aqtq314/VogenSVS/blob/main/vogensvs/assets/%5Bvocv1-hbdecomp%5D%5B220228-001211%5D%5Bimg73%5Dmodelv5c/model.onnx) and place it at `assets/[vocv1-hbdecomp][220228-001211][img73]modelv5c/model.onnx`


## Dataset Dependencies

The data preprocessing script is designed for the **OpenSinger dataset**. 

Download the raw dataset from [Multi-Singer](https://github.com/Multi-Singer/Multi-Singer.github.io) and extract the dataset to a directory of your choice.


## Usage of data preprocessing script `preproc.py`

- Wave files longer than 20 seconds or shorter than 1 second are ignored.
- Maximum VRAM usage per worker process is approximately **14 GB**.
- The script runs in about 53 minutes with 6 worker processes on 6 RTX 3090 GPUs.
- Avoid using the `CUDA_VISIBLE_DEVICES` environment variable. Use the `-d` switch instead.

```bash
python preproc.py -i [opensinger-dir]
python preproc.py -i [opensinger-dir] -d [gpu-indices]
```

e.g.:
```bash
python preproc.py -i ~/datasets/OpenSinger -d 2,3,4,5,6,7
```


## Usage of training script `train.py`

- VRAM usage is approximately **6.9 GB** per GPU device with default settings.
- Under default configuration, training time is approximately:
  - 23 minutes/epoch on RTX 3090
  - 12 minutes/epoch on RTX 4090
- No save/load functionality is implemented.
- Saved models are compiled with `torch.jit.script`.

**Commands**:
```bash
CUDA_VISIBLE_DEVICES=[gpu-indices] python train.py -n [session-name]
```

e.g.:
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python train.py -n 220801a-d1+WNorm-c16c12
```


## Acknowledgments

- [UnivNet Repository](https://github.com/maum-ai/univnet)
- [RMVPE Repository](https://github.com/Dream-High/RMVPE)
- [Pretrained RMVPE Weights](https://huggingface.co/lj1995/VoiceConversionWebUI)
- [OpenSinger Dataset](https://github.com/Multi-Singer/Multi-Singer.github.io)


