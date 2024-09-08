# MUSE: Mamba is Efficient Multi-scale Learner for Text-video Retrieval
[![Paper](https://img.shields.io/badge/Paper-arxiv.2408.10575-FF6B6B.svg)](https://www.arxiv.org/pdf/2408.10575)

This is an official implementation of MUSE built on model [CLIP4clip](https://github.com/ArrowLuo/CLIP4Clip).

![MUSE](./pictures/MUSE_1.png)

## Requirement
```sh
# Pytorch version
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# From CLIP4clip
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas
```
Install Causal_conv1d and Mamba_ssm following [Vim](https://github.com/doodleima/vision_mamba).

## Data Preparing

**For MSRVTT**

The official data and video links can be found in [link](http://ms-multimedia-challenge.com/2017/dataset). 

For the convenience, you can also download the splits and captions by,
```sh
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
```

Besides, the raw videos can be found in [sharing](https://github.com/m-bain/frozen-in-time#-finetuning-benchmarks-msr-vtt) from *Frozen️ in Time*, i.e.,
```sh
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```

### Train MSR-VTT

```sh
bash train_msrvtt.sh
```

# Acknowledgments
Our code is based on [CLIP](https://github.com/openai/CLIP), [CLIP4clip](https://github.com/ArrowLuo/CLIP4Clip) and [Vim](https://github.com/doodleima/vision_mamba).
