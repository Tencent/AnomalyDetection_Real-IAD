# Real-IAD Dataset
Official experiment example of [Real-IAD](https://realiad4ad.github.io/Real-IAD) Dataset using [UniAD](README_uniad.md)

## 1. Preparation

### 1.1. Download the decompress the dataset
- Download jsons of Real-IAD dataset (named `realiad_jsons.zip`) and extract into `data/Real-IAD/`
- Download images (of resolution 1024 pixels) of Real-IAD dataset (one ZIP archive per object) and extract them into `data/Real-IAD/realiad_1024/`
- [Optional] Download images (original resolution) of Real-IAD dataset (one ZIP archive per object) and extract them into `data/Real-IAD/realiad_raw/` if you want to conduct experiments on the raw images

The Real-IAD dataset directory should be as follow: (`audiojack` is one of the 30 objects in Real-IAD)
```shell
data
└── Real-IAD
        ├── realiad_1024
        │   ├── audiojack
        │   │   │── *.jpg
        │   │   │── *.png
        │   │   ...
        │   ...
        ├── realiad_jsons
        │   ├── audiojack.json
        │   ...
        ├── realiad_jsons_sv
        │   ├── audiojack.json
        │   ...
        ├── realiad_jsons_fuiad_0.0
        │   ├── audiojack.json
        │   ...
        ├── realiad_jsons_fuiad_0.1
        │   ├── audiojack.json
        │   ...
        ├── realiad_jsons_fuiad_0.2
        │   ├── audiojack.json
        │   ...
        ├── realiad_jsons_fuiad_0.4
        │   ├── audiojack.json
        │   ...
        └── realiad_raw
            ├── audiojack
            │   │── *.jpg
            │   │── *.png
            │   ...
            ...
```

### 1.2. Setup environment
Setup `python` environments following `requirements.txt`. We have tested the code under the environment with packages of versions listed below:
```text
einops==0.4.1
scikit-learn==0.24.2
scipy==1.9.1
tabulate==0.8.10
timm==0.6.12
torch==1.13.1+cu117
torchvision==0.14.1+cu117
```
You may change them if you have to and should adjust the code accordingly.

## 2. Training
We provide config for Single-View/Multi-View UIAD and FUIAD, they are located under `experiments` directory as follow:
```shell
experiments
├── RealIAD-C1       # Single-View UIAD
├── RealIAD-fuad-n0  # FUIAD (NR=0.0)
├── RealIAD-fuad-n1  # FUIAD (NR=0.1)
├── RealIAD-fuad-n2  # FUIAD (NR=0.2)
├── RealIAD-fuad-n4  # FUIAD (NR=0.4)
├── RealIAD-full     # Multi-View UIAD
...
```
- Single-View UIAD:
  ```shell
  cd experiments/RealIAD-C1 && train_torch.sh 8 0,1,2,3,4,5,6,7
  # run locally with 8 GPUs
  ```
- Multi-View UIAD:
  ```shell
  cd experiments/RealIAD-full && train_torch.sh 8 0,1,2,3,4,5,6,7
  # run locally with 8 GPUs
  ```
- FUIAD:
  ```shell
  # under bash
  pushd experiments/RealIAD-fuad-n0 && train_torch.sh 8 0,1,2,3,4,5,6,7 && popd
  pushd experiments/RealIAD-fuad-n1 && train_torch.sh 8 0,1,2,3,4,5,6,7 && popd
  pushd experiments/RealIAD-fuad-n2 && train_torch.sh 8 0,1,2,3,4,5,6,7 && popd
  pushd experiments/RealIAD-fuad-n4 && train_torch.sh 8 0,1,2,3,4,5,6,7 && popd
  # run locally with 8 GPUs
  ```

- [Optional] Experiments on Images of Original Resolution

  To conduct experiments on images of original resolution, change the config value `dataset.image_reader.kwargs.image_dir` from `data/Real-IAD/realiad_1024` to `data/Real-IAD/realiad_raw` in config file `experiments/{your_setting}/config.yaml`

## 3. Evaluating
After training finished, ano-map of evaluation set is generated under `experiments/{your_setting}/checkpoints/` and store in `*.pkl` files, one file per object. Then use [ADEval](https://pypi.org/project/ADEval/) to evaluate the result.

- Install ADEval
  ```shell
  python3 -m pip install ADEval
  ```

- Execute the evaluate command

  Take Multi-View UIAD as an example:

  ```shell
  # calculate S-AUROC, I-AUROC and P-AUPRO for each object
  find experiments/RealAD-full/checkpoints/ | \
      grep pkl$ | sort | \
      xargs -n 1 python3 -m adeval --sample_key_pat "([a-zA-Z][a-zA-Z0-9_]*_[0-9]{4}_[A-Z][A-Z_]*[A-Z])_C[0-9]_"
  ```
  > Note: the argument `--sample_key_pat` is identical for all experiment settings of Real-IAD

## Acknowledgement
This repo is built on the top of Offical Implementation of [UniAD](https://github.com/zhiyuanyou/UniAD.git), which use some codes from repositories including [detr](https://github.com/facebookresearch/detr) and [efficientnet](https://github.com/lukemelas/EfficientNet-PyTorch). 

## Notice
The copyright notice pertaining to the Tencent code in this repo was previously in the name of "THL A29 Limited." That entity has now been de-registered. You should treat all previously distributed copies of the code as if the copyright notice was in the name of "Tencent".
