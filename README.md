# A Unified Efficient Pyramid Transformer for Semantic Segmentation [paper link](https://openaccess.thecvf.com/content/ICCV2021W/VSPW/papers/Zhu_A_Unified_Efficient_Pyramid_Transformer_for_Semantic_Segmentation_ICCVW_2021_paper.pdf)

## Installation

* Linux, CUDA>=10.0, GCC>=5.4
* Python>=3.7
* Create a conda environment:

```bash
    conda create -n unept python=3.7 pip
```

Then, activate the environment:
```bash
    conda activate unept
```
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

For example:
```
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
```

* Install [MMCV](https://mmcv.readthedocs.io/en/latest/), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/install.md), [timm](https://pypi.org/project/timm/)

```
pip install -r requirements.txt
```

* Install [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and compile the CUDA operators


## Data Preparation
Please following the code from [openseg](https://github.com/openseg-group/openseg.pytorch) to generate ground truth for boundary refinement. 

The data format should be like this.

### ADE20k

```
path/to/ADEChallengeData2016/
  images/
    training/
    validation/
  annotations/ 
    training/
    validation/
  dt_offset/
    training/
    validation/
```
### PASCAL-Context
```
path/to/PASCAL-Context/
  train/
    image/
    label/
    dt_offset/
  val/
    image/
    label/
    dt_offset/
```

## Usage 
### Training 
**The default is for multi-gpu, DistributedDataParallel training.**

```
python -m torch.distributed.launch --nproc_per_node=8 \ # specify gpu number
--master_port=29500  \
train.py  --launcher pytorch \
--config /path/to/config_file 
```

- specify the ```data_root``` in the config file;
- log dir will be created in ```./work_dirs```;
- download the [DeiT pretrained model](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth) and specify the ```pretrained``` path in the config file.


<!-- **3. I try to take 'H/8*W/8 x 256' as the vector size for decoder input and remain stride 32 for the backbone.**

In the config, ```num_queries=4096``` and ```dec_stride=8``` are specified.

It can run with single gpu (one sample on it). (```norm_cfg = dict(type='BN', requires_grad=True)```)
```
python train.py --config_file ./configs/resV1c50_32x_dec8x_tr2+2_512x512_adamW_step_640k_ade20k.py
```

But, it got the ```CUDA error: an illegal memory access was encountered``` when trained with multi-gpu (```norm_cfg = dict(type='SyncBN', requires_grad=True)```):

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 train.py  --launcher pytorch \
--config_file ./configs/resV1c50_32x_dec8x_tr2+2_512x512_adamW_step_640k_ade20k.py
``` -->


### Evaluation
<!-- 1. directly resize image and ground truth to 512x512 by bilinear interpolation and nearest interpolation, respectively. ('test_mode': 'direct_resize')

The default is for single-gpu evaluation with batch size being 1.
```
python evaluate.py --eval \
--config_file configs/resV1c50_segtr2_512x512_ade20k.yaml \
--checkpoint log/checkpoint_0131.pth.tar \
--data_root /path/to/ADEChallengeData2016
``` -->

```
# single-gpu testing
python test.py --checkpoint /path/to/checkpoint \
--config /path/to/config_file \
--eval mIoU \
[--out ${RESULT_FILE}] [--show] \
--aug-test \ # for multi-scale flip aug

# multi-gpu testing (4 gpus, 1 sample per gpu)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
test.py  --launcher pytorch --eval mIoU \
--config_file /path/to/config_file \
--checkpoint /path/to/checkpoint \
--aug-test \ # for multi-scale flip aug
```

## Results
We report results on validation sets.

| Backbone | Crop Size | Batch Size | Dataset | Lr schd | Mem(GB) | mIoU(ms+flip) | config |
| :------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | 
| Res-50 | 480x480 | 16 | ADE20K | 160K | 7.0G | 46.1 | [config](https://github.com/amazon-research/unified-ept/configs/res50_unept_ade20k.py) |
| DeiT | 480x480 | 16 | ADE20K | 160K | 8.5G | 50.5 | [config](https://github.com/amazon-research/unified-ept/configs/deit_unept_ade20k.py) |
| DeiT | 480x480 | 16 | PASCAL-Context | 160K | 8.5G | 55.2 | [config](https://github.com/amazon-research/unified-ept/configs/deit_unept_pcontext.py) |


## License

This project is licensed under the Apache-2.0 License.

## Citation

If you use this code and models for your research, please consider citing:

```
@article{zhu2021unified,
  title={A Unified Efficient Pyramid Transformer for Semantic Segmentation},
  author={Zhu, Fangrui and Zhu, Yi and Zhang, Li and Wu, Chongruo and Fu, Yanwei and Li, Mu},
  journal={arXiv preprint arXiv:2107.14209},
  year={2021}
}
```

## Acknowledgment

We thank the authors and contributors of [MMCV](https://mmcv.readthedocs.io/en/latest/), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/install.md), [timm](https://pypi.org/project/timm/) and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
<!-- ### Using Longformer

1. install longformer (when using tvm mode, the compiled file only supports CUDA 10.0 with pyTorch 1.2)
```
pip install git+https://github.com/allenai/longformer.git
```

2. reinstall mmcv and mmsegmentation
```
pip install mmcv==1.1.0
pip install mmsegmentation==0.5.0
``` -->

<!-- ### Evaluate on Cityscapes
1. on validation set.

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
test.py  --launcher pytorch  \
--aug-test \ # for multi-scale flip aug
--out results.pkl --eval mIoU cityscapes
```

2. on test set. save images for submitting to the server.

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
test.py  --launcher pytorch  \
--aug-test \ # for multi-scale flip aug
--format-only --eval-options "imgfile_prefix=./test_results"
```






  -->