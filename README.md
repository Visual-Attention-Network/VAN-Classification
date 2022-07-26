# Visual Attention Network (VAN)  [paper pdf](https://arxiv.org/pdf/2202.09741.pdf)

This is a PyTorch implementation of **VAN** proposed by our paper "**Visual Attention Network**".

![Comparsion](https://github.com/Visual-Attention-Network/VAN-Classification/blob/main/images/Comparsion.png)

Figure 1: **Compare with different vision backbones on ImageNet-1K validation set.** 


## Citation:
```
@article{guo2022visual,
  title={Visual Attention Network},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Liu, Zheng-Ning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2202.09741},
  year={2022}
}
```

## News:

### 2022.02.22 Release paper on ArXiv and code on github.

### 2022.02.25 Supported by [Jimm](https://github.com/Jittor-Image-Models/Jittor-Image-Models)

### 2022.03.15 Supported by [Hugging Face](https://github.com/huggingface/transformers).

### 2022.04 Supported by [PaddleCls](https://github.com/PaddlePaddle/PaddleClas).

### 2022.05 Supported by [OpenMMLab](https://github.com/open-mmlab/mmclassification).

### For More Code, please refer to [Paper with code](https://paperswithcode.com/paper/visual-attention-network).

### 2022.07.08 Update paper on ArXiv. (ImageNet-22K results, SOTA for panoptic segmentation (58.2 PQ). Segmentation models are [available](https://github.com/Visual-Attention-Network/VAN-Classification). 
Classification models see [Here](https://github.com/LGYoung/VAN-Classification). We are working on it. 


### Abstract: 

While originally designed for natural language processing (NLP) tasks, the self-attention mechanism has recently taken various computer vision areas by storm. However, the 2D nature of images brings three challenges for applying self-attention in computer vision. (1) Treating images as 1D sequences neglects their 2D structures. (2) The quadratic complexity is too expensive for high-resolution images. (3) It only captures spatial adaptability but ignores channel adaptability. In this paper, we propose a novel large kernel attention (LKA) module to enable self-adaptive and long-range correlations in self-attention while avoiding the above issues. We further introduce a novel neural network based on LKA, namely Visual Attention Network (VAN). While extremely simple and efficient, VAN outperforms the state-of-the-art vision transformers (ViTs) and convolutional neural networks (CNNs) with a large margin in extensive experiments, including image classification, object detection, semantic segmentation, instance segmentation, etc.

![Decomposition](https://github.com/Visual-Attention-Network/VAN-Classification/blob/main/images/decomposition.png)

Figure 2: Decomposition diagram of large-kernel convolution. A standard convolution can be decomposed into three parts: a depth-wise convolution (DW-Conv), a depth-wise dilation convolution (DW-D-Conv) and a 1×1 convolution (1×1 Conv). 



![LKA](https://github.com/Visual-Attention-Network/VAN-Classification/blob/main/images/LKA.png)

Figure 3: The structure of different modules: (a) the proposed Large Kernel Attention (LKA); (b) non-attention module; (c) the self-attention module (d) a stage of our Visual Attention Network (VAN). CFF means convolutional feed-forward network. The difference between (a) and (b) is the element-wise multiply. It is worth noting that (c) is designed for 1D sequences. .

## Image Classification

Data prepare: ImageNet with the following folder structure.

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```



### 2. VAN Models (IN-1K)

| Model     | #Params(M) | GFLOPs | Top1 Acc(%) |                           Download                           |
| :-------- | :--------: | :----: | :---------: | :----------------------------------------------------------: |
| VAN-B0  |    4.1     |  0.9   |    75.4     |[Google Drive](https://drive.google.com/file/d/1KYoIe1Zl3ZaPCwRuvnpkLyOEK04JKemu/view?usp=sharing), [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/aada2242a16245d6a561/?dl=1), [Hugging Face 🤗](https://huggingface.co/Visual-Attention-Network/VAN-Tiny-original) |
| VAN-B1 |    13.9    |  2.5   |    81.1     |[Google Drive](https://drive.google.com/file/d/1LFsJHwxAs1TcXAjJ28G86_jwYwV8DzuG/view?usp=sharing), [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/dd3eb73692f74a2499c9/?dl=1), [Hugging Face 🤗](https://huggingface.co/Visual-Attention-Network/VAN-Small-original) |
| VAN-B2  |    26.6    |  5.0   |    82.8     |[Google Drive](https://drive.google.com/file/d/1qApsgXCbngNYOji2UzJsfeEsPOu6dBo3/view?usp=sharing), [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/58e7acceaf334ecdba89/?dl=1),[Hugging Face 🤗](https://huggingface.co/Visual-Attention-Network/VAN-Base-original), |
| VAN-B3 |    44.8    |  9.0   |    83.9     |[Google Drive](https://drive.google.com/file/d/10n6u-W3IrqiCD-7wkotejV_1XiS9kuWF/view?usp=sharing), [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/0201745f6920482490a0/?dl=1), [Hugging Face 🤗](https://huggingface.co/Visual-Attention-Network/VAN-Large-original) |
| VAN-B4  |    TODO    |  TODO  |    TODO     |                             TODO                             |




### 3.Requirement

```
1. Pytorch >= 1.7
2. timm == 0.4.12
```

### 4. Train 

We use 8 GPUs for training by default.  Run command (It has been writen in train.sh):

```bash
MODEL=van_tiny # van_{tiny, small, base, large}
DROP_PATH=0.1 # drop path rates [0.1, 0.1, 0.1, 0.2] for [tiny, small, base, large]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash distributed_train.sh 8 /path/to/imagenet \
	  --model $MODEL -b 128 --lr 1e-3 --drop-path $DROP_PATH
```



### 5. Validate

Run command (It has been writen in eval.sh) as:


```bash
MODEL=van_tiny # van_{tiny, small, base, large}
python3 validate.py /path/to/imagenet --model $MODEL \
  --checkpoint /path/to/model -b 128

```

## 6.Acknowledgment

Our implementation is mainly based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and [PoolFormer](https://github.com/sail-sg/poolformer). Thanks for their authors. 


## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
