# From Global to Local: Multi-scale Out-of-distribution Detection
## Abstract
Out-of-distribution (OOD) detection aims to detect “unknown” data whose labels have not been seen during the in-distribution (ID) training process. Recent progress in representation learning gives rise to distance-based OOD detection that recognizes testing examples as ID/OOD according to their relatively distances to the training data of ID classes. Previous approaches calculate pairwise distances relying only on global image representations, which can be sub-optimal as the inevitable background clutter and intra-class variation may drive imagelevel representations from the same ID class far apart in a given representation space. In this work, we tackle this challenge by proposing Multi-scale OOD DEtection (MODE), a first framework leveraging both global visual information and local region details of images to maximally benefit OOD detection. Specifically, we first find that existing models pretrained by offthe-shelf cross-entropy or contrastive losses are incompetent to
capture valuable local representations for MODE, due to the scale-discrepancy between the ID training and OOD detection processes. To mitigate this issue and encourage locally discriminative representations in ID training, we propose Attention-based Local PropAgation (ALPA), a trainable objective that exploits a cross-attention mechanism to align and highlight the local regions of the target objects for pairwise examples. In test-time OOD detection, a Cross-Scale Decision (CSD) function is further devised on the most discriminative multi-scale representations to separate ID-OOD data more faithfully. We demonstrate the effectiveness and flexibility of MODE on several benchmarks. Remarkably, MODE outperforms the previous state-of-the-art up to 19.24% in FPR, 2.77% in AUROC. 

## Motivation 
Motivation of exploring local, region-level representations to enhance the distance calculation between pairwise examples: the inevitable background
clutter and intra-class variation may drive global, image-level representations from the same ID class far apart in a given representation space. For the
first time, we take advantage of both global visual information and local region details of images to maximally benefit OOD detection.
<p align="center">
  <img src="./figures/f1.png" style="width:50%">
</p>
 
## Overview of training-time Attention-based Local Propagation (ALPA)
During ID training, we develop ALPA, an end-to-end, plug-and-play, and cross-attention based learning objective tailored for encouraging locally discriminative representations for MODE.
<p align="center">
  <img src="./figures/f2.png" style="width:80%">
</p>

## Overview of test-time OOD detection with Cross-scale Decision (CSD) 
During test-time OOD detection, we devise CSD, a simple, effective and multi-scale representations based ID-OOD decision function for MODE.
<p align="center">
  <img src="./figures/f3.png" style="width:50%">
</p>


## Contributions
- We propose MODE, a first framework that takes advantage of multi-scale (i.e., both global and local) representations for OOD detection.

- In ID training, we develop ALPA, an end-to-end, plug-and-play, and cross-attention based learning objective tailored for encouraging locally discriminative representations for MODE.

- In test-time OOD detection, we devise CSD, a simple, effective and multi-scale representations based ID-OOD decision function for MODE.

- Extensive experiments on several benchmark datasets demonstrate the effectiveness and flexibility of MODE. Remarkably, MODE outperforms the previous state-ofthe-art up to 19.24% in FPR, 2.77% in AUROC.


## Strong Performance
- CIFAR-10 (ID) with ResNet-18
<p align="center">
  <img src="./figures/f4.png" style="width:80%">
</p>

- CIFAR-100 (ID) with ResNet-34
<p align="center">
  <img src="./figures/f5.png" style="width:80%">
</p>

- ImageNet (ID) with ResNet-50
<p align="center">
  <img src="./figures/f6.png" style="width:80%">
</p>

## Visualization
-  Visualization with tSNE
<p align="center">
  <img src="./figures/f7.png" style="width:70%">
</p>

- Visualization analysis on k-nearest neighbors
<p align="center">
  <img src="./figures/f8.png" style="width:80%">
</p>

## Dependencies
* [PyTorch](https://pytorch.org/)
* [scipy](https://github.com/scipy/scipy)
* [numpy](http://www.numpy.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [faiss](https://github.com/facebookresearch/faiss)
* [pytorch-vit](https://github.com/lukemelas/PyTorch-Pretrained-ViT)
* [ylib](https://github.com/sunyiyou/ylib)

## Datasets
#### In-distribution dataset
Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in
`./datasets/imagenet/train` and  `./datasets/imagenet/val`, respectively.

#### Out-of-distribution dataset

We have curated 4 OOD datasets from 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf), 
and de-duplicated concepts overlapped with ImageNet-1k.

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset,
which can be download via the following links:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, we use the entire dataset, which can be downloaded from their
[original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Please put all downloaded OOD datasets into `./datasets/ood_data`.

### 2. Dataset Preparation for CIFAR Experiment 

#### In-distribution dataset

The downloading process will start immediately upon running. 

#### Out-of-distribution dataset


We provide links and instructions to download each dataset:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/ood_data/svhn`. Then run `python select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/ood_data/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/ood_data/places365/test_subset`. We randomly sample 10,000 images from the original test dataset. 
* [LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/ood_data/LSUN`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/ood_data/iSUN`.
* [LSUN_fix](https://drive.google.com/file/d/1KVWj9xpHfVwGcErH5huVujk9snhEGOxE/view?usp=sharing): download it and place it in the folder of `datasets/ood_data/LSUN_fix`.
* [ImageNet_fix](https://drive.google.com/file/d/1sO_-noq10mmziB1ECDyNhD5T4u5otyKA/view?usp=sharing): download it and place it in the folder of `datasets/ood_data/ImageNet_fix`.
* [ImageNet_resize](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz): download it and place it in the folder of `datasets/ood_data/Imagenet_resize`.

[//]: # (For example, run the following commands in the **root** directory to download **LSUN**:)

[//]: # (```)

[//]: # (cd datasets/ood_datasets)

[//]: # (wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)

[//]: # (tar -xvzf LSUN.tar.gz)

[//]: # (```)


## Pretrained Models
- [URL (RN-18)](https://github.com/VICO-UoE/URL)

- [DINO (ViT-S)](https://github.com/facebookresearch/dino)

- [MoCo-v2 (RN-50)](https://github.com/facebookresearch/moco)

- [CLIP (RN-50)](https://github.com/OpenAI/CLIP)

- [Deit (ViT-S)](https://github.com/facebookresearch/deit)

- [Swin Transformer (Tiny)](https://github.com/microsoft/Swin-Transformer)

## Model pretraining w/ or w/o ALPA
**Standard cross-entropy (CE)**
```
python ./ALPA/main_ce.py --batch_size 512 \
  --learning_rate 0.5 \
  --cosine --syncBN \
```

**Standard contrastive loss (CL)**
```
python ./ALPA/main_supcon.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine
```

**ID training  w/ ALPA-train**
```
python ./ALPA/main_supcon.py --batch_size 128 \
--learning_rate 0.1 \
--temp 0.1 \
--cosine \
--trial 0 \
--dataset cifar100 \
--model resnet34 \
--alpa_train
```

**ID training w/ ALPA-finetune**
```
python ./ALPA/main_supcon.py --batch_size 128 \
--learning_rate 0.1 \
--temp 0.1 \
--cosine \
--trial 0 \
--dataset cifar100 \
--model resnet34 \
--alpa_finetune
```

## Test-time OOD detection with CSD
Specify the path to the pretrained model, and execute the following command.
```
Run ./demo_cifar.sh.
```

## References
<div style="text-align:justify; font-size:80%">
    <p>
        [1] Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku Evci, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol, Hugo Larochelle; <a href="https://arxiv.org/abs/1903.03096">Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples</a>; ICLR 2020.
    </p>
    <p>
        [2] Li, Wei-Hong and Liu, Xialei and Bilen, Hakan; <a href="https://arxiv.org/abs/2107.00358">Cross-domain Few-shot Learning with Task-specific Adapters</a>; CVPR 2022.
    </p>
    <p>
        [3] Xu, Chengming and Yang, Siqian and Wang, Yabiao and Wang, Zhanxiong and Fu, Yanwei and Xue, Xiangyang; <a href="https://openreview.net/pdf?id=n3qLz4eL1l">Exploring Efficient Few-shot Adaptation for Vision Transformers</a>; Transactions on Machine Learning Research 2022.
    </p>
    <p>
        [4] Liang, Kevin J and Rangrej, Samrudhdhi B and Petrovic, Vladan and Hassner, Tal; <a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Liang_Few-Shot_Learning_With_Noisy_Labels_CVPR_2022_paper.pdf">Few-shot learning with noisy labels</a>; CVPR 2022.
    </p>
    <p>
        [5] Chen, Pengguang, Shu Liu, and Jiaya Jia; <a href="http://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Jigsaw_Clustering_for_Unsupervised_Visual_Representation_Learning_CVPR_2021_paper.pdf">Jigsaw clustering for unsupervised visual representation learning</a>; CVPR 2020.
    </p>

</div>


## Acknowledge
We thank authors of [Meta-Dataset](https://github.com/google-research/meta-dataset), [URL/TSA](https://github.com/VICO-UoE/URL), [eTT](https://github.com/loadder/eTT_TMLR2022), [JigsawClustering](https://github.com/dvlab-research/JigsawClustering) for their source code. 

