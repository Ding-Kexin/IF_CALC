# CALC
The repository contains the implementations for "**Coupled Adversarial Learning for Fusion Classification of Hyperspectral and LiDAR Data**". You can find [the PDF of this paper](https://www.sciencedirect.com/science/article/pii/S156625352200269X)
****
![CALC](https://github.com/Ding-Kexin/CALC/blob/main/figure/CALC.jpg)
****
# Datasets
[MUUFL](https://github.com/GatorSense/MUUFLGulfport/)
[Houston2013](http://www.grss-ieee.org/community/technical-committees/data-fusion/2013-ieee-grss-data-fusion-contest/)
****
# Citation
>@article{LU2023118,
title = {Coupled adversarial learning for fusion classification of hyperspectral and LiDAR data},
journal = {Information Fusion},
volume = {93},
pages = {118-131},
year = {2023},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2022.12.020},
url = {https://www.sciencedirect.com/science/article/pii/S156625352200269X},
author = {Ting Lu and Kexin Ding and Wei Fu and Shutao Li and Anjing Guo},
keywords = {Hyperspectral image, Light detection and ranging, Multimodal data classification, Adversarial learning, Feature fusion},
abstract = {Hyperspectral image (HSI) provides rich spectral–spatial information and the light detection and ranging (LiDAR) data reflect the elevation information, which can be jointly exploited for better land-cover classification. However, due to different imaging mechanisms, HSI and LiDAR data always present significant image difference, current pixel-wise feature fusion classification methods relying on concatenation or weighted fusion are not effective. To achieve accurate classification result, it is important to extract and fuse similar high-order semantic information and complementary discriminative information contained in multimodal data. In this paper, we propose a novel coupled adversarial learning based classification (CALC) method for fusion classification of HSI and LiDAR data. In specific, a coupled adversarial feature learning (CAFL) sub-network is first trained, to effectively learn the high-order semantic features from HSI and LiDAR data in an unsupervised manner. On one hand, the proposed CAFL sub-network establishes an adversarial game between dual generators and discriminators, so that the learnt features can preserve detail information in HSI and LiDAR data, respectively. On the other hand, by designing weight-sharing and linear fusion structure in the dual generators, we can simultaneously extract similar high-order semantic information and modal-specific complementary information. Meanwhile, a supervised multi-level feature fusion classification (MFFC) sub-network is trained, to further improve the classification performance via adaptive probability fusion strategy. In brief, the low-level, mid-level and high-level features learnt by the CAFL sub-network lead to multiple class estimation probabilities, which are then adaptively combined to generate a final accurate classification result. Both the CAFL and MFFC sub-networks are collaboratively trained by optimizing a designed joint loss function, which consists of unsupervised adversarial loss and supervised classification loss. Overall, by optimizing the joint loss function, the proposed CALC network is pushed to learn highly discriminative fusion features from multimodal data, leading to higher classification accuracies. Extensive experiments on three well-known HSI and LiDAR data sets demonstrate the superior classification performance by the proposed CALC method than several state-of-the-art methods. The source code of the proposed method will be made publicly available at https://github.com/Ding-Kexin/CALC.}
}
