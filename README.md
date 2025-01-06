# MTCAE-DFER: Multi-Task Cascaded Autoencoder for Dynamic Facial Expression Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multimae-der-multimodal-masked-autoencoder/emotion-recognition-on-ravdess)](https://paperswithcode.com/sota/emotion-recognition-on-ravdess?p=multimae-der-multimodal-masked-autoencoder)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multimae-der-multimodal-masked-autoencoder/video-emotion-recognition-on-crema-d)](https://paperswithcode.com/sota/video-emotion-recognition-on-crema-d?p=multimae-der-multimodal-masked-autoencoder)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multimae-der-multimodal-masked-autoencoder/multimodal-emotion-recognition-on-iemocap)](https://paperswithcode.com/sota/multimodal-emotion-recognition-on-iemocap?p=multimae-der-multimodal-masked-autoencoder)<br>

> [![HCPS link](https://img.shields.io/badge/FIU-HCPS-red)](https://hcps.fiu.edu/) [![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=MultiMAE-DER&color=red&logo=arxiv)](https://arxiv.org/abs/2412.18988) [![Citation link](https://img.shields.io/badge/Citation-BibTeX-red)](#citation)<br>
> [Peihao Xiang](https://scholar.google.com/citations?user=k--3fM4AAAAJ&hl=zh-CN&oi=ao), [Kaida Wu](https://ieeexplore.ieee.org/author/167739911238744), [Chaohao Lin](https://scholar.google.com/citations?hl=zh-CN&user=V3l7dAEAAAAJ), and [Ou Bai](https://scholar.google.com/citations?hl=zh-CN&user=S0j4DOoAAAAJ)<br>
> HCPS Laboratory, Department of Electrical and Computer Engineering, Florida International University<br>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Peihao-Xiang/MTCAE-DFER/blob/main/MTCAE-DFER_Fine-Tuning%20Code/MTCAE_DFER_Cascaded_ViTDecoder_Example.ipynb)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/NoahMartinezXiang/RAVDESS)

Official TensorFlow implementation and ViT Decoder Module codes for MTCAE-DFER: Multi-Task Cascaded Autoencoder for Dynamic Facial Expression Recognitionn.

Note: The .ipynb is just a simple example. In addition, the VideoMAE encoder model should be pre-trained using the MAE-DFER method, but this repository does not provide it.

## Overview

This paper expands the cascaded network branch of the autoencoder-based multi-task learning (MTL) framework for dynamic facial expression recognition, namely Multi-Task Cascaded Autoencoder for Dynamic Facial Expression Recognition (MTCAE-DFER). MTCAE-DFER builds a plug-and-play cascaded decoder module, which is based on the Vision Transformer (ViT) architecture and employs the decoder concept of Transformer to reconstruct the multi-head attention module. The decoder output from the previous task serves as the query (Q), representing local dynamic features, while the Video Masked Autoencoder (VideoMAE) shared encoder output acts as both the key (K) and value (V), representing global dynamic features. This setup facilitates interaction between global and local dynamic features across related tasks. Additionally, this proposal aims to alleviate overfitting of complex large model. We utilize autoencoder-based multi-task cascaded learning approach to explore the impact of dynamic face detection and dynamic face landmark on dynamic facial expression recognition, which enhances the model's generalization ability. After we conduct extensive ablation experiments and comparison with state-of-the-art (SOTA) methods on various public datasets for dynamic facial expression recognition, the robustness of the MTCAE-DFER model and the effectiveness of global-local dynamic feature interaction among related tasks have been proven.

<p align="center">
  <img src="Images/MTL_MTCL.png" width=50%
    class="center"><br>
  Illustration of the frameworks.
</p>

Illustration of the differences between the following four frameworks: (a) Autoencoder-Based Single-Task Learning Framework, (b) Autoencoder-Based Non-Fully Shared Multi-Task Learning Framework, (c) Autoencoder-Based Fully Shared Multi-Task Learning Framework and (d) Our Autoencoder-Based Multi-Task Cascaded Learning Framework.

<p align="center">
  <img src="Images/MTSAE-DFER.png" width=70%
    class="center"><br>
  MTCAE-DFER Model Structure.
</p>

## Implementation details

<p align="center">
  <img src="Images/MTCAE-DFER Flowchart.png" width=50%> <br>
  The architecture of MultiMAE-DER.
</p>

## Main Results

### RAVDESS

![Result_on_RAVDESS](images/Result_on_RAVDESS.png)

### CREMA-D

![Result_on_CREMA-D](images/Result_on_CREMA-D.png)

### IEMOCAP

![Result_on_IEMOCAP](images/Result_on_IEMOCAP.png)

## Contact 

If you have any questions, please feel free to reach me out at pxian001@fiu.edu.

## Acknowledgments
This project is built upon [VideoMAE](https://github.com/innat/VideoMAE) and [MAE-DFER](https://github.com/sunlicai/MAE-DFER). Thanks for their great codebase.

In addition, this project is inspired by [MTFormer](https://github.com/xiaogang00/MTFormer) and [MNC](https://github.com/daijifeng001/MNC).

## License

This project is under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you find this repository helpful, please consider citing our work:

```BibTeX
@misc{xiang2024mtcaedfermultitaskcascadedautoencoder,
      title={MTCAE-DFER: Multi-Task Cascaded Autoencoder for Dynamic Facial Expression Recognition}, 
      author={Peihao Xiang and Kaida Wu and Chaohao Lin and Ou Bai},
      year={2024},
      eprint={2412.18988},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.18988}, 
}
```
