# Deep Deformable 3D Caricature with Learned Shape Control (DD3C)

[[Paper](https://dl.acm.org/doi/abs/10.1145/3528233.3530748)] [[Project page](https://ycjungsubhuman.github.io/DeepDeformable3DCaricatures)] [[ArXiv](http://arxiv.org/abs/2207.14593)] [[Additional result gallery](https://ycjungsubhuman.github.io/DeepDeformable3DCaricatures/gallery)] [[Video](https://youtu.be/WLMPEaK6E4M)]

![teaser](./imgs/teaser.jpg)

📝 This repository contains the official PyTorch implementation of the following paper:

> **[Deep Deformable 3D Caricature with Learned Shape Control](https://dl.acm.org/doi/abs/10.1145/3528233.3530748)**<br>
> Yucheol Jung, Wonjong Jang, Soongjin Kim, Jiaolong Yang, Xin Tong, Seungyong Lee, SIGGRAPH 2022

## Overview
<div align="center">
  
![method](./imgs/overview.jpg)

</div>

<details>
<summary><b>Explanation</b></summary>
<div markdown="1">
We build a data-driven toolkit for handling 3D caricature deformations. Our deformable model provides a nice editing space for 3D caricatures, supporting label-based semantic editing and point-handle-based deformation. To achieve the goal, we propose an MLP-based framework for building a deformable surface model. We adopt hyper-network architecture to model the latent space of highly complex 3D caricature shapes. Given a latent code, a SIREN MLP is generated. The SIREN MLP provides a mapping from a 3D coordinate on a fixed template mesh to a 3D displacement that is applied to the point. Once the model is trained, the learned mapping from a latent code to a 3D shape is used for various shape control.
</div>
</details>


# Environment

The code was tested on an Arch Linux desktop with a NVIDIA TITAN Xp GPU.

## Setup

Clone this repository using `--recursive` option.
```bash
git clone --recursive <URL>
```
If clonded already
```bash
git submodule update --init
```

Use provided docker image in [docker/Dockerfile](docker/Dockerfile).
```bash
cd docker
docker build -t dd3c .
cd ..
docker run --gpus all -it --rm -v $PWD:/workspace dd3c /bin/bash
cd /workspace
# Run your code
```
or build an equivalent environment yourself (Read Dockerfile and requirements.txt)


# Executables

## Download pre-trained models

You are required to download the models before running the executables. Select one of the mirrors to download the models. If one mirror does not work, try the other ones.

* [Mirror 1](https://postechackr-my.sharepoint.com/:u:/g/personal/ycjung_postech_ac_kr/EYrq_pQzvDdKiQcCD5uGiS8BgiKUCxnVbDPssBv-f9EfMw)
* [Mirror 2](https://1drv.ms/u/s!AuGv4oQ7PodugbVO3VcYTIHz2RDNOg?e=cCsjzO)
* [Mirror 3](https://drive.google.com/file/d/1sqHP8aNz23t3NZm72WBnnCX6COqxNONL/view?usp=sharing)

Extract the tar.zst archive and place `logs` folder on the same level as `README.md`.

## Fitting the 3D caricature model to 2D landmarks (68 landmark annotation)
Fit the model to 68 2D landmarks (68 landmarks used in DLIB face landmark detection).
By default the code runs fitting on the 50 test examples in data provided by the authors of *Wu et al. Alive Caricature from 2D to 3D. CVPR 2018.* (Included as submodule in `dataset/Caricature-Data`).
```shell
python run_fitting_2d_68.py --config ./configs/eval/deepdeformable_68.yml
```
You may test on different data by specifying different `dir_caricature_data` in `configs/eval/deepdeformable.yml`.

## Fitting the 3D caricature model to 2D landmarks (81 landmark annotation)
This code does the the same fitting as the above one, but uses a different landmark annoataion. The annotation adds 13 additional landmarks for forehead, ears, and cheekbones. We recommend using this annotation for caricature in frontal view to guarantee good fitting around forehead and ears.

Visual demonstration of the 81 landmarks is in ([imgs/landmarks_81.png](imgs/landmarks_81.png)). The annotation itself is contained in `staticdata/rawlandmarks.py:CaricShop3D.LANDMARKS` as vertex indices. The first 68 landmarks are the same as the 68 annotation.

```shell
python run_fitting_2d_81.py --config ./configs/eval/deepdeformable_81.yml
```

## Semantic-label-based 3D caricature editing
Given a 3D caricature (In our demo, 3D caricatures from trained latent codes), edit the shape using semantic labels.

```bash
python latent_manipulation_interfacegan.py --config ./configs/eval/edit_Smile.yml
```

The default behaviour is to adjust "Smile" attribute of the fittings. To change the semantic label, change `attr_index` option in the config yml file. The labels corresponding to each index are listed in `attr_list.txt`.


## Point-handle-based 3D caricature editing
PREREQUISITE: Run this code after running `run_fitting_2d_68.py`.

Given a 3D caricature (In our demo, a 3D caricature generated with 2D landmark fitting), edit the shape using landmark point handles.

To edit the faces from 68-landmark fittings, run
```bash
python latent_manipulation_pointhandle.py --config ./configs/eval/point.yml
```

Current implementation runs a pre-defined set of editings. To change the editing behaviour, refer to `latent_manipulation_pointhandle.py:L123:L149` and change them to fit your application.

## Automatic caricaturization

Automatically exaggerate regular 3D face using a model trained both on 3DCaricShop and FaceWarehouse.

```bash
python latent_manipulation_caricature.py --config ./configs/eval/deepdeformable_fw_caricature.yml
```

# Contact
📫 You can contact us via email: [ycjung@postech.ac.kr](mailto:ycjung@postech.ac.kr) or [wonjong@postech.ac.kr](mailto:wonjong@postech.ac.kr)


# License
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms require a license from the Pohang University of Science and Technology.


# Citation


If you find our material useful for your work, please consider citing our paper:

```bibtex
@inproceedings{jung2022deepdeformable,
author = {Jung, Yucheol and Jang, Wonjong and Kim, Soongjin and Yang, Jiaolong and Tong, Xin and Lee, Seungyong},
title = {Deep Deformable 3D Caricatures with Learned Shape Control},
year = {2022},
isbn = {9781450393379},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3528233.3530748},
doi = {10.1145/3528233.3530748},
booktitle = {ACM SIGGRAPH 2022 Conference Proceedings},
articleno = {59},
numpages = {9},
keywords = {3D face deformation, Deformable model, Semantic 3D face control, Auto-decoder, Parametric model, 3D face model},
location = {Vancouver, BC, Canada},
series = {SIGGRAPH '22}
}
```

# Credits
This implementation builds upon [DIF-Net](https://github.com/microsoft/DIF-Net) and [InterFaceGAN](https://github.com/genforce/interfacegan). We thank the authors for sharing the code for the work publicly.

