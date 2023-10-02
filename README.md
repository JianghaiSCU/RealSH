# [ICCV 2023]. Supervised Homography Learning with Realistic Dataset Generation. [[Paper]](https://arxiv.org/abs/2307.15353).
<h4 align="center">Hai Jiang<sup>1,2</sup>, Haipeng Li<sup>3,2</sup>, Haoqiang Fan<sup>2</sup>, Bing Zeng<sup>3</sup>, Songchen Han<sup>1</sup>, Shuaicheng Liu<sup>3,2</sup></center>
<h4 align="center">1.Sichuan University, 2.Megvii Technology 
<h4 align="center">3.University of Electronic Science and Technology of China</center></center>

## Presentation video:  
[[Youtube]]() and [[Bilibili]]()
## Pipeline
![](https://github.com/JianghaiSCU/RealHomo/blob/main/Figs/Pipeline.jpg)
## Dependencies
```
pip install -r requirements.txt
````

## Download the raw CA-unsup dataset
Please refer to [Content-Aware Unsupervised Deep Homography Estimation (CAHomo)](https://github.com/JirongZhang/DeepHomography).

- Original video data download links: [[GoogleDriver]](https://drive.google.com/file/d/19d2ylBUPcMQBb_MNBBGl9rCAS7SU-oGm/view?usp=sharing), [[BaiduYun]](https://pan.baidu.com/s/1Dkmz4MEzMtBx-T7nG0ORqA) (key:gvor)

- Unzip the data and Run "video2img.py" to save the images to the directory "./Homo_data/img"
```
Be sure to scale the image to (640, 360) since the point coordinate system is based on the (640, 360).
e.g. img = cv2.imresize(img, (640, 360))
```
- Using the images in "train.txt" and "test.txt" for training and evaluation.

- The manually labeled evaluation files can be download from [[GoogleDriver]](https://drive.google.com/file/d/16itODlnviqTRA_nSXXsg9s9EBWCir4gZ/view?usp=sharing), [[BaiduYun]](https://pan.baidu.com/s/1A0pyX_bHkKn_qiDCZF7S3Q) (key:hhnc)

## Download the dominant plane masks for image generation
- Download links: [[GoogleDriver]](https://drive.google.com/file/d/1cPdh08C-7zYpBtfgnhc2qmKgHR0UO8o-/view?usp=sharing), [[BaiduYun]](https://pan.baidu.com/s/1mSAB8kIczj5AliqlurcOTg) (key:j1zw)

- Unzip the masks to the directory "./Homo_data/mask"

## Pre-trained model

| model    | RE | LT | LL | SF | LF | Avg | Model |
| --------- | ----------- | ------------ |------------ |------------ |------------ |------------ |------------ |
| Pre-trained | 0.22 | 0.35 | 0.44 | 0.42 | 0.29 | 0.34 |[[Google]](https://drive.google.com/file/d/1U_GmwFZBzV-mmFOj8BlWOwoxVD3lxaUq/view?usp=sharing) [[Baidu]](https://pan.baidu.com/s/1A8IaNEDcbuoDj0UTmKrqTw)(key:qqed)
## How to train?
You need to modify ```dataset/data_loader.py``` slightly for your environment, and then
```
python train.py --model_dir experiments/Base/ 
```
## How to test?
```
python evaluate.py --model_dir experiments/Base/ --restore_file Iter2_0.3445.pth
```

## Supplementary Material
- Download links: [[GoogleDriver]](https://drive.google.com/file/d/16itODlnviqTRA_nSXXsg9s9EBWCir4gZ/view?usp=sharing), [[BaiduYun]](https://pan.baidu.com/s/1LAoi8nn0vdVEPz4CxP4wtA) (key:gwry)

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@InProceedings{Jiang_2023_ICCV,
    author    = {Jiang, Hai and Li, Haipeng and Han, Songchen and Fan, Haoqiang and Zeng, Bing and Liu, Shuaicheng},
    title     = {Supervised Homography Learning with Realistic Dataset Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {9806-9815}
}
```
