# cycle-image-gan
Based on https://github.com/taoxugit/AttnGAN/tree/master/code

Paper https://arxiv.org/abs/2003.12137

## Spring 2019 CS 224U Project
* BERT encoder
* Cycle-GAN
* Image2Text encoder

## Download Data
1. Download AttnGAN preprocessed data and captions [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ)
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`

## Instructions
* pretrain STREAM
```
python pretrain_STREAM.py --cfg cfg/STREAM/bird.yaml --gpu 0
```
* train CycleGAN
```
python main.py --cfg cfg/bird_cycle.yaml --gpu 0
```
