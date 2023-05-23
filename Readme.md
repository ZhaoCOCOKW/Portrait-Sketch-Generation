# Portrait Sketch Generation Based On Pix2pix Network

## Introduction

​	This is the project for portrait sketch generation project in CS3511. 

​	Portrait sketch generation aims to generate high-quality sketches from simple hand-drawn lines, which requires a generation network from tensors to tensors. In this project, we implement a traditional generative adversarial network model: **pix2pix**  network, and applies it to train and generate our sketch pieces. Then we add some optimization methods like **Conditional Training**  and **Noise Removal** to make the generated sketches more realistic with higher scores.

​	Here is an example for generation:

![](.\Readme_pics\line.jpg)![](.\Readme_pics\sketch.jpg)

​	Our network is OK in large feature generation like nose and mouth, but is not good in features like eyes and hairs. However, this project would be a stepping stone for our further works.



## Try our network

* Environments: `python 3.10; torch2.0.0+cu117 `

​	Download this repository to your own computer, then use `python main.py` to run our network. The generated sketches will be saved in directory `Result`.

​	To tune hyper parameters, you can open *main.py* and change them for what you need. These parameters contain:

`BATCHSIZE` size for one batch in training

`LEARNING_RATE` learning_rate for optimizer

`LAMBDA` weights for generators' L1 loss

`EPOCHS` training epochs

`TIMER` and `LOSS_THRESHOLD` parameters for conditional training, please read our report for details.

​	Hope you like our network...though it's seems a little bad. Thanks!   \*w\*~