# Federated Alignment (FedAli)
Tensorflow implementation of **Federated Alignment and the ALP layer**:

**FedAli: Personalized Federated Learning with Aligned Prototypes through Optimal Transport** [[Paper](https://arxiv.org/abs/2306.13735)]

*[Sannara Ek](https://scholar.google.com/citations?user=P1F8sQgAAAAJ&hl=en&oi=ao),[Kaile Wang](https://www4.comp.polyu.edu.hk/~labimcl/profile/kaile-wang.html), [François Portet](https://lig-membres.imag.fr/portet/home.php), [Philippe Lalanda](https://lig-membres.imag.fr/lalanda/), [Jiannong Cao](https://www4.comp.polyu.edu.hk/~csjcao/)*


<p align="center">
  <img width="80%" alt="Leave-One-Dataset-Out" src="Figs/LODO_Fig.png">
</p>

If our project is helpful for your research, please consider citing : 
``` 
@article{ek2024fedali,
  title={FedAli: Personalized Federated Learning with Aligned Prototypes through Optimal Transport},
  author={Ek, Sannara and Wang, Kaile and Portet, Fran{\c{c}}ois and Lalanda, Philippe and Cao, Jiannong},
  journal={arXiv preprint arXiv:2411.10595},
  year={2024}
}
```


## Table of Content
* [1. Updates](#1-Updates)
* [2. Installation](#2-installation)
  * [2.1 Dependencies](#21-dependencies)
  * [2.2 Data](#22-data)
* [3. Quick Start](#3-quick-start)
  * [3.1 Using the LODO partition](#31-Using-the-LODO-partition)
  * [3.2 Running Our Supervised Pretraining Pipeline](#32-Running-Our-Supervised-Pretraining-Pipeline)
  * [3.3 Loading and Using our Pre-Trained Models](#33-Loading-and-Using-our-Pre-Trained-Models)
* [4. Acknowledgement](#4-acknowledgement)

### 3.3 Loading and Using our Pre-Trained Models
## 1. Updates


***14/02/2025***
Initial commit: The Tensorflow Code of FedAli has been released.

## 2. Installation
### 2.1 Dependencies

This Tensorflow version of this code was implemented with Python 3.11, Tensorflow 2.15.1 and CUDA 12.2. Please refer to [the official installation](https://www.tensorflow.org/install). If CUDA 12.2 has been properly installed : 
```
pip3 install tensorflow==2.15.1
```

Another core library of our work is Hickle for data storage management. Please launch the following command to be able to run our data partitioning scripts: 
```
pip3 install hickle
```

To run our training and evaluation pipeline, additional dependencies are needed. Please launch the following command:

```
pip3 install -r requirements.txt
```

Our baseline experiments were conducted on a Debian GNU/Linux 10 (buster) machine with the following specs:

CPU : Intel(R) Xeon(R) CPU E5-2623 v4 @ 2.60GHz

GPU : Nvidia GeForce Titan Xp 12GB VRAM

Memory: 80GB 


### 2.2 Data

We provide scripts to automate the downloading and processing of the datasets used for this study.
See scripts in dataset folders. e.g., for the RealWorld dataset, run DATA_RealWorld.py

Please run all scripts in the 'datasets' folder.


Tip: Manually downloading the datasets and placing them in the 'datasets/dataset' folder may be a good alternative for stability if the download pipeline keeps failing VIA the provided scripts.


HHAR
```
http://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition
```

RealWorld
```
https://www.uni-mannheim.de/dws/research/projects/activity-recognition/#dataset_dailylog
```

## 3. Quick Start

We provide both a jupyter notebook (.ipynb) and a python script (.py) versions for all the codes.

### 3.1 Using the our client partitioning 

After downloading and running all the DATA processing scripts in the dataset folder, launch the processFLData.ipynb jupyter notebook OR processFLData.py script to partition the datasets as used in our study.  


### 3.2 Running our federated learning pipeline

After running the provided data processing and partitioning scripts, launch the main.ipynb jupyter notebook OR main.py script to launch our simulated federated learning pipeline. 

An example to launch the script is below:

```
python3.11 ./main.py --algorithm FEDALI --dataset RealWorld --loadPretrain False --communicationRound 200 --clientLearningRate 1e-4 --parallelInstancesGPU 4
```

To select a different federated learning strategy, change the value of the 'algorithm' flag to one of the following:

```
FEDALI, FEDAVG, FEDPROTO, MOON, FEDAVG, FEDALI, FEDPROX
```


To set the number of communication rounds to train the FL strategy, change the value of the 'communicationRound' flag to one of the following. Note we have implemented a checkpoint mechanism; re-launching will resume the last communication round of the training.

To start the training, use our model that was pre-trained on five different datasets using MAE. Set the loadPretrain to 'True'; otherwise, set it to 'False'.

Additionally, we implemented parallel client training. The 'parallelInstancesGPU' flag can be used to specify the number of training instances per GPU. (For example, a value of 4 implies four clients trained in parallel per GPU. If your system has 2 GPUs, this means that 8 clients will be trained in parallel.) Use this cautiously, as multiple instances may exceed your system's memory limitation.



## 4. Acknowledgement

This work has been partially funded by Naval Group, by MIAI@Grenoble Alpes (ANR-19-P3IA-0003), and granted access to the HPC resources of IDRIS under the allocation 2024-AD011013233R2 made by GENCI.

