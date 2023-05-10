# HSurf-Net: Normal Estimation for 3D Point Clouds by Learning Hyper Surfaces (NeurIPS 2022)

### **[Project](https://leoqli.github.io/HSurf-Net/) | [arXiv](https://arxiv.org/abs/2210.07158) | [Supplementary](https://drive.google.com/file/d/1F1LwZEzvupztCeXEtlMJ1zOrV-EOoZTT/view?usp=sharing) | [Dataset](https://drive.google.com/drive/folders/1eNpDh5ivE7Ap1HkqCMbRZpVKMQB1TQ6H?usp=share_link)**

We propose a novel normal estimation method called HSurf-Net, which can accurately predict normals from point clouds with noise and density variations. Previous methods focus on learning point weights to fit neighborhoods into a geometric surface approximated by a polynomial function with a predefined order, based on which normals are estimated. However, fitting surfaces explicitly from raw point clouds suffers from overfitting or underfitting issues caused by inappropriate polynomial orders and outliers, which significantly limits the performance of existing methods. To address these issues, we introduce hyper surface fitting to implicitly learn hyper surfaces, which are represented by multi-layer perceptron (MLP) layers that take point features as input and output surface patterns in a high dimensional feature space. We introduce a novel space transformation module, which consists of a sequence of local aggregation layers and global shift layers, to learn an optimal feature space, and a relative position encoding module to effectively convert point clouds into the learned feature space. Our model learns hyper surfaces from the noise-less features and directly predicts normal vectors. We jointly optimize the MLP weights and module parameters in a data-driven manner to make the model adaptively find the most suitable surface pattern for various points. Experimental results show that our HSurf-Net achieves the state-of-the-art performance on the synthetic shape dataset, the real-world indoor and outdoor scene datasets.

## Requirements

The code is implemented in the following environment settings:
- Ubuntu 16.04
- CUDA 10.1
- Python 3.8
- Pytorch 1.8
- Pytorch3d 0.6
- Numpy 1.23
- Scipy 1.6

## Dataset
We train our network model on the PCPNet dataset.
We provide the preprocessed data for SceneNN dataset and Semantic3D dataset.
They can be downloaded from [here](https://drive.google.com/drive/folders/1eNpDh5ivE7Ap1HkqCMbRZpVKMQB1TQ6H?usp=share_link).
Unzip them to a folder `***/dataset/` and set the value of `dataset_root` in `run.py`. The dataset is organized as follows:
```
│dataset/
├──PCPNet/
│  ├── list
│      ├── ***.txt
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
├──SceneNN/
│  ├── list
│      ├── ***.txt
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
├──Semantic3D/
│  ├── list
│      ├── ***.txt
│  ├── ***.xyz
```

## Train
Our trained model is provided in `./log/001/ckpts/ckpt_900.pt`.
To train a new model on the PCPNet dataset, simply run:
```
python run.py --gpu=0 --mode=train --data_set=PCPNet
```
Your trained model will be save in `./log/***_***/`.

## Test
You can use the provided model for testing:
- PCPNet dataset
```
python run.py --gpu=0 --mode=test --data_set=PCPNet
```
- SceneNN dataset
```
python run.py --gpu=0 --mode=test --data_set=SceneNN
```
- Semantic3D dataset
```
python run.py --gpu=0 --mode=test --data_set=Semantic3D
```
The evaluation results will be saved in `./log/001/results_***/ckpt_900/`.
To test with your trained model, simply run:
```
python run.py --gpu=0 --mode=test --data_set=*** --ckpt_dirs=***_*** --epoch=***
```
To save the normals of the input point cloud, you need to change the variables in `run.py`:
```
save_pn = False        # to save the point normals as '.normals' file
sparse_patches = True  # to output sparse point normals or not
```

## Results
Our normal estimation results on the datasets PCPNet, SceneNN and Semantic3D can be downloaded from [here](https://drive.google.com/drive/folders/1fZnUqqJLHYhF3zjqx6owrqFJQ_EJsINN?usp=sharing).

## Citation
If you find our work useful in your research, please cite our paper:

    @article{li2022hsurf,
      title={{HSurf-Net}: Normal Estimation for {3D} Point Clouds by Learning Hyper Surfaces},
      author={Li, Qing and Liu, Yu-Shen and Cheng, Jin-San and Wang, Cheng and Fang, Yi and Han, Zhizhong},
      journal={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2022}
    }

