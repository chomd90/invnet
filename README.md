# InvNet: Encoding Geometric and Statistical Invariance in Deep Generative Models

Pytorch implementation of WGAN-GP with a projection operator.

## Acknowledgements
* Modification of [link](https://github.com/jalola/improved-wgan-pytorch/)
* [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training)
* [caogang/wgan-gp](https://github.com/caogang/wgan-gp)
* [LayerNorm](https://github.com/pytorch/pytorch/issues/1959)

## Prerequisites
* Python >= 3.6
* [Pytorch >= v1.0.0](https://github.com/pytorch/pytorch)
* Numpy
* SciPy
* tensorboardX ([installation here](https://github.com/lanpa/tensorboard-pytorch)). It is very convenient to see costs and results during training with TensorboardX for Pytorch
* TensorFlow for tensorboardX

## Model

* `gan_train.py`: This model is mainly based on `GoodGenerator` and `GoodDiscriminator` from [Improved Training of Wasserstein GANs](https://github.com/igul222/improved_wgan_training). Generator and discriminator are modified for 128x128 (width, height) dataset. 

* Usage example, run: `python gan_train.py --dataset 'circle' --output 'output_dir'`. Then, the output file will be written to `./output_dir` folder. 

## Dataset

* This is the toy example dataset. You can use "train_toyCircle_3Ch_128.h5" dataset for the training. 
* https://drive.google.com/drive/folders/1eQCZtni4UvilOI4-nQHBhRQyMvTpOizN?usp=sharing

## Testing
During the implementation of this model, we built a test module to compare the result between original model (Tensorflow) and our model (Pytorch) for every layer we implemented. It is available at [compare-tensorflow-pytorch](https://github.com/jalola/compare-tensorflow-pytorch)

## TensorboardX
Results such as costs, generated images (every 200 iters) for tensorboard will be written to `./runs` folder.

To display the results to tensorboard, run: `tensorboard --logdir runs`

