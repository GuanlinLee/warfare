# Requirements

python, 
pytorch,
torchvision,
numpy,
torchsde,
torchdiffeq,
lpips,
kornia,
pytorch-fid,
clip


# Code Structure

The code is organized as follows:

- `case_study/`: the code for the case study part. Refer to the README.md in this directory for more details.
- `DiffPure/`: the code for training and evaluating diffusion models. Refer to the README.md in this directory for more details.
- `models/`: the code for auto-encoder / generator and decoder models.
- `adversary_AE_train.py`: the code for train a generative model for the adversary.
- `adversary_data_gen.py`: the code for generating data for the adversary.
- `dataloader.py`: the code for loading data.
- `evaluation.py`: the code for evaluating the performance of the adversary.
- `utils.py`: the code for some utility functions.
- `owner_AE_train.py`: the code for training a generative model for the watermark for the model owner.


# Dataset

It supports CIFAR-10, CelebA, LSUN, and FFHQ.

# How to use

## Step 1. Train the watermark model

First, train the watermark model. For example, train a watermark model with 1 bit watermark on CIFAR-10:

```
python owner_AE_train.py --dataset cifar10 --save cifar-10 --exp cifar-10 --watermark_size 1 --watermark_test 1 
```

## Step 2. Generate data for the adversary

Second, generate data for the adversary. For example, generate 25000 images with 1 bit watermark:

```
python adversary_data_gen.py --config ./path/to/your/diffusionmodel/config.yaml
--exp cifar-10-gen --dataset cifar10 --save cifar-10 --expn cifar-10 
--watermark_size 1 --watermark_test 1
```

## Step 3. Train the auto-encoder model for the adversary

Third, train the auto-encoder model for the adversary. For example, train a model with 1 bit watermark:

```
python adversary_AE_train.py --dataset cifar10 --w1_errG 10 --w2_errG 100 --train_size 25000
--save cifar10 --exp cifar10 --targe forge --watermark_size 1 --watermark_test 1
```

## Step 4. Evaluate the performance of the adversary

Finally, evaluate the performance of the adversary. For example, evaluate the performance of the adversary with 1 bit watermark:

```
python evaluation.py --dataset cifar10 --save cifar10 --exp cifar10 
--targe forge --watermark_size 1 --watermark_test 1
```

## Remark: How to calculate the FID score

If running the evaluation.py, you may have already saved the generated images and the real images. Then you can use the following command to calculate the FID score:

```
python -m pytorch_fid path/to/generated_images path/to/real_images
``` 

Check the evaluation.py for more details.

