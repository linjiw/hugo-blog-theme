---
title: When Cats meet GANs 
subtitle: A Comprehensive Study on DCGANs and CycleGANs with Advanced Augmentation Techniques
summary: In this assignment, we implemented two types of GANs - a Deep Convolutional GAN (DCGAN) and a CycleGAN. The DCGAN was trained to generate grumpy cats from random noise, while the CycleGAN was trained to convert between two types of cats (Grumpy and Russian Blue) and between apples and oranges. Both GANs were implemented with data augmentation and differentiable augmentation techniques.
authors:
- admin
tags:
- Computer Vision
- Image Generation
- Deep Learning
# - 开源

categories:
- Project
# - 教程
projects: []
date: "2019-02-05T00:00:00Z"
lastMod: "2019-09-05T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'DCGAN Results'
  focal_point: ""
  placement: 2
  preview_only: false
---

## Introduction


In this assignment, we get hands-on experience coding and training GANs. This assignment includes two parts:

Implementing a Deep Convolutional GAN (DCGAN) to generate grumpy cats from samples of random noise.
Implementing a more complex GAN architecture called CycleGAN for the task of image-to-image translation. We train the CycleGAN to convert between different types of two kinds of cats (Grumpy and Russian Blue) and between apples and oranges.

## Part 1: Deep Convolutional GAN

For the first part of this assignment, we implement a slightly modified version of Deep Convolutional GAN (DCGAN).

### Implement Data Augmentation
Implemented the deluxe version of data augmentation in 'data_loader.py'.

```python
elif opts.data_preprocess == 'deluxe':
        load_size = int(1.1 * opts.image_size)
        osize = [load_size, load_size]
        deluxe_transform = transforms.Compose([
        transforms.Resize(opts.image_size, Image.BICUBIC),
        transforms.RandomCrop(opts.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_transform = deluxe_transform
    pass
```

### Implement the Discriminator of the DCGAN
(Answer for padding calculation goes here)

Implemented the architecture by filling in the '__init__' and 'forward' method of the 'DCDiscriminator' class in 'models.py'.

```python
def __init__(self, conv_dim=64, norm='instance'):
    super().__init__()
    self.conv1 = conv(3, 32, 4, 2, 1, norm, False, 'relu')
    self.conv2 = conv(32, 64, 4, 2, 1, norm, False, 'relu')
    self.conv3 = conv(64, 128, 4, 2, 1, norm, False, 'relu')
    self.conv4 = conv(128, 256, 4, 2, 1, norm, False, 'relu')
    self.conv5 = conv(256, 1, 4, 2, 0, None, False)

def forward(self, x):
    """Forward pass, x is (B, C, H, W)."""
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    return x.squeeze()
```

### Generator
Implemented the generator of the DCGAN by filling in the '__init__' and 'forward' method of the 'DCGenerator' class in 'models.py'.

```python
def __init__(self, noise_size, conv_dim=64):
    super().__init__()

    self.up_conv1 = conv(100, 256, 2, 1, 2, 'instance', False,'relu' )
    self.up_conv2 = up_conv(256, 128, 3, stride=1, padding=1, scale_factor=2, norm='instance', activ='relu')
    self.up_conv3 = up_conv(128, 64, 3, stride=1, padding=1, scale_factor=2, norm='instance', activ='relu')
    self.up_conv4 = up_conv(64, 32, 3, stride=1, padding=1, scale_factor=2, norm='instance', activ='relu')
    self.up_conv5 = up_conv(32, 3, 3, stride=1, padding=1, scale_factor=2, norm= None, activ='tanh')

def forward(self, z):
    """
    Generate an image given a sample of random noise.

    Input
    -----
        z: BS x noise_size x 1 x 1   -->  16x100x1x1

    Output
    ------
        out: BS x channels x image_width x image_height  -->  16x3x64x64
    """

    z = self.up_conv1(z)
    z = self.up_conv2(z)
    z = self.up_conv3(z)
    z = self.up_conv4(z)
    z = self.up_conv5(z)
    return z
```

### Training Loop
Implemented the training loop for the DCGAN by filling in the indicated parts of the training_loop function in vanilla_gan.py.

```python
            # TRAIN THE DISCRIMINATOR
            # 1. Compute the discriminator loss on real images
            if opts.use_diffaug:
                D_real_loss = torch.mean((D(DiffAugment(real_images, policy='color,translation,cutout', channels_first=False )) - 1) ** 2)
            else:
                D_real_loss = torch.mean((D(real_images) - 1) ** 2)


            # 2. Sample noise
            noise = sample_noise(opts.batch_size, opts.noise_size)

            # 3. Generate fake images from the noise
            fake_images = G(noise)

            # 4. Compute the discriminator loss on the fake images
            if opts.use_diffaug:

                D_fake_loss = torch.mean((D(DiffAugment(fake_images.detach(), policy='color,translation,cutout', channels_first=False ))) ** 2)
            else:
                D_real_loss = torch.mean((D(fake_images.detach())) ** 2)
            D_total_loss = (D_real_loss + D_fake_loss) / 2

            # update the discriminator D
            d_optimizer.zero_grad()
            D_total_loss.backward()
            d_optimizer.step()

            # TRAIN THE GENERATOR
            # 1. Sample noise
            noise = sample_noise(opts.batch_size, opts.noise_size)

            # 2. Generate fake images from the noise
            fake_images = G(noise)

            # 3. Compute the generator loss
            if opts.use_diffaug:
                G_loss = torch.mean((D(DiffAugment(fake_images, policy='color,translation,cutout', channels_first=False ))-1) ** 2)
            else:
                G_loss = torch.mean((D(fake_images)-1) ** 2)

```

#### Differentiable Augmentation
(Discussion of results with and without applying differentiable augmentations, and the difference between two augmentation schemes in terms of implementation and effects)

### Experiment with DCGANs
INSERT IMAGE: Screenshots of discriminator and generator training loss with --data_preprocess=basic, --data_preprocess=deluxe.

(Brief explanation of what the curves should look like if GAN manages to train)

INSERT IMAGE: With --data_preprocess=deluxe and differentiable augmentation enabled, show one of the samples from early in training (e.g., iteration 200) and one of the samples from later in training, and give the iteration number for those samples.

(Brief comment on the quality of the samples, and in what way they improve through training)

## Part 2: CycleGAN
Implemented the CycleGAN architecture.

### Data Augmentation
Set the --data_preprocess flag to deluxe.

### Generator
Implemented the generator architecture by completing the __init__ method of the CycleGenerator class in models.py.

```python
def __init__(self, conv_dim=64, init_zero_weights=False, norm='instance'):
    super().__init__()

    # # 1. Define the encoder part of the generator
    self.conv1 = conv(3, 32, 4, 2, 1, norm, False, 'relu')
    self.conv2 = conv(32, 64, 4, 2, 1, norm, False, 'relu')

    # # 2. Define the transformation part of the generator
    self.resnet_block = nn.Sequential(ResnetBlock(conv_dim = 64, norm = norm, activ = 'relu'),
                                      ResnetBlock(conv_dim = 64, norm = norm, activ = 'relu'),
                                      ResnetBlock(conv_dim = 64, norm = norm, activ = 'relu'),)

    # # 3. Define the decoder part of the generator
    self.up_conv1 = up_conv(64, 32, 3, stride=1, padding=1, scale_factor=2, norm='instance', activ='relu')
    self.up_conv2 = up_conv(32, 3, 3, stride=1, padding=1, scale_factor=2, norm= None, activ='tanh')

def forward(self, x):
    """
    Generate an image conditioned on an input image.

    Input
    -----
        x: BS x 3 x 32 x 32

    Output
    ------
        out: BS x 3 x 32 x 32
    """
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.resnet_block(x)
    x = self.up_conv1(x)
    x = self.up_conv2(x)

    return x
```
### Training Loop
Implemented the training loop for the CycleGAN by filling in the indicated parts of the training_loop function in cycle_gan.py.

### Experiment with CycleGAN
INSERT IMAGE: Two example images of generated Grumpy cats from Russian Blue cats, and two example images of generated Russian Blue cats from Grumpy cats.

(Brief comment on the quality of the generated images, and whether the CycleGAN has captured the main differences between the two domains)

INSERT IMAGE: Two example images of generated apples from oranges, and two example images of generated oranges from apples.

(Brief comment on the quality of the generated images, and whether the CycleGAN has captured the main differences between the two domains)

## Conclusion
This report presents our implementation of DCGAN and CycleGAN for various image generation tasks. Through these experiments, we have observed the impact of data augmentation and differentiable augmentation on the training process and final results. We have also seen the capabilities of CycleGAN in generating realistic images for domain-to-domain translation tasks, such as converting Grumpy cats to Russian Blue cats and vice versa, and converting apples to oranges and vice versa.
