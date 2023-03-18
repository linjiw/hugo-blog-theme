---
title: When Cats meet GANs Results
subtitle: Learn how to blog in Academic using Jupyter notebooks
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
    # add addtional data augmentation here
    # load_size = int(1.1 * opts.image_size)
    # osize = [load_size, load_size]
    # transforms.Resize(osize, Image.BICUBIC)
    # transforms.RandomCrop(opts.image_size)
    # transforms.RandomHorizontalFlip()
    pass
```

### Implement the Discriminator of the DCGAN
(Answer for padding calculation goes here)

Implemented the architecture by filling in the '__init__' and 'forward' method of the 'DCDiscriminator' class in 'models.py'.

```python
def __init__(self, conv_dim=64):
    ...
    # self.conv1 = conv(...)
    # self.conv2 = conv(...)
    # self.conv3 = conv(...)
    # self.conv4 = conv(...)
    # self.conv5 = conv(...)
    
def forward(self, x):
    ...
    pass
```

### Generator
Implemented the generator of the DCGAN by filling in the '__init__' and 'forward' method of the 'DCGenerator' class in 'models.py'.

```python
Copy code
def __init__(self, conv_dim=64):
    ...
    # self.up_conv1 = ...
    # self.up_conv2 = ...
    # self.up_conv3 = ...
    # self.up_conv4 = ...
    
def forward(self, x):
    ...
    pass
```






```python
from IPython.core.display import Image
Image('https://www.python.org/static/community_logos/python-logo-master-v3-TM-flattened.png')
```




![png](./index_1_0.png)




```python
print("Welcome to Academic!")
```

    Welcome to Academic!


## Install Python and JupyterLab

[Install Anaconda](https://www.anaconda.com/distribution/#download-section) which includes Python 3 and JupyterLab.

Alternatively, install JupyterLab with `pip3 install jupyterlab`.

## Create or upload a Jupyter notebook

Run the following commands in your Terminal, substituting `<MY-WEBSITE-FOLDER>` and `<SHORT-POST-TITLE>` with the file path to your Academic website folder and a short title for your blog post (use hyphens instead of spaces), respectively:

```bash
mkdir -p <MY-WEBSITE-FOLDER>/content/post/<SHORT-POST-TITLE>/
cd <MY-WEBSITE-FOLDER>/content/post/<SHORT-POST-TITLE>/
jupyter lab index.ipynb
```

The `jupyter` command above will launch the JupyterLab editor, allowing us to add Academic metadata and write the content.

## Edit your post metadata

The first cell of your Jupter notebook will contain your post metadata ([front matter](https://sourcethemes.com/academic/docs/front-matter/)).

In Jupter, choose _Markdown_ as the type of the first cell and wrap your Academic metadata in three dashes, indicating that it is YAML front matter: 

```
---
title: My post's title
date: 2019-09-01

# Put any other Academic metadata here...
---
```

Edit the metadata of your post, using the [documentation](https://sourcethemes.com/academic/docs/managing-content) as a guide to the available options.

To set a [featured image](https://sourcethemes.com/academic/docs/managing-content/#featured-image), place an image named `featured` into your post's folder.

For other tips, such as using math, see the guide on [writing content with Academic](https://sourcethemes.com/academic/docs/writing-markdown-latex/). 

## Convert notebook to Markdown

```bash
jupyter nbconvert index.ipynb --to markdown --NbConvertApp.output_files_dir=.
```

## Example

This post was created with Jupyter. The orginal files can be found at https://github.com/gcushen/hugo-academic/tree/master/exampleSite/content/post/jupyter
