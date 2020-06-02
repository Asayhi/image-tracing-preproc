import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0


import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D, Dense
from keras.models import Model
from keras.optimizers import RMSprop
import cv2
import dataset


im = cv2.imread('images/cat-dataset(1)/CAT_00/00000015_028.jpg')

train_path = 'images/cat-dataset(1)/CAT_00'

print(type(im))
# <class 'numpy.ndarray'>

print(im.shape)
print(type(im.shape))


# 10% of the data will automatically be used for validation
validation_size = 0.1
img_size = 256 # resize images to be 48x48
num_channels = 3 # RGB
sample_size = 4096 #We'll use 8192 pictures (2**13)

data = dataset.read_train_sets(train_path, img_size, ['cats'],
                               validation_size=validation_size, 
                               sample_size=sample_size)

x_train, _, _, _ = data.train.next_batch(7373) #no need to load them in batches
x_valid, _, _, _ = data.valid.next_batch(819)  # now that tensorflow can do it
x_train[0] # array([[[0.6784314 , 0.6784314 , 0.7019608 ], ...)



