# Part 1 - Building the CNN
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0

# use this to not run out of VRAM
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.optimizers import RMSprop

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Deconvolution2D, Conv2DTranspose, BatchNormalization
from keras.models import Model

import tkinter
from tkinter import filedialog

# convinince
def getPathFromExplorer(filetype):

    tkinter.Tk().withdraw() # Close the root window
    in_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select"+ filetype)
    return in_path


#Activate/deactivate cnn training
train = True

#enable/disable prediction
predict = True

epoch = 20000


# The encoding process
input_img = Input(shape=(256, 256, 3))  

# Part 1 - Building the convuluted neural network

#Enoder
e = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(32, (3, 3), activation="relu", padding="same")(e)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(32, (3, 3), activation="relu", padding="same")(e)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(32, (3, 3), activation="relu", padding="same")(e)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(32, (3, 3), activation="relu", padding="same")(e)
l = Flatten()(e)

#point of most dense information
l = Dense(8192, activation='softmax')(l)

#Deocder
d = Reshape((16,16,32))(l)
d = Conv2DTranspose(32,(3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(32,(3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(32,(3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(32,(3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(16,(3, 3), activation='relu', padding='same')(d)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(d)

autoencoder = Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer="adam", loss="mse")


