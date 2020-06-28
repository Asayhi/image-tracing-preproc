import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import autencoder_training as ac
import numpy as np
import pickle as pk
from sklearn.decomposition import PCA

def loadPCA():
    pca_path = ac.getPathFromExplorer(".pkl")
    pca_reload = pk.load(open(pca_path,'rb'))
    return pca_reload

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   horizontal_flip = False)


test_generator = test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size = (256, 256),
                                                    color_mode='grayscale',
                                                    batch_size = 9,
                                                    shuffle= False
                                                    )
images, y_images = next(test_generator)


acOutputDir = "processing/Autoencoder/"

ac.ensureDirExists(acOutputDir)


fig = plt.figure(figsize=(3,3))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(9):
    image = images[i]
    image = image[:,:,0]
    ax = fig.add_subplot(3, 3, i+1, xticks=[], yticks=[])
    ax.imshow(image, cmap=plt.cm.bone, interpolation='nearest')

plt.show()


autoencoder, modelDir = ac.loadAutoencoder()
prediction = autoencoder.predict(images, verbose=1)# you can now display an image to see it is reconstructed well
predictions = []
w = h = 3

for i in range(9):
    fig = plt.figure(frameon=False) # Figure without frame
    fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.]) # Make the image fill out the entire figure
    ax.set_axis_off()
    fig.add_axes(ax)
    x = prediction[i]
    ax.imshow(np.reshape(x, (256, 256)), cmap=plt.cm.bone, interpolation='nearest', aspect='auto')
    fig.savefig(fname=acOutputDir + "Pic_"+'{0:03d}'.format(i))

pca = loadPCA()
X_proj = pca.fit_transform(images)