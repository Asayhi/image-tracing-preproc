import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

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

fig = plt.figure(figsize=(3,3))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(9):
    image = images[i]
    image = image[:,:,0]
    ax = fig.add_subplot(3, 3, i+1, xticks=[], yticks=[])
    ax.imshow(image, cmap=plt.cm.bone, interpolation='nearest')

plt.show()
