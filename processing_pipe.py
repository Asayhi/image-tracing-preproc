import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import autencoder_training as ac
import numpy as np

def loading_autoencoder_model():
    json_path = ac.getPathFromExplorer("json")
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    print("Json-file loaded")
    # load weights into new model
    h5_path = ac.getPathFromExplorer("h5")
    loaded_model.load_weights(h5_path)
    print("Loaded model from disk")
    autoencoder = loaded_model
    print("Checking loaded Model...")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


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


resultDir = "processing/"

fig = plt.figure(figsize=(3,3))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(9):
    image = images[i]
    image = image[:,:,0]
    ax = fig.add_subplot(3, 3, i+1, xticks=[], yticks=[])
    ax.imshow(image, cmap=plt.cm.bone, interpolation='nearest')

plt.show()


autoencoder = loading_autoencoder_model()
prediction = autoencoder.predict(images, verbose=1)# you can now display an image to see it is reconstructed well
predictions = []
fig=plt.figure(figsize=(8, 8))
col = 2
row = 4

for i in range(8):
    x = prediction[i]
    predictions.append(x)
    fig.add_subplot(row, col, i+1)
    plt.imshow(np.reshape(x, (256, 256)), cmap=plt.cm.bone, interpolation='nearest')


ac.ensureDirExists(resultDir)
plt.savefig(fname=resultDir+"Autoencoder")
plt.show()

