# Convolutional Neural Network

# Part 1 - Building the CNN
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0

# use this to not run out of VRAM
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)


import keras
from matplotlib import pyplot as plt
import numpy as np
from keras.optimizers import RMSprop

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Deconvolution2D, Conv2DTranspose, BatchNormalization
from keras.models import Model

import tkinter
from tkinter import filedialog

# convinince
def getPathFromExplorer(filetype):

    tkinter.Tk().withdraw() # Close the root window
    in_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select "+ filetype)
    return in_path

def ensureDirExists(file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)


#Activate/deactivate cnn training
train = True

#enable/disable prediction
predict = True

epoch = 20000

modelDir = "models/epoch_" + str(epoch) + "/"

resultDir = "autencoder_output/" + str(epoch) + "_epochs/"

# The encoding process
input_img = Input(shape=(256, 256, 1))  


#--------------------------Building the convuluted neural network--------------------------------
#################################################################################################
#################################################################################################

#-----------------------------------------Encoder------------------------------------------------
#################################################################################################
e = Conv2D(16, (3, 3), activation="relu", padding="same")(input_img)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(8, (3, 3), activation="relu", padding="same")(e)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(8, (3, 3), activation="relu", padding="same")(e)
e = MaxPooling2D((2, 2))(e)
e = Conv2D(4, (3, 3), activation="relu", padding="same")(e)


#-------------------------------point of densest information-------------------------------------
# l = Flatten()(e)
# l = Dense(4096, activation='softmax')(l)

#-----------------------------------------Deocder------------------------------------------------
#################################################################################################
# d = Reshape((16,16,16))(l)
d = Conv2DTranspose(4,(3, 3), strides=2, activation='relu', padding='same')(e)
d = BatchNormalization()(d)
d = Conv2DTranspose(8,(3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(8,(3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(16,(3, 3), activation='relu', padding='same')(d)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

#----------------------------------End of Network Description--------------------------------------
###################################################################################################
###################################################################################################

autoencoder = Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer="adam", loss="mse")



# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator()

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   horizontal_flip = False)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   horizontal_flip = False)

def train_images():
    train_generator = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size = (256, 256),
                                                     color_mode='grayscale',
                                                     batch_size = 32,
                                                     shuffle= False
                                                     )
    x = train_generator
    return x[0][0], x[0][1]


def test_images():
    test_generator = test_datagen.flow_from_directory('dataset/test_set',
                                                     target_size = (256, 256),
                                                     color_mode='grayscale',
                                                     batch_size = 4,
                                                     shuffle= False
                                                     )
    x = test_generator
    return x[0][0], x[0][1]

model_json = None
x_train, y_train = train_images()
x_test, y_test = test_images()

if train:    
    # plt.imshow(x_train[0])
    # plt.show()

    ensureDirExists(modelDir)

    history = autoencoder.fit(x_train, x_train, epochs=epoch)

    model_json = autoencoder.to_json()
    with open(modelDir + "model_tex_" + str(epoch) + ".json", "w") as json_file:
        json_file.write(model_json)

    autoencoder.save_weights(modelDir + "model_tex_" + str(epoch) + ".h5")
    print("Saved model")

    print("Plotting Loss")
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

else:
    json_path = getPathFromExplorer("json")
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    print("Json-file loaded")
    # load weights into new model
    h5_path = getPathFromExplorer("h5")
    loaded_model.load_weights(h5_path)
    print("Loaded model from disk")
    autoencoder = loaded_model
    print("Checking loaded Model...")
    autoencoder.compile(optimizer="adam", loss="mse")
    evaluation = autoencoder.evaluate(x_test, x_test)

graph = tf.Graph()
m = autoencoder  # Your model implementation
with graph.as_default():
  # compile method actually creates the model in the graph.
  m.compile(loss='mse',
            optimizer='adam')
writer = tf.summary.FileWriter(logdir='logdir', graph=graph)
writer.flush()

if predict:
    prediction = autoencoder.predict(x_test, verbose=1)# you can now display an image to see it is reconstructed well
    predictions = []
    fig=plt.figure(figsize=(8, 8))
    col = 2
    row = 2

    for i in range(4):
        x = prediction[i].reshape(256, 256, 3)
        predictions.append(x)
        fig.add_subplot(row, col, i+1)
        plt.imshow(x)

    
    ensureDirExists(resultDir)
    plt.savefig(fname=resultDir+"/Result")
    plt.show()

# history = autoencoder.fit_generator(training_set,
#                              epochs = 10,
#                              validation_data = test_set,
#                              )
