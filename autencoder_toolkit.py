# Convolutional Neural Network

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0

# use this to not run out of VRAM
import tensorflow as tf

import keras
from matplotlib import pyplot as plt
import numpy as np
import time
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Deconvolution2D, Conv2DTranspose, BatchNormalization
from keras.models import Model
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import tensorboard

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



def getPathFromExplorer(filetype="data"):

    ''' Open a file explorer at the current working directory and
        return a selected path to a file.'''

    tk.Tk().withdraw() # Close the root window
    in_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select "+ filetype)
    return in_path

def ensureDirExists(file_path):

    '''Chekc if directories of a given file path exist and create them if not.'''

    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def saveAutoencoder(autoencoder, modelDir):

    ''' Save a keras model to a chosen directory by splitting it in a .json 
    and a .h5 file.'''

    model_json = autoencoder.to_json()
    with open(modelDir + "model_tex_" + str(epoch) + ".json", "w") as json_file:
        json_file.write(model_json)

    autoencoder.save_weights(modelDir + "model_tex_" + str(epoch) + ".h5")
    print("Saved model")

def loadAutoencoder(loadDefault=True):

    '''Loads a keras model from the hard drive. Opens a file explorer to
    choose a .json and then a .h5 file from which the model is loaded'''

    if loadDefault:
        json_path = "models/convOnly_5x5_Filter_endTranspose/_epoch_5000/model_tex_5000.json"

    else:
        json_path = getPathFromExplorer("json")

    json_file = open(json_path, 'r')
    resultDir = os.path.dirname(os.path.dirname(json_path)) + "/autencoder_output/"
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    print("Json-file loaded")
    # load weights into new model
    if loadDefault:
        h5_path = "models/convOnly_5x5_Filter_endTranspose/_epoch_5000/model_tex_5000.h5"

    else:    
        h5_path = getPathFromExplorer("h5")
        
    loaded_model.load_weights(h5_path)
    print("Loaded model from disk")
    autoencoder = loaded_model
    return autoencoder, resultDir

def defineAutoencoder():

    '''Defines the Autoencoder which is trained.
    This function should be edited to suit your needs'''

    # The encoding process
    input_img = Input(shape=(256, 256, 1))  


    #--------------------------Building the convuluted neural network--------------------------------
    #################################################################################################
    #################################################################################################

    #-----------------------------------------Encoder------------------------------------------------
    #################################################################################################
    e = Conv2D(16, (5, 5), activation="relu", padding="same")(input_img)
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
    # e = Reshape((32,32,4))(l)
    d = Conv2DTranspose(4,(3, 3), strides=2, activation='relu', padding='same')(e)
    d = BatchNormalization()(d)
    d = Conv2DTranspose(8,(3, 3), strides=2, activation='relu', padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2DTranspose(8,(3, 3), strides=2, activation='relu', padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2DTranspose(16,(3, 3), activation='relu', padding='same')(d)
    decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(d)

    #----------------------------------End of Network Description--------------------------------------
    ###################################################################################################
    ###################################################################################################

    autoencoder = Model(input_img, decoded)

    autoencoder.summary()

    autoencoder.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

    return autoencoder

def main():

    #Activate/deactivate cnn training
    train = False

    #enable/disable prediction
    predict = True

    specifica = "dense_mid_layer_5x5_Kernel"
    epoch = 5000

    modelDir = "models/"+ specifica + "/epoch_" + str(epoch) + "/"

    resultDir = modelDir + "autencoder_output/"

    # Define the Keras TensorBoard callback.
    logDir="logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logDir)

    autoencoder = defineAutoencoder()


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
                                                        batch_size = 64,
                                                        shuffle= False
                                                        )
        x = train_generator
        return x[0][0], x[0][1]


    def test_images():
        test_generator = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size = (256, 256),
                                                        color_mode='grayscale',
                                                        batch_size = 8,
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
        ensureDirExists(logDir)

        time.sleep(1)

        history = autoencoder.fit(x_train, x_train, epochs=epoch, callbacks=[tensorboard_callback])

        saveAutoencoder(autoencoder, modelDir)

        print("Plotting Loss")
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    else:
        autoencoder, resultDir = loadAutoencoder()
        print("Checking loaded Model...")
        autoencoder.compile(optimizer="adam", loss="mse")
        evaluation = autoencoder.evaluate(x_test, x_test)



    if predict:
        prediction = autoencoder.predict(x_test, verbose=1)# you can now display an image to see it is reconstructed well
        predictions = []
        fig = plt.figure(frameon=False)
        col = 4
        row = 2

        for i in range(8):
            x = prediction[i]
            predictions.append(x)
            fig.add_subplot(row, col, i+1)
            plt.imshow(np.reshape(x, (256, 256)), cmap=plt.cm.bone, interpolation='nearest')
            plt.tight_layout()

        
        ensureDirExists(resultDir)
        plt.savefig(fname=resultDir+"/Result")
        plt.show()

if __name__ == "__main__":
    main()