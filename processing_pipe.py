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

image_count = 9

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   horizontal_flip = False)


test_generator = test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size = (256, 256),
                                                    color_mode='grayscale',
                                                    batch_size = image_count,
                                                    shuffle= False
                                                    )
images, y_images = next(test_generator)

defOutputDir = "processing/Default/"
acOutputDir = "processing/Autoencoder/"
pcaOutputDir = "processing/PCA/"
vecDefDir = "processing/Vectorized/Default/"
vecAcDir = "processing/Vectorized/Autoencoder/"
vecPcaDir = "processing/Vectorized/PCA/"
tempDir = "processing/tmp/"

pathList = [defOutputDir, acOutputDir, pcaOutputDir, vecDefDir, vecAcDir, vecPcaDir, tempDir]

for path in pathList:
    ac.ensureDirExists(path)


fig = plt.figure(figsize=(3,3))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

w = h = 3

#-----------------------------------Vanilla Images------------------------------------------
for i in range(image_count):
    image = images[i]
    image = image[:,:,0]
    fig = plt.figure(frameon=False) # Figure without frame
    fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.]) # Make the image fill out the entire figure
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, cmap=plt.cm.bone, interpolation='nearest', aspect='auto')
    fig.savefig(fname=defOutputDir + "Pic_"+'{0:03d}'.format(i))


#-----------------------------------Autoencoder--------------------------------------------
autoencoder, modelDir = ac.loadAutoencoder()
prediction = autoencoder.predict(images, verbose=1)# you can now display an image to see it is reconstructed well
predictions = []


for i in range(image_count):
    fig = plt.figure(frameon=False) # Figure without frame
    fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.]) # Make the image fill out the entire figure
    ax.set_axis_off()
    fig.add_axes(ax)
    x = prediction[i]
    ax.imshow(np.reshape(x, (256, 256)), cmap=plt.cm.bone, interpolation='nearest', aspect='auto')
    fig.savefig(fname=acOutputDir + "Pic_"+'{0:03d}'.format(i))



#------------------------------------PCA--------------------------------------------------
pca = loadPCA()
#apply array magic
x = images.reshape(image_count, 65536)
X_proj = pca.transform(x)

x_inv_proj = pca.inverse_transform(X_proj)

X_proj_img = x_inv_proj.reshape(image_count, 256, 256)

for i in range(image_count):
    fig = plt.figure(frameon=False) # Figure without frame
    fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.]) # Make the image fill out the entire figure
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(X_proj_img[i], cmap=plt.cm.bone, interpolation='nearest', aspect='auto')
    fig.savefig(fname=pcaOutputDir + "Pic_"+'{0:03d}'.format(i))


#---------------------------------Image Conversion------------------------------------------
for i in range(image_count):

    #default
    os.system("convert " + defOutputDir + "Pic_"+'{0:03d}'.format(i)+ ".png " + tempDir + "default.ppm")
    os.system("potrace " + tempDir + "default.ppm --output " + vecDefDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".svg -s")

    #autoencoder
    os.system("convert " + acOutputDir + "Pic_"+'{0:03d}'.format(i)+ ".png " + tempDir + "autoenc.ppm")
    os.system("potrace " + tempDir + "autoenc.ppm --output " + vecAcDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".svg -s")
    
    #pca
    os.system("convert " + pcaOutputDir + "Pic_"+'{0:03d}'.format(i)+ ".png " + tempDir + "pca.ppm")
    os.system("potrace " + tempDir + "pca.ppm --output " + vecPcaDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".svg -s")