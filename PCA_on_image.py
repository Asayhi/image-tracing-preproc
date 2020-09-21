from sklearn.decomposition import PCA
import pickle as pk
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

PCA_modeldir = "models/PCA/"

def ensureDirExists(file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

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
                                                     batch_size = 800,
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

x_train, y_train = train_images()
x_test, y_test = test_images()

#apply some array magic to get the right shapes
train_set = x_train.reshape(800, 65536)

print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)
print (train_set.shape)
print (train_set)


x = train_set


pca = PCA(256)

print('Fitting PCA')
pca.fit(x)

X_proj = pca.fit_transform(x)
ensureDirExists(PCA_modeldir)
pk.dump(pca, open(PCA_modeldir + "PCA_256.pkl","wb"))
print (X_proj.shape)

#everything else is for visualiziation
print (np.cumsum(pca.explained_variance_ratio_))

fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(10):
    ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
    ax.imshow(np.reshape(pca.components_[i,:], (256,256)), cmap=plt.cm.bone, interpolation='nearest')

plt.savefig(fname=PCA_modeldir+"PCA_compononts.png")
plt.show()

x_inv_proj = pca.inverse_transform(X_proj)

X_proj_img = x_inv_proj.reshape(800, 256, 256)

fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(X_proj_img[i], cmap=plt.cm.bone, interpolation='nearest')

plt.savefig(fname=PCA_modeldir+"PCA_result.png")
plt.show()

#x_train = pca.transform(x_train)
#x_test = pca.transform(x_test)