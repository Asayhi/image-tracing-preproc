import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import autencoder_training as ac
import numpy as np
import pickle as pk
from sklearn.decomposition import PCA
from xml.dom import minidom
import cairosvg
from skimage.io import imread
from PIL import Image
from skimage import img_as_float64
from skimage.metrics import structural_similarity as ssim

def saveWithoutTransparantBackground(inputImagePath, index):
    bg_colour=(255, 255, 255)
    im = Image.open(inputImagePath + "potrace_Pic_"+'{0:03d}'.format(index)+ ".png")
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        bg.save(inputImagePath + "potrace_Pic_"+'{0:03d}'.format(index)+ ".png")
        
def loadPCA():
    pcaPath = ac.getPathFromExplorer("pkl")
    pca_reload = pk.load(open(pcaPath,'rb'))
    return pca_reload


imageCount = 9

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   horizontal_flip = False)


test_generator = test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size = (256, 256),
                                                    color_mode='grayscale',
                                                    batch_size = imageCount,
                                                    shuffle= False
                                                    )
images, y_images = next(test_generator)

defOutputDir    = "processing/Default/"
acOutputDir     = "processing/Autoencoder/"
pcaOutputDir    = "processing/PCA/"
vecDefDir       = "processing/Vectorized/Default/"
vecAcDir        = "processing/Vectorized/Autoencoder/"
vecPcaDir       = "processing/Vectorized/PCA/"
rasterDefDir    = "processing/Rasterized/Default/"
rasterAcDir     = "processing/Rasterized/Autoencoder/"
rasterPCADir    = "processing/Rasterized/PCA/"
compareDir      = "processing/comparision/"
tempDir         = "processing/tmp/"

pathList = [defOutputDir, acOutputDir, pcaOutputDir, vecDefDir, vecAcDir, vecPcaDir, rasterDefDir, rasterAcDir, rasterPCADir, compareDir, tempDir]



evalArray = np.zeros((imageCount, 11))

for path in pathList:
    ac.ensureDirExists(path)


w = h = 3

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def getDefaultImages():
    #-----------------------------------Vanilla Images------------------------------------------
    for i in range(imageCount):
        image = images[i]
        image = image[:,:,0]
        fig = plt.figure(frameon=False) # Figure without frame
        fig.set_size_inches(w,h)
        ax = plt.Axes(fig, [0., 0., 1., 1.]) # Make the image fill out the entire figure
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, cmap=plt.cm.bone, interpolation='nearest', aspect='auto')
        fig.savefig(fname=defOutputDir + "Pic_"+'{0:03d}'.format(i))

def getAutoencoderImages():
    #-----------------------------------Autoencoder--------------------------------------------
    autoencoder, modelDir = ac.loadAutoencoder()
    prediction = autoencoder.predict(images, verbose=1)# you can now display an image to see it is reconstructed well


    for i in range(imageCount):
        fig = plt.figure(frameon=False) # Figure without frame
        fig.set_size_inches(w,h)
        ax = plt.Axes(fig, [0., 0., 1., 1.]) # Make the image fill out the entire figure
        ax.set_axis_off()
        fig.add_axes(ax)
        x = prediction[i]
        ax.imshow(np.reshape(x, (256, 256)), cmap=plt.cm.bone, interpolation='nearest', aspect='auto')
        fig.savefig(fname=acOutputDir + "Pic_"+'{0:03d}'.format(i))


def getPCAImages():
    #------------------------------------PCA--------------------------------------------------
    pca = loadPCA()
    #apply array magic
    x = images.reshape(imageCount, 65536)
    X_proj = pca.transform(x)

    x_inv_proj = pca.inverse_transform(X_proj)

    X_proj_img = x_inv_proj.reshape(imageCount, 256, 256)

    for i in range(imageCount):
        fig = plt.figure(frameon=False) # Figure without frame
        fig.set_size_inches(w,h)
        ax = plt.Axes(fig, [0., 0., 1., 1.]) # Make the image fill out the entire figure
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(X_proj_img[i], cmap=plt.cm.bone, interpolation='nearest', aspect='auto')
        fig.savefig(fname=pcaOutputDir + "Pic_"+'{0:03d}'.format(i))

def convertImagestoVector():
    #---------------------------------Image Conversion------------------------------------------
    for i in range(imageCount):

        #default
        os.system("convert " + defOutputDir + "Pic_"+'{0:03d}'.format(i)+ ".png " + tempDir + "default.ppm")
        os.system("potrace " + tempDir + "default.ppm --output " + vecDefDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".svg -s")

        #autoencoder
        os.system("convert " + acOutputDir + "Pic_"+'{0:03d}'.format(i)+ ".png " + tempDir + "autoenc.ppm")
        os.system("potrace " + tempDir + "autoenc.ppm --output " + vecAcDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".svg -s")
        
        #pca
        os.system("convert " + pcaOutputDir + "Pic_"+'{0:03d}'.format(i)+ ".png " + tempDir + "pca.ppm")
        os.system("potrace " + tempDir + "pca.ppm --output " + vecPcaDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".svg -s")

def svgComparision():
    #-------------------------------------SVG Comparision---------------------------------------


    for i in range(imageCount):

        defsvg = minidom.parse(vecDefDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".svg")
        defCount = len(defsvg.getElementsByTagName('path'))

        acsvg = minidom.parse(vecAcDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".svg")
        acCount = len(acsvg.getElementsByTagName('path'))

        pcasvg = minidom.parse(vecPcaDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".svg")
        pcaCount = len(pcasvg.getElementsByTagName('path'))

        evalArray[i, 0] = defCount
        evalArray[i, 1] = acCount
        evalArray[i, 2] = pcaCount
        


def convertSvgToPng():

    for i in range(imageCount):
        cairosvg.svg2png(url=vecDefDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".svg", 
                         write_to=rasterDefDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".png",
                         output_height=300,
                         output_width=300)

        saveWithoutTransparantBackground(rasterDefDir, i)

        cairosvg.svg2png(url=vecAcDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".svg", 
                         write_to=rasterAcDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".png",
                         output_height=300,
                         output_width=300)

        saveWithoutTransparantBackground(rasterAcDir, i)

        cairosvg.svg2png(url=vecPcaDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".svg", 
                         write_to=rasterPCADir + "potrace_Pic_"+'{0:03d}'.format(i) + ".png",
                         output_height=300,
                         output_width=300)

        saveWithoutTransparantBackground(rasterPCADir, i)

def comparePictures():

    for i in range(imageCount):

        img         = img_as_float64(imread(defOutputDir + "Pic_"+'{0:03d}'.format(i) + ".png ", as_gray=True))
        imgVecDef   = img_as_float64(imread(rasterDefDir + "potrace_Pic_"+'{0:03d}'.format(i) + ".png ", as_gray=True))
        imgVecAc    = img_as_float64(imread(rasterAcDir + "potrace_Pic_"+'{0:03d}'.format(i)+ ".png ", as_gray=True))
        imgVecPCA   = img_as_float64(imread(rasterPCADir + "potrace_Pic_"+'{0:03d}'.format(i)+ ".png ", as_gray=True))

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 4),
                         sharex=True, sharey=True)
        ax = axes.ravel()

        mse_none = mse(img, img)
        ssim_none = ssim(img, img, data_range=img.max() - img.min())

        mse_defaultVec = mse(img, imgVecDef)
        ssim_defaultVec = ssim(img, imgVecDef,
                        data_range=imgVecDef.max() - imgVecDef.min())

        mse_acVec = mse(img, imgVecAc)
        ssim_acVec = ssim(img, imgVecAc,
                        data_range=imgVecAc.max() - imgVecAc.min())

        mse_pcaVec = mse(img, imgVecPCA)
        ssim_pcaVec = ssim(img, imgVecPCA,
                        data_range=imgVecPCA.max() - imgVecPCA.min())

        evalArray[i, 3] = mse_none
        evalArray[i, 4] = mse_defaultVec
        evalArray[i, 5] = mse_acVec
        evalArray[i, 6] = mse_pcaVec

        evalArray[i, 7] = ssim_none
        evalArray[i, 8] = ssim_defaultVec
        evalArray[i, 9] = ssim_acVec
        evalArray[i, 10] = ssim_pcaVec

        label = 'MSE: {:.2f}, SSIM: {:.2f}'

        ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[0].set_xlabel(label.format(mse_none, ssim_none))
        ax[0].set_title('Original image')

        ax[1].imshow(imgVecDef, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[1].set_xlabel(label.format(mse_defaultVec, ssim_defaultVec))
        ax[1].set_title('Default vectorization')

        ax[2].imshow(imgVecAc, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[2].set_xlabel(label.format(mse_acVec, ssim_acVec))
        ax[2].set_title('Autoencoder vectorization')

        ax[3].imshow(imgVecPCA, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax[3].set_xlabel(label.format(mse_pcaVec, ssim_pcaVec))
        ax[3].set_title('PCA vectorization')

        plt.tight_layout()
        plt.savefig(compareDir + "Comparision_"+'{0:03d}'.format(i))

def plotEvaluation():

    svgCategories = ["Default", "Autoencoder", "PCA"]
    mseCategories = ["Reference", "Default", "Autoencoder", "PCA"]
    ssimCategories = ["Reference", "Default", "Autoencoder", "PCA"]
    meanArray = np.mean(evalArray, axis=0)

    svgContent = meanArray[:3].tolist()
    mseContent = meanArray[3:7].tolist()
    ssimContent = meanArray[7:].tolist()

    fig, ax = plt.subplots()
    ax.bar(svgCategories,svgContent)
    fig.suptitle("Number of path objects in svg file")

    plt.savefig(compareDir + "Eval_svg")

    fig, ax = plt.subplots()
    ax.bar(mseCategories,mseContent)
    fig.suptitle("Mean square error compared to original image")

    plt.savefig(compareDir + "Eval_mse")

    fig, ax = plt.subplots()
    ax.bar(ssimCategories,ssimContent)
    fig.suptitle("Structural similarity index compared to original image")

    plt.savefig(compareDir + "Eval_ssim")


def main():
    getDefaultImages()
    getAutoencoderImages()
    getPCAImages()
    convertImagestoVector()
    svgComparision()
    convertSvgToPng()
    comparePictures()
    plotEvaluation()


if __name__ == "__main__":
    main()


