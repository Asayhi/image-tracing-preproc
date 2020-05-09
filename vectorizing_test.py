from PIL import Image
import os
imageLoc = "dataset/test_set/cats/00001292_021.jpg"
imageDest = "dataset/test_set/vector_cats/00001292_021.svg"
im = Image.open("dataset/test_set/cats/00001292_021.jpg")
im.show()
data = list(im.getdata())
os.system("convert " + imageLoc + " -flatten out.pgm")
os.system("potrace out.pgm --svg -o " + imageDest)
