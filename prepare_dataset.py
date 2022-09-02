from PIL import Image
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

root = 'dataset/data_train'
dest = 'C:/Users/tobys/Documents/TerraGAN/dataset/'
rgbpaths = np.array([np.array(fname) for fname in glob.glob(os.path.join(root, '*t.png'))])
heightpaths = np.array([np.array(fname) for fname in glob.glob(os.path.join(root, '*h.png'))])
segpaths = np.array([np.array(fname) for fname in glob.glob(os.path.join(root, '*i2.png'))])

i=1
table=[ i/256 for i in range(65536) ]
for path in heightpaths:
    im = Image.open(path)
    im2 = im.point(table,'L')
    im2.save(dest + 'height/' + str(i) + '.png')
    i+=1

# i=1
# for path in rgbpaths:
#     im = Image.open(path)
#     im.save(dest + 'texture/' + str(i) + '.png')
#     i+=1

# i=1
# for path in segpaths:
#     im = Image.open(path)
#     im.save(dest + 'segmentation/' + str(i) + '.png')
#     i+=1

# i=1
# for t, h in zip(rgbpaths, heightpaths):
#     imt = Image.open(t)
#     imh = Image.open(h)
#     rgba = np.array(imt.convert('RGBA'))
#     rgba[:,:,3] = np.array(imh)
#     imt = Image.fromarray(rgba)
#     imt.save(dest + 'combined/' + str(i) + '.png')
#     i+=1