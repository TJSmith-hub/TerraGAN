import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# get all data paths
root = 'dataset/data_train'
rgbpaths = np.array([np.array(fname) for fname in glob.glob(os.path.join(root, '*t.png'))])
heightpaths = np.array([np.array(fname) for fname in glob.glob(os.path.join(root, '*h.png'))])
segpaths = np.array([np.array(fname) for fname in glob.glob(os.path.join(root, '*i2.png'))])

# class names with corresponding colors
def class_count():
    classes = {                                     #class name: rgb value #gray value
        'Water': (0.06667,0.55294,0.84314),         #112
        'Forest': (0.49804,0.67843,0.48235),        #153
        'Grassland': (0.88235,0.89020,0.60784),     #218
        'Hills': ( 0.72549,0.47843,0.34118),        #136
        'Desert': (0.90196,0.78431,0.70980),        #206
        'Tundra': (0.75686,0.74510,0.68627),        #189
        'Mountain': (0.58824,0.58824,0.58824),      #150
    }

    # get all seg images
    segdata = []
    for path in segpaths:
        segdata.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        
    # count number of pixels for each class
    colours, counts = np.unique(segdata, return_counts=True)

    # sort classes by number of pixels
    colours = [x for _, x in sorted(zip(counts, colours), reverse=True)]
    counts = sorted(counts, reverse=True)

    # plot class distribution
    print(colours, counts)

    # plot class distribution
    plt.bar(classes.keys(), counts, color=classes.values())
    plt.ylabel('Count of pixels over all images')
    plt.title('Class Distribution')
    plt.show()