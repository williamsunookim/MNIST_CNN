import numpy as np 
import struct
from array import array

class MnistDataloader(object):
    def __init__(self, images_filepath,labels_filepath):
        self.images_filepath = images_filepath
        self.labels_filepath = labels_filepath
    
    #using the code of kaggle to read images and label from unknown type of file
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        images, label = self.read_images_labels(self.images_filepath, self.labels_filepath)
        self.preprocess(images) 
        y_train = []
        for i in label: # one-hot encoding
            tmp = np.zeros(10)
            tmp[i] = 1
            y_train.append(tmp)
        
        return (images, label, y_train)
    
    def preprocess(self, images): #min-max regularization
        for i in range(len(images)):
            image = np.array(images[i], dtype=np.float64)  # in order to divide by 255, should be float type
            image /= 255  # Reularization
            images[i] = image