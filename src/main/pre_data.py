#coding=utf-8
'''
Created on 2017年9月1日
@author: zhengying
'''
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def pre_data():
    mnist = input_data.read_data_sets("/Users/zhengying/Documents/4_mechine_learning/dataset/MNIST/MNIST_data/", reshape=False)
    X_train, y_train           = mnist.train.images, mnist.train.labels
    X_validation, y_validation = mnist.validation.images, mnist.validation.labels
    X_test, y_test             = mnist.test.images, mnist.test.labels
    
    assert(len(X_train) == len(y_train))
    assert(len(X_validation) == len(y_validation))
    assert(len(X_test) == len(y_test))
    
    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_validation)))
    print("Test Set:       {} samples".format(len(X_test)))
    
    # Pad images with 0s
    X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
    return X_train,y_train,X_validation,y_validation,X_test,y_test