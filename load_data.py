'''
The load_data code is inspired by the Medium blog: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
and modified based on the original preprocess_data.py code including in the download dataset.
'''
import tensorflow as tf
import os
import SimpleITK as sitk
import numpy as np

class CustomDataset(tf.keras.utils.Sequence):

    '''
    function: init
        1. input: 
            - training_set_path: directory of the training path
            - testing_set_path: directory of the testing path
            - input_size = 112 (I take the default setting of inputsize in preprocess_data.py included in the LA dataset)
            - batch_size: size of batch for training
            - n_train_patients:
            - n_test_patients:
            - task: the task is in the list ['La Cavity', 'Panceras CT']
        2. output: none
    '''
    def __init__(self, training_path, testing_path, input_size = 112, 
                 batch_size = 16, n_train_patients = 10, n_test_patience = 5, task = 'LA Cavity'):
        
        self.training_path = training_path
        self.testing_path= testing_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.n_train_patients = n_train_patients
        self.n_test_patience = n_test_patience
        self.task = task

    '''
    function: the function will be called at the end of every epoch by fit method
    '''
    def on_epoch_end(self):
        pass

    '''
    function: generate one batch of data
        1. intput: self
        2. output: a batch of data (x, y)
    '''
    def __getitem__(self):
        pass

    '''
    function: get length of the whole data
        1. input: self
        2. output: length of the data
    '''
    def __len__(self):
        pass

    '''
    function: a helper function for loading input data
        1. input: self
        2. output: input data
    '''
    def __get_input(self):
        pass

    '''
    function: a helper function for loading output data
        1. input: self
        2. output: output data
    '''
    def __get_output(self):
        pass


