'''
The load_data code is inspired by the Medium blog: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
and modified based on the original preprocess_data.py code including in the download dataset.
'''
import tensorflow as tf
import os
import SimpleITK as sitk
import numpy as np
import shutil


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

    def __init__(self, training_path, testing_path, input_size=112, normalization='standard',
                 load_train=True, batch_size=1, n_train_patients=10, n_test_patients=5, task='LA Cavity'):

        super().__init__()

        self.training_path = training_path
        self.testing_path = testing_path
        self.input_size = input_size
        self.normalization = normalization
        self.load_train = load_train
        self.batch_size = batch_size
        self.n_train_patients = n_train_patients
        self.n_test_patients = n_test_patients
        self.task = task

        # create folder for data training
        if task == "La Cavity":
            dir1 = "LA_subset"
            dir2 = "La_subset/log"

            if os.path.exists(dir1):
                shutil.rmtree(dir1)
            if os.path.exists(dir2):
                shutil.rmtree(dir2)
            os.makedirs('LA_subset')
            os.makedirs('LA_subset/log')

        self.train_files = os.listdir(self.training_path)
        self.test_files = os.listdir(self.testing_path)

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

    def __getitem__(self, index):
        x, y = self.get_data()
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
        print(f"x: {np.shape(x_tensor)}")
        print(f"y: {np.shape(y_tensor)}")
        return x_tensor[index], y_tensor[index]


    def __len__(self):
        return (self.n_train_patients * np.shape(self.get_input())[0]) // self.batch_size

    '''
    helper function: this function load nnrd file into a 3D matrix
        1. input: filenam_path
        2. output with shape = #slices x width x length
    '''

    def load_nrrd(self, filename_path):
        data = sitk.ReadImage(filename_path)						# read in image
        # convert to 8 bit (0-255)
        data = sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8)
        data = sitk.GetArrayFromImage(data)								# convert to numpy array
        return (data)

    def get_data(self):

        input_samples = self.get_input()
        output_samples = self.get_output()
        
        return np.array(input_samples), np.stack([np.array(output_samples[:, :, :, 0]), np.array(output_samples[:, :, :, 1])], axis=-1)

    '''
    function: a helper function for loading input data
        1. input: self
        2. output: input data
    '''

    def get_input(self):
        images = []

        if self.load_train:
            number_of_patient = self.n_train_patients
        else:
            number_of_patient = self.n_test_patients

        for patient in range(number_of_patient):

            if self.load_train:
                raw_mri_sample = self.load_nrrd(os.path.join(
                    "../dataset/LA_dataset/Training Set", self.train_files[patient], 'lgemri.nrrd'))
            else:
                raw_mri_sample = self.load_nrrd(os.path.join(
                    "../dataset/LA_dataset/Testing Set", self.test_files[patient], 'lgemri.nrrd'))

            # move the dimension of slice into the last axis
            raw_mri_sample = np.rollaxis(
                raw_mri_sample, 0, 3)  # (width, height, slice)

            # based off the image size, and the specified input size, find coordinates to crop image
            # the input size of the model is 112
            midpoint = raw_mri_sample.shape[0]//2
            n11, n12 = midpoint - \
                int(self.input_size/2), midpoint + int(self.input_size/2)

            for slice in range(raw_mri_sample.shape[2]):
                images.append(raw_mri_sample[n11:n12, n11:n12, slice])

        images = np.array(images)
        images = np.reshape(images, newshape=[-1, self.input_size, self.input_size, 1])

        if self.normalization == 'standard':  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
            subset_mean = np.mean(images)
            subset_sd = np.std(images)

            images = (images-subset_mean)/subset_sd
        else:
            print('I dunno')
        
        self.total_samples = np.shape(images)[0]

        return images

    '''
    function: a helper function for loading output data
        1. input: self
        2. output: output data
    '''

    def get_output(self):
        labels = []

        if self.load_train:
            number_of_patient = self.n_train_patients
        else:
            number_of_patient = self.n_test_patients

        for patient in range(number_of_patient):

            if self.load_train:
                cavity_mri_sample = self.load_nrrd(os.path.join(
                    "../dataset/LA_dataset/Training Set", self.train_files[patient], 'lgemri.nrrd')) // 255.0
            else:
                cavity_mri_sample = self.load_nrrd(os.path.join(
                    "../dataset/LA_dataset/Testing Set", self.test_files[patient], 'laendo.nrrd')) // 255.0

            cavity_mri_sample = np.rollaxis(
                cavity_mri_sample, 0, 3)  # (width, height, slice)
            midpoint = cavity_mri_sample.shape[0]//2
            n11, n12 = midpoint - \
                int(self.input_size/2), midpoint + int(self.input_size/2)

            for slice in range(cavity_mri_sample.shape[2]):
                labels.append(cavity_mri_sample[n11:n12, n11:n12, slice])

        labels = np.array(labels)

        '''
        Encoding label to neural network output format
        When separated the binary segmentation task into 2 channels, the first channel is for segmentation of value '0'. And the second is for
        The channel 2 are still the same.
        '''
        temp = np.empty(
            shape=[labels.shape[0], self.input_size, self.input_size, 2])
        temp[:, :, :, 0] = 1 - labels
        temp[:, :, :, 1] = labels
        label = np.reshape(
            temp, newshape=[-1, self.input_size, self.input_size, 2])

        return label


'''
Note:
1. The code for get input and get output are quite similar as they were stacked together in the original code.
'''
