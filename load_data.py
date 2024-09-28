import tensorflow as tf
import os
import SimpleITK as sitk
import numpy as np
import shutil
import torch
import copy
from torch.utils.data import Dataset, DataLoader
import torch
from config import (DATASET_PATH, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE)

class CustomDataset(tf.keras.utils.Sequence):


    def __init__(self, 
                 n_train_patients=50, 
                 n_test_patients=10, 
                 task = 'La Cavity', 
                 transforms = None):
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
        super().__init__()
        self.data_path = DATASET_PATH
        self.n_train_patients = n_train_patients
        self.n_test_patients = n_test_patients
        self.task = task
        self.transform = transforms

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

            self.training_files= os.listdir(os.path.join(self.data_path, "Training Set"))[:self.n_train_patients]
            self.testing_files= os.listdir(os.path.join(self.data_path, "Testing Set"))[:self.n_test_patients]
            self.input_format = 'lgemri.nrrd'
            self.output_format = 'laendo.nrrd'
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.training_files)
        elif self.mode == 'val':
            return len(self.testing_files)

    def load_nrrd(self, filename_path):
        '''
        helper function: this function load nnrd file into a 3D matrix
            1. input: filenam_path
            2. output with shape = #slices x width x length
    '''
        data = sitk.ReadImage(filename_path)						# read in image
        # convert to 8 bit (0-255)
        data = sitk.Cast(sitk.RescaleIntensity(data), sitk.sitkUInt8)
        data = sitk.GetArrayFromImage(data)								# convert to numpy array
        return (data)
    
    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.mode == "train":
            name = self.training_files[idx]
            img_array = self.load_nrrd(os.path.join(self.data_path, "Training Set", self.training_files[idx], self.input_format))
            label_array = self.load_nrrd(os.path.join(self.data_path, "Training Set", self.training_files[idx], self.output_format))
        
        elif self.mode == 'val':
            name = self.testing_files[idx]
            img_array = self.load_nrrd(os.path.join(self.data_path, "Testing Set", self.testing_files[idx], self.input_format))
            label_array = self.load_nrrd(os.path.join(self.data_path, "Testing Set", self.testing_files[idx], self.output_format))

        img_array = img_array.reshape(1, np.shape(img_array)[0], np.shape(img_array)[1], np.shape(img_array)[2])
        
        temp = np.empty(shape=[2, np.shape(label_array)[0], np.shape(label_array)[1], np.shape(label_array)[2]])
        temp[0, :, :, :] = 1 - label_array
        temp[1, :, :, :] = label_array
        label_array = np.reshape(temp, newshape=[-1, np.shape(label_array)[0], np.shape(label_array)[1], np.shape(label_array)[2]])
        
        proccessed_out = {'name': name,
                          'image': img_array, 'label': label_array}
        if self.transform:
            if self.mode == "train":
                proccessed_out = self.transform[0](proccessed_out)
            elif self.mode == "val":
                proccessed_out = self.transform[1](proccessed_out)
            # elif self.mode == "test":
            #     proccessed_out = self.transform[2](proccessed_out)
            else:
                proccessed_out = self.transform(proccessed_out)
        return proccessed_out


def get_train_val_test_Dataloaders(train_transforms, val_transforms):
    """
    The utility function to generate splitted train, validation and test dataloaders

    Note: all the configs to generate dataloaders in included in "config.py"
    """

    dataset = CustomDataset(n_train_patients=1,
                            n_test_patients=1,
                            task = 'La Cavity',
                            transforms=[train_transforms, val_transforms])

    # Spliting dataset and building their respective DataLoaders
    train_set, val_set= copy.deepcopy(
        dataset), copy.deepcopy(dataset)
    
    train_set.set_mode('train')
    val_set.set_mode('val')

    train_dataloader = DataLoader(
        dataset=train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    val_dataloader = DataLoader(
        dataset=val_set, batch_size=VAL_BATCH_SIZE, shuffle=False)
    
    return train_dataloader, val_dataloader
    
        

